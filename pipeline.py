"""TFX Pipeline - End-to-end ML pipeline untuk UNSW-NB15 NIDS.

Komponen pipeline:
1. ExampleGen     - Ingest data CSV
2. StatisticsGen  - Generate statistik deskriptif
3. SchemaGen      - Infer schema otomatis
4. ExampleValidator - Validasi data terhadap schema
5. Transform      - Preprocessing data (TF Transform)
6. Tuner          - Hyperparameter tuning
7. Trainer        - Training model DNN
8. Resolver       - Cari model blessed terakhir
9. Evaluator      - Evaluasi model (blessing)
10. Pusher        - Push model ke serving directory

Pipeline orchestrator: Apache Beam (BeamDagRunner)
"""

import os

from tfx import v1 as tfx
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
    ResolverNode,
    Tuner,
)
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# ============================================================
# Konfigurasi Pipeline
# ============================================================

PIPELINE_NAME = 'davit_zarly-pipeline'
PIPELINE_ROOT = os.path.join('davit_zarly-pipeline')
DATA_ROOT = os.path.join('data')
SERVING_MODEL_DIR = os.path.join('serving_model')

os.makedirs(PIPELINE_ROOT, exist_ok=True)
os.makedirs(SERVING_MODEL_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_ROOT, 'train.csv')
EVAL_CSV = os.path.join(DATA_ROOT, 'eval.csv')


def _ensure_csv_data():
    """Konversi parquet ke CSV jika belum ada.

    Dataset parquet dikonversi ke format CSV karena TFX CsvExampleGen
    memerlukan input berupa file CSV. Kolom 'attack_cat' dihapus karena
    hanya merupakan informasi tambahan, bukan fitur untuk model.
    """
    import pandas as pd

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(EVAL_CSV):
        print('Konversi data dari Parquet ke CSV...')
        df_train = pd.read_parquet(
            os.path.join(DATA_ROOT, 'UNSW_NB15_training-set.parquet')
        )
        df_test = pd.read_parquet(
            os.path.join(DATA_ROOT, 'UNSW_NB15_testing-set.parquet')
        )

        for col in df_train.select_dtypes(include=['category']).columns:
            df_train[col] = df_train[col].astype(str)
        for col in df_test.select_dtypes(include=['category']).columns:
            df_test[col] = df_test[col].astype(str)

        if 'attack_cat' in df_train.columns:
            df_train = df_train.drop(columns=['attack_cat'])
        if 'attack_cat' in df_test.columns:
            df_test = df_test.drop(columns=['attack_cat'])

        df_train.to_csv(TRAIN_CSV, index=False)
        df_test.to_csv(EVAL_CSV, index=False)
        print(f'Training CSV: {len(df_train)} rows -> {TRAIN_CSV}')
        print(f'Eval CSV:     {len(df_test)} rows -> {EVAL_CSV}')
    else:
        print('CSV data sudah tersedia.')


def create_pipeline():
    """Buat dan kembalikan TFX pipeline lengkap.

    Returns:
        tfx.dsl.Pipeline: Pipeline TFX yang siap dijalankan.
    """
    _ensure_csv_data()

    # ========================================
    # 1. ExampleGen - Ingest data
    # ========================================
    example_gen = CsvExampleGen(input_base=DATA_ROOT)

    # ========================================
    # 2. StatisticsGen - Generate statistics
    # ========================================
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # ========================================
    # 3. SchemaGen - Infer schema
    # ========================================
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics']
    )

    # ========================================
    # 4. ExampleValidator - Validasi data
    # ========================================
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )

    # ========================================
    # 5. Transform - Preprocessing
    # ========================================
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath('modules/transform.py'),
    )

    # ========================================
    # 6. Tuner - Hyperparameter Tuning
    # ========================================
    tuner = Tuner(
        module_file=os.path.abspath('modules/tuner.py'),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=500),
        eval_args=trainer_pb2.EvalArgs(num_steps=100),
    )

    # ========================================
    # 7. Trainer - Training model
    # ========================================
    trainer = Trainer(
        module_file=os.path.abspath('modules/trainer.py'),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        # Menggunakan hyperparameter dari Tuner
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=200),
    )

    # ========================================
    # 8. Resolver - Cari model blessed terakhir
    # ========================================
    model_resolver = ResolverNode(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id('latest_blessed_model_resolver')

    # ========================================
    # 9. Evaluator - Evaluasi & Blessing
    # ========================================
    eval_config = tfx.proto.evaluator_pb2.EvalConfig(
        model_specs=[tfx.proto.evaluator_pb2.ModelSpec(label_key='label')],
        slicing_specs=[tfx.proto.evaluator_pb2.SlicingSpec()],
        metrics_specs=[tfx.proto.evaluator_pb2.MetricsSpec(
            metrics=[
                tfx.proto.evaluator_pb2.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfx.proto.evaluator_pb2.MetricThreshold(
                        value_threshold=tfx.proto.evaluator_pb2.ValueThreshold(
                            lower_bound={'value': 0.90}
                        ),
                        change_threshold=tfx.proto.evaluator_pb2.ChangeThreshold(
                            direction=tfx.proto.evaluator_pb2.ChangeDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.01},
                        ),
                    )
                ),
                tfx.proto.evaluator_pb2.MetricConfig(class_name='AUC'),
                tfx.proto.evaluator_pb2.MetricConfig(class_name='Precision'),
                tfx.proto.evaluator_pb2.MetricConfig(class_name='Recall'),
            ]
        )],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config,
    )

    # ========================================
    # 10. Pusher - Push ke serving
    # ========================================
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.pusher_pb2.PushDestination(
            filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        ),
    )

    # ========================================
    # Assemble Pipeline
    # ========================================
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=components,
        metadata_connection_config=(
            tfx.orchestration.metadata.sqlite_metadata_connection_config(
                os.path.join(PIPELINE_ROOT, 'metadata.sqlite')
            )
        ),
    )


if __name__ == '__main__':
    print('=' * 60)
    print('  TFX Pipeline - UNSW-NB15 Network Intrusion Detection')
    print('=' * 60)

    pipeline = create_pipeline()
    BeamDagRunner().run(pipeline)

    print()
    print('=' * 60)
    print('  Pipeline selesai!')
    print(f'  Serving model: {SERVING_MODEL_DIR}')
    print('=' * 60)
