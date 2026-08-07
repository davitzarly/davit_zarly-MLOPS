"""TFX Tuner Module - Hyperparameter tuning untuk UNSW-NB15 NIDS.

Modul ini mendefinisikan ``tuner_fn`` untuk TFX Tuner component yang
mencari hyperparameter optimal menggunakan KerasTuner ``RandomSearch``.

Hyperparameter yang dituning:

+--------------------+----------------------------------+---------+
| Hyperparameter     | Rentang nilai                    | Default |
+====================+==================================+=========+
| ``num_layers``     | 2, 3, 4                          | 3       |
+--------------------+----------------------------------+---------+
| ``units_i``        | 64, 128, 192, 256 (per layer)    | 128     |
+--------------------+----------------------------------+---------+
| ``dropout_i``      | 0.1, 0.2, 0.3, 0.4 (per layer)  | 0.3     |
+--------------------+----------------------------------+---------+
| ``learning_rate``  | 1e-4, 5e-4, 1e-3, 5e-3          | 1e-3    |
+--------------------+----------------------------------+---------+
"""

from typing import List

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio


# ---------------------------------------------------------------------------
# Konstanta fitur
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: List[str] = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
    'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports',
]

CATEGORICAL_FEATURES: List[str] = ['proto', 'service', 'state']
LABEL_KEY: str = 'label'

NUMERIC_TRANSFORMED_KEYS: List[str] = [f + '_xf' for f in NUMERIC_FEATURES]
CATEGORICAL_TRANSFORMED_KEYS: List[str] = [f + '_xf' for f in CATEGORICAL_FEATURES]

BATCH_SIZE: int = 256
EMBEDDING_DIM: int = 8
VOCAB_SIZE: int = 1000
MAX_TUNER_TRIALS: int = 5
TUNER_EPOCHS: int = 5


# ---------------------------------------------------------------------------
# Model builder dengan hyperparameter
# ---------------------------------------------------------------------------

def _build_model(hp: kt.HyperParameters) -> keras.Model:
    """Bangun model DNN dengan hyperparameter dari KerasTuner.

    Arsitektur identik dengan ``trainer.py``, namun jumlah layer,
    jumlah unit, dropout rate, dan learning rate bervariasi sesuai
    yang dipilih KerasTuner pada setiap trial.

    Args:
        hp: ``HyperParameters`` object dari KerasTuner yang menyimpan
            nilai hyperparameter untuk trial saat ini.

    Returns:
        ``keras.Model`` yang sudah di-compile, siap untuk ``tuner.search``.
    """
    # -- Input layers ---------------------------------------------------------
    numeric_input = keras.layers.Input(
        shape=(len(NUMERIC_FEATURES),),
        name='numeric_input',
        dtype=keras.floatx(),
    )
    cat_input = keras.layers.Input(
        shape=(len(CATEGORICAL_FEATURES),),
        name='categorical_input',
        dtype='int32',
    )

    # -- Embedding per fitur kategorikal --------------------------------------
    embeddings = []
    for i in range(len(CATEGORICAL_FEATURES)):
        slice_layer = keras.layers.Lambda(
            lambda x, idx=i: x[:, idx],
            name=f'cat_slice_{i}',
        )(cat_input)
        emb = keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            name=f'embedding_{i}',
        )(slice_layer)
        embeddings.append(keras.layers.Flatten(name=f'flatten_{i}')(emb))

    # -- Penggabungan dan hidden layers (jumlah ditentukan tuner) -------------
    x = keras.layers.Concatenate(name='concat')([numeric_input] + embeddings)

    num_layers = hp.Int('num_layers', min_value=2, max_value=4, default=3)
    for i in range(num_layers):
        units = hp.Int(
            f'units_{i}',
            min_value=64,
            max_value=256,
            step=64,
            default=128,
        )
        dropout_rate = hp.Float(
            f'dropout_{i}',
            min_value=0.1,
            max_value=0.4,
            step=0.1,
            default=0.3,
        )
        x = keras.layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
        x = keras.layers.BatchNormalization(name=f'bn_{i}')(x)
        x = keras.layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    output = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    # -- Compile dengan learning rate yang juga di-tune -----------------------
    learning_rate = hp.Choice(
        'learning_rate',
        values=[1e-4, 5e-4, 1e-3, 5e-3],
        default=1e-3,
    )
    model = keras.Model(
        inputs=[numeric_input, cat_input],
        outputs=output,
        name='nids_dnn_tunable',
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _input_fn(
    file_pattern: List[str],
    data_accessor,
    tf_transform_output,
) -> tf.data.Dataset:
    """Membuat ``tf.data.Dataset`` dari file TFRecord transformed examples.

    Args:
        file_pattern: List pola path file TFRecord.
        data_accessor: TFX ``DataAccessor`` untuk membaca data.
        tf_transform_output: TFX Transform output yang berisi schema.

    Returns:
        ``tf.data.Dataset`` yang di-repeat, siap untuk ``tuner.search``.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=BATCH_SIZE,
            label_key=LABEL_KEY,
        ),
        tf_transform_output.transformed_metadata.schema,
    ).repeat()


# ---------------------------------------------------------------------------
# Entry point untuk TFX Tuner component
# ---------------------------------------------------------------------------

def tuner_fn(fn_args: FnArgs) -> kt.Tuner:
    """Entry point tuning yang dipanggil oleh TFX Tuner component.

    Menjalankan ``RandomSearch`` selama ``MAX_TUNER_TRIALS`` trial,
    masing-masing dilatih selama ``TUNER_EPOCHS`` epoch, mengoptimasi
    ``val_loss`` (minimisasi).

    Args:
        fn_args: ``FnArgs`` dari TFX Tuner berisi path file train/eval,
            transform output, working dir, jumlah steps, dll.

    Returns:
        ``kt.Tuner`` yang sudah selesai ``search`` dan siap diambil
        best hyperparameter-nya oleh TFX.
    """
    import tensorflow_transform as tft  # noqa: PLC0415

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    tuner = kt.RandomSearch(
        _build_model,
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=MAX_TUNER_TRIALS,
        overwrite=True,
        directory=fn_args.working_dir,
        project_name='nids_tuning',
    )

    tuner.search(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=TUNER_EPOCHS,
    )

    return tuner