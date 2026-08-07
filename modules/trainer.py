"""TFX Trainer Module - Training model DNN untuk UNSW-NB15 NIDS.

Modul ini mendefinisikan tiga fungsi utama yang dipanggil TFX Trainer:

- ``_build_model``: Membangun arsitektur DNN untuk klasifikasi biner.
- ``_input_fn``   : Membuat ``tf.data.Dataset`` dari transformed examples.
- ``run_fn``      : Entry point yang dipanggil TFX Trainer component.

Arsitektur model (dua-cabang input)::

    Numerik (31 fitur) ──┐
                          ├─ Concatenate ─ Dense(256)+BN+Drop(0.3)
    Kategorikal (3 emb) ─┘               ─ Dense(128)+BN+Drop(0.3)
                                          ─ Dense(64)+BN+Drop(0.2)
                                          ─ Dense(32)+ReLU
                                          ─ Dense(1, sigmoid)
"""

import os
from typing import List

import tensorflow as tf
from tensorflow import keras
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio


# ---------------------------------------------------------------------------
# Konstanta fitur dan hyperparameter default
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

EMBEDDING_DIM: int = 8
VOCAB_SIZE: int = 1000
BATCH_SIZE: int = 256
EPOCHS: int = 30


# ---------------------------------------------------------------------------
# Serving signature
# ---------------------------------------------------------------------------

def _get_serve_signature(
    model: keras.Model,
    tf_transform_output,
) -> tf.types.experimental.ConcreteFunction:
    """Buat serving signature yang menerima serialized ``tf.Example``.

    Signature ini memungkinkan TF Serving menerima request dalam format
    protobuf ``tf.Example`` yang sudah di-encode ke bytes.

    Args:
        model: Model Keras yang sudah selesai di-train.
        tf_transform_output: Output dari TensorFlow Transform (berisi
            feature spec dan transformation graph).

    Returns:
        ``tf.function`` yang siap dijadikan ``serving_default`` signature.
    """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'),
    ])
    def serve_tf_examples(serialized_tf_examples: tf.Tensor) -> dict:
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed = tf_transform_output.transform_raw_features(
            parsed_features
        )

        numeric_inputs = tf.stack(
            [transformed[k] for k in NUMERIC_TRANSFORMED_KEYS], axis=1
        )
        cat_inputs = tf.stack(
            [tf.cast(transformed[k], tf.int32) for k in CATEGORICAL_TRANSFORMED_KEYS],
            axis=1,
        )
        return {'predictions': model([numeric_inputs, cat_inputs])}

    return serve_tf_examples


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_model(
    numeric_size: int,
    categorical_size: int,
) -> keras.Model:
    """Bangun arsitektur DNN dua-cabang untuk klasifikasi biner.

    Dua cabang input digabung lalu melewati empat Dense layer dengan
    BatchNormalization dan Dropout untuk regularisasi.

    Args:
        numeric_size: Jumlah fitur numerik (setelah Z-score transform).
        categorical_size: Jumlah fitur kategorikal (setelah vocabulary
            encoding), setiap fitur diproses oleh Embedding layer
            terpisah.

    Returns:
        ``keras.Model`` yang siap di-compile dan di-train.
    """
    # -- Cabang numerik -------------------------------------------------------
    numeric_input = keras.layers.Input(
        shape=(numeric_size,),
        name='numeric_input',
        dtype=tf.float32,
    )

    # -- Cabang kategorikal (Embedding per fitur) -----------------------------
    cat_input = keras.layers.Input(
        shape=(categorical_size,),
        name='categorical_input',
        dtype=tf.int32,
    )
    embeddings = []
    for i in range(categorical_size):
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

    # -- Penggabungan dan fully-connected layers ------------------------------
    x = keras.layers.Concatenate(name='concat')([numeric_input] + embeddings)

    for units, dropout_rate, suffix in [
        (256, 0.3, 'block1'),
        (128, 0.3, 'block2'),
        (64, 0.2, 'block3'),
    ]:
        x = keras.layers.Dense(units, activation='relu', name=f'dense_{suffix}')(x)
        x = keras.layers.BatchNormalization(name=f'bn_{suffix}')(x)
        x = keras.layers.Dropout(dropout_rate, name=f'dropout_{suffix}')(x)

    x = keras.layers.Dense(32, activation='relu', name='dense_block4')(x)
    output = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    return keras.Model(
        inputs=[numeric_input, cat_input],
        outputs=output,
        name='nids_dnn',
    )


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
        file_pattern: List pola path file TFRecord (train atau eval).
        data_accessor: TFX ``DataAccessor`` untuk membaca file.
        tf_transform_output: TFX Transform output yang berisi schema.

    Returns:
        ``tf.data.Dataset`` yang di-repeat dan siap di-feed ke ``model.fit``.
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
# Entry point untuk TFX Trainer component
# ---------------------------------------------------------------------------

def run_fn(fn_args: FnArgs) -> None:
    """Entry point training yang dipanggil oleh TFX Trainer component.

    Alur kerja:
    1. Load TFTransformOutput untuk mendapatkan feature spec dan schema.
    2. Buat dataset train dan eval dari transformed TFRecord.
    3. Build, compile, dan train model DNN.
    4. Evaluasi pada eval set dan cetak hasil metrik.
    5. Simpan SavedModel beserta serving signature.

    Args:
        fn_args: ``FnArgs`` dari TFX Trainer berisi path file train/eval,
            transform output, serving model dir, jumlah steps, dll.
    """
    import tensorflow_transform as tft  # noqa: PLC0415 (lazy import untuk TFX)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    model = _build_model(
        numeric_size=len(NUMERIC_FEATURES),
        categorical_size=len(CATEGORICAL_FEATURES),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    eval_result = model.evaluate(
        eval_dataset,
        steps=fn_args.eval_steps,
        return_dict=True,
    )
    print('\n=== Evaluation Results ===')
    for metric_name, value in eval_result.items():
        print(f'  {metric_name}: {value:.4f}')

    signatures = {
        'serving_default': _get_serve_signature(model, tf_transform_output),
    }
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures,
    )
    print(f'\nModel saved to: {fn_args.serving_model_dir}')
