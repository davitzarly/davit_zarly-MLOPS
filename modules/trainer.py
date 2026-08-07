"""TFX Trainer Module - Training model DNN untuk UNSW-NB15 NIDS.

Modul ini mendefinisikan:
- _build_model(): Arsitektur DNN untuk klasifikasi biner.
- _input_fn(): Pembuatan tf.data.Dataset dari transformed examples.
- run_fn(): Fungsi utama yang dipanggil TFX Trainer component.

Arsitektur model:
    Input (31 numerik + 3 kategorikal via embedding)
    -> Dense 256 + BatchNorm + Dropout(0.3)
    -> Dense 128 + BatchNorm + Dropout(0.3)
    -> Dense 64  + BatchNorm + Dropout(0.2)
    -> Dense 32  + ReLU
    -> Dense 1   + Sigmoid (Output)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio

NUMERIC_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
    'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports'
]

CATEGORICAL_FEATURES = ['proto', 'service', 'state']
LABEL_KEY = 'label'
NUMERIC_TRANSFORMED_KEYS = [f + '_xf' for f in NUMERIC_FEATURES]
CATEGORICAL_TRANSFORMED_KEYS = [f + '_xf' for f in CATEGORICAL_FEATURES]
EMBEDDING_DIM = 8
VOCAB_SIZE = 1000
BATCH_SIZE = 256
EPOCHS = 30


def _get_serve_signature(model, tf_transform_output):
    """Buat serving signature untuk SavedModel.

    Args:
        model: Model Keras yang sudah di-train.
        tf_transform_output: Output dari TensorFlow Transform.

    Returns:
        tf.function yang menerima serialized tf.Example.
    """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = tf_transform_output.transform_raw_features(
            parsed_features
        )
        numeric_inputs = tf.stack([
            transformed_features[key] for key in NUMERIC_TRANSFORMED_KEYS
        ], axis=1)
        cat_inputs = tf.stack([
            tf.cast(transformed_features[key], tf.int32)
            for key in CATEGORICAL_TRANSFORMED_KEYS
        ], axis=1)
        return {'predictions': model([numeric_inputs, cat_inputs])}

    return serve_tf_examples


def _build_model(numeric_size, categorical_size):
    """Bangun arsitektur model DNN untuk klasifikasi biner.

    Args:
        numeric_size: Jumlah fitur numerik.
        categorical_size: Jumlah fitur kategorikal.

    Returns:
        Model Keras.
    """
    numeric_input = keras.layers.Input(
        shape=(numeric_size,), name='numeric_input', dtype=tf.float32
    )
    cat_input = keras.layers.Input(
        shape=(categorical_size,), name='categorical_input', dtype=tf.int32
    )

    embeddings = []
    for i in range(categorical_size):
        embedding = keras.layers.Embedding(
            input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM
        )(keras.layers.Lambda(lambda x, idx=i: x[:, idx])(cat_input))
        embeddings.append(keras.layers.Flatten()(embedding))

    x = keras.layers.Concatenate()([numeric_input] + embeddings)

    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(32, activation='relu')(x)

    output = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs=[numeric_input, cat_input], outputs=output)


def _input_fn(file_pattern, data_accessor, tf_transform_output):
    """Membuat tf.data.Dataset dari transformed examples.

    Args:
        file_pattern: Pattern file TFRecord.
        data_accessor: TFX DataAccessor.
        tf_transform_output: Output dari TensorFlow Transform.

    Returns:
        tf.data.Dataset.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=BATCH_SIZE,
            label_key=LABEL_KEY
        ),
        tf_transform_output.transformed_metadata.schema
    ).repeat()


def run_fn(fn_args: FnArgs):
    """Fungsi utama training yang dipanggil oleh TFX Trainer.

    Args:
        fn_args: Argumen dari TFX Trainer (train_files, eval_files,
            transform_output, serving_model_dir, dll).
    """
    import tensorflow_transform as tft

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    model = _build_model(len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
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
        eval_dataset, steps=fn_args.eval_steps, return_dict=True
    )
    print('\n=== Evaluation Results ===')
    for metric_name, value in eval_result.items():
        print(f'  {metric_name}: {value:.4f}')

    signatures = {
        'serving_default': _get_serve_signature(model, tf_transform_output),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
    print(f'\nModel saved to {fn_args.serving_model_dir}')
