"""TFX Tuner Module - Hyperparameter tuning untuk UNSW-NB15 NIDS.

Modul ini mendefinisikan tuner_fn untuk TFX Tuner component yang
mencari hyperparameter optimal menggunakan KerasTuner RandomSearch.

Hyperparameter yang dituning:
- learning_rate: [1e-4, 1e-3, 1e-2]
- num_hidden_layers: [2, 3, 4]
- hidden_units: [64, 128, 256]
- dropout_rate: [0.2, 0.3, 0.4]
"""

import keras_tuner as kt
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
BATCH_SIZE = 256


def _build_model(hp):
    """Bangun model dengan hyperparameter dari KerasTuner.

    Args:
        hp: KerasTuner HyperParameters object.

    Returns:
        Model Keras.
    """
    numeric_input = keras.layers.Input(
        shape=(len(NUMERIC_FEATURES),), name='numeric_input', dtype=keras.floatx()
    )
    cat_input = keras.layers.Input(
        shape=(len(CATEGORICAL_FEATURES),), name='categorical_input', dtype='int32'
    )

    embeddings = []
    for i in range(len(CATEGORICAL_FEATURES)):
        embedding = keras.layers.Embedding(
            input_dim=1000, output_dim=8
        )(keras.layers.Lambda(lambda x, idx=i: x[:, idx])(cat_input))
        embeddings.append(keras.layers.Flatten()(embedding))

    x = keras.layers.Concatenate()([numeric_input] + embeddings)

    num_layers = hp.Int('num_layers', min_value=2, max_value=4, default=3)
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=64, max_value=256, step=64, default=128)
        x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.4, step=0.1, default=0.3)
        x = keras.layers.Dropout(dropout)(x)

    output = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=[numeric_input, cat_input], outputs=output)


def _input_fn(file_pattern, data_accessor, tf_transform_output):
    """Membuat tf.data.Dataset dari transformed examples."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=BATCH_SIZE,
            label_key=LABEL_KEY
        ),
        tf_transform_output.transformed_metadata.schema
    ).repeat()


def tuner_fn(fn_args: FnArgs):
    """Fungsi tuner yang dipanggil oleh TFX Tuner component.

    Args:
        fn_args: Argumen dari TFX Tuner.

    Returns:
        Tuple (KerasTuner, model building function).
    """
    import tensorflow_transform as tft

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
        max_trials=5,
        overwrite=True,
        directory=fn_args.working_dir,
        project_name='nids_tuning',
    )

    tuner.search(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=5,
    )

    return tuner