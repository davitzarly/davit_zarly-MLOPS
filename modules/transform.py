"""TFX Transform Module - Preprocessing untuk UNSW-NB15 NIDS.

Modul ini mendefinisikan preprocessing_fn untuk TensorFlow Transform
yang mencakup normalisasi fitur numerik (Z-score) dan encoding
fitur kategorikal (Vocabulary-based one-hot encoding).
"""

import tensorflow as tf
import tensorflow_transform as tft

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

VOCAB_SIZE = 1000
OOV_SIZE = 1


def _normalize_numeric(inputs, outputs, key):
    """Normalisasi fitur numerik menggunakan Z-score.

    Args:
        inputs: Dict tensor input.
        outputs: Dict tensor output.
        key: Nama fitur.
    """
    value = tf.cast(inputs[key], tf.float32)
    value = tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))
    outputs[key + '_xf'] = tft.scale_to_z_score(value)


def _encode_categorical(inputs, outputs, key):
    """Encode fitur kategorikal menggunakan vocabulary + index.

    Args:
        inputs: Dict tensor input.
        outputs: Dict tensor output.
        key: Nama fitur.
    """
    cleaned = tf.strings.lower(tf.strings.strip(inputs[key]))
    indices = tft.compute_and_apply_vocabulary(
        cleaned,
        num_oov_buckets=OOV_SIZE,
        vocab_filename=key + '_vocab'
    )
    outputs[key + '_xf'] = tf.cast(indices, tf.int64)


def preprocessing_fn(inputs):
    """Transform function untuk TFX pipeline.

    Melakukan preprocessing terhadap seluruh fitur input:
    - Fitur numerik: Z-score normalization.
    - Fitur kategorikal: Vocabulary-based integer encoding.
    - Label: Diteruskan sebagai integer.

    Args:
        inputs: Dict dari tensor input dari ExampleGen.

    Returns:
        Dict dari tensor yang sudah ditransformasi.
    """
    outputs = {}

    for key in NUMERIC_FEATURES:
        _normalize_numeric(inputs, outputs, key)

    for key in CATEGORICAL_FEATURES:
        _encode_categorical(inputs, outputs, key)

    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
