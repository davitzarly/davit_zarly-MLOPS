"""TFX Transform Module - Preprocessing untuk UNSW-NB15 NIDS.

Modul ini mendefinisikan ``preprocessing_fn`` untuk TensorFlow Transform
yang mencakup:
- Normalisasi fitur numerik dengan Z-score (``tft.scale_to_z_score``)
- Encoding fitur kategorikal berbasis vocabulary integer
  (``tft.compute_and_apply_vocabulary``)

Semua transformasi dijalankan di dalam graph TF sehingga konsisten
antara training dan serving (menghindari training-serving skew).
"""

import tensorflow as tf
import tensorflow_transform as tft


# ---------------------------------------------------------------------------
# Konstanta fitur
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
    'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports',
]

CATEGORICAL_FEATURES = ['proto', 'service', 'state']

LABEL_KEY = 'label'
VOCAB_SIZE = 1000
OOV_SIZE = 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_numeric(inputs: dict, outputs: dict, key: str) -> None:
    """Normalisasi satu fitur numerik menggunakan Z-score.

    Nilai non-finite (NaN/Inf) diganti 0 sebelum normalisasi agar
    pipeline tidak crash pada data yang kotor.

    Args:
        inputs: Dict tensor mentah dari ExampleGen.
        outputs: Dict tensor hasil transformasi (dimodifikasi in-place).
        key: Nama fitur numerik yang akan dinormalisasi.
    """
    value = tf.cast(inputs[key], tf.float32)
    value = tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))
    outputs[key + '_xf'] = tft.scale_to_z_score(value)


def _encode_categorical(inputs: dict, outputs: dict, key: str) -> None:
    """Encode satu fitur kategorikal menjadi integer berbasis vocabulary.

    String di-lowercase dan di-strip sebelum di-encode agar variasi
    penulisan ('TCP', 'tcp', ' tcp ') diperlakukan sama.

    Args:
        inputs: Dict tensor mentah dari ExampleGen.
        outputs: Dict tensor hasil transformasi (dimodifikasi in-place).
        key: Nama fitur kategorikal yang akan di-encode.
    """
    cleaned = tf.strings.lower(tf.strings.strip(inputs[key]))
    indices = tft.compute_and_apply_vocabulary(
        cleaned,
        num_oov_buckets=OOV_SIZE,
        vocab_filename=key + '_vocab',
    )
    outputs[key + '_xf'] = tf.cast(indices, tf.int64)


# ---------------------------------------------------------------------------
# Entry point untuk TFX Transform component
# ---------------------------------------------------------------------------

def preprocessing_fn(inputs: dict) -> dict:
    """Fungsi preprocessing utama yang dipanggil oleh TFX Transform.

    Melakukan preprocessing terhadap seluruh fitur input:
    - Fitur numerik  : Z-score normalization -> ``<feat>_xf``
    - Fitur kategorikal: Vocabulary integer encoding -> ``<feat>_xf``
    - Label          : Diteruskan sebagai ``tf.int64``

    Args:
        inputs: Dict dari tensor input mentah yang berasal dari ExampleGen.

    Returns:
        Dict dari tensor yang sudah ditransformasi, siap dipakai Trainer.
    """
    outputs = {}

    for key in NUMERIC_FEATURES:
        _normalize_numeric(inputs, outputs, key)

    for key in CATEGORICAL_FEATURES:
        _encode_categorical(inputs, outputs, key)

    outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
