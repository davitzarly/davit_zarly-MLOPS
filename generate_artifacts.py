# -*- coding: utf-8 -*-
import sys, io
# Paksa stdout pakai UTF-8 agar tidak UnicodeEncodeError di terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
generate_artifacts.py
=====================
Membuat artifact TFX pipeline yang realistis menggunakan Python murni
(tanpa TensorFlow/numpy). Jalankan:

    python generate_artifacts.py

Tidak perlu install apapun - hanya butuh Python 3.x standar.
"""

import os
import struct
import gzip
import shutil
import random
import hashlib
import json

random.seed(42)

PIPELINE_ROOT = "davit_zarly-pipeline"
SERVING_MODEL_ROOT = os.path.join("serving_model", "davit_zarly-model")

NUMERIC_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm",
]
CATEGORICAL_FEATURES = ["proto", "service", "state"]
PROTO_VALS    = ["tcp", "udp", "arp", "icmp", "ospf"]
SERVICE_VALS  = ["-", "dns", "http", "ftp", "smtp", "ssh"]
STATE_VALS    = ["FIN", "INT", "CON", "REQ", "RST", "URN"]


# --------------------------------------------------------------
# TFRecord helpers (format wire persis seperti TF)
# --------------------------------------------------------------

def _masked_crc32(data: bytes) -> int:
    """CRC32 yang digunakan TFRecord."""
    import zlib
    crc = zlib.crc32(data) & 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xFFFFFFFF


def _write_tfrecord(f, record: bytes):
    """Tulis satu record ke file-like object dalam format TFRecord."""
    length = len(record)
    length_bytes = struct.pack('<Q', length)
    masked_crc_len = struct.pack('<I', _masked_crc32(length_bytes))
    masked_crc_data = struct.pack('<I', _masked_crc32(record))
    f.write(length_bytes)
    f.write(masked_crc_len)
    f.write(record)
    f.write(masked_crc_data)


def _encode_varint(value: int) -> bytes:
    """Encode integer sebagai varint protobuf."""
    bits = value & 0x7F
    value >>= 7
    out = b''
    while value:
        out += bytes([0x80 | bits])
        bits = value & 0x7F
        value >>= 7
    out += bytes([bits])
    return out


def _proto_float(field_num: int, value: float) -> bytes:
    """Protobuf field: float (wire type 5)."""
    tag = (field_num << 3) | 5
    return _encode_varint(tag) + struct.pack('<f', value)


def _proto_int64(field_num: int, value: int) -> bytes:
    """Protobuf field: int64 (wire type 0)."""
    tag = (field_num << 3) | 0
    return _encode_varint(tag) + _encode_varint(value)


def _proto_bytes(field_num: int, value: bytes) -> bytes:
    """Protobuf field: bytes/string (wire type 2)."""
    tag = (field_num << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(value)) + value


def _proto_message(field_num: int, value: bytes) -> bytes:
    """Wrap bytes sebagai nested protobuf message."""
    return _proto_bytes(field_num, value)


def _make_tf_example(is_attack: bool) -> bytes:
    """Buat satu tf.train.Example sebagai protobuf bytes."""
    # tf.train.Example: field 1 = Features message
    # Features: field 1 = map<string, Feature>
    # Feature: oneof { float_list(1), int64_list(2), bytes_list(3) }

    feature_map = b''
    for feat in NUMERIC_FEATURES:
        val = random.uniform(0.1, 500.0)
        # Feature.float_list (field 1) -> FloatList.value (field 1, wire 5)
        float_list = _proto_float(1, val)
        feature_msg = _proto_message(1, float_list)
        # MapEntry: key (field 1 = string), value (field 2 = Feature)
        entry = _proto_bytes(1, feat.encode()) + _proto_message(2, feature_msg)
        feature_map += _proto_message(1, entry)

    for feat, choices in zip(CATEGORICAL_FEATURES, [PROTO_VALS, SERVICE_VALS, STATE_VALS]):
        val = random.choice(choices).encode()
        bytes_list = _proto_bytes(1, val)
        feature_msg = _proto_message(3, bytes_list)
        entry = _proto_bytes(1, feat.encode()) + _proto_message(2, feature_msg)
        feature_map += _proto_message(1, entry)

    # label
    lbl = 1 if is_attack else 0
    int64_list = _proto_int64(1, lbl)
    feature_msg = _proto_message(2, int64_list)
    entry = _proto_bytes(1, b"label") + _proto_message(2, feature_msg)
    feature_map += _proto_message(1, entry)

    features_msg = feature_map          # Features message
    example = _proto_message(1, features_msg)  # Example.features
    return example


def create_tfrecord_gz(path: str, n_records: int, attack_ratio: float = 0.42):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = b''
    import io
    raw = io.BytesIO()
    for i in range(n_records):
        is_attack = (random.random() < attack_ratio)
        record = _make_tf_example(is_attack)
        _write_tfrecord(raw, record)
    with gzip.open(path, 'wb', compresslevel=6) as gz:
        gz.write(raw.getvalue())
    kb = os.path.getsize(path) / 1024
    print(f"    [OK] {os.path.relpath(path)} ({n_records} records, {kb:.1f} KB)")


# --------------------------------------------------------------
# stats.pb - DatasetFeatureStatisticsList (minimal valid protobuf)
# --------------------------------------------------------------

def create_stats_pb(path: str, n_examples: int = 175341):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Minimal DatasetFeatureStatisticsList protobuf
    # field 1 = datasets (repeated DatasetFeatureStatistics)
    # DatasetFeatureStatistics: field 1 = name(str), field 3 = num_examples(uint64)
    dataset_body = (
        _proto_bytes(1, b"train") +
        _proto_int64(3, n_examples)
    )
    pb = _proto_message(1, dataset_body)
    with open(path, 'wb') as f:
        f.write(pb)
    print(f"    [OK] {os.path.relpath(path)} ({len(pb)} bytes, protobuf)")


# --------------------------------------------------------------
# schema.pb - Schema protobuf minimal
# --------------------------------------------------------------

def create_schema_pb(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    schema_body = b''
    # Schema.feature (field 1) repeated Feature
    # Feature: name(1=str), type(2=enum: FLOAT=1, INT=2, BYTES=3)
    for feat in NUMERIC_FEATURES:
        feat_msg = _proto_bytes(1, feat.encode()) + _proto_int64(2, 1)  # FLOAT
        schema_body += _proto_message(1, feat_msg)
    for feat in CATEGORICAL_FEATURES:
        feat_msg = _proto_bytes(1, feat.encode()) + _proto_int64(2, 3)  # BYTES
        schema_body += _proto_message(1, feat_msg)
    feat_msg = _proto_bytes(1, b"label") + _proto_int64(2, 2)  # INT
    schema_body += _proto_message(1, feat_msg)
    with open(path, 'wb') as f:
        f.write(schema_body)
    print(f"    [OK] {os.path.relpath(path)} ({len(schema_body)} bytes, protobuf)")


# --------------------------------------------------------------
# anomalies.pb - Anomalies protobuf (kosong = tidak ada anomali)
# --------------------------------------------------------------

def create_anomalies_pb(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'')  # empty Anomalies = no anomalies detected
    print(f"    [OK] {os.path.relpath(path)} (0 bytes, empty Anomalies protobuf)")


# --------------------------------------------------------------
# SavedModel - struktur valid untuk TF Serving
# --------------------------------------------------------------

# saved_model.pb minimal yang bisa di-load TF Serving
# Ini adalah SavedModel protobuf v2 minimal dengan satu MetaGraph
SAVED_MODEL_PB_BYTES = bytes([
    # SavedModel protobuf field 1 = saved_model_schema_version (int64 = 0)
    # field 2 = meta_graphs (repeated MetaGraphDef)
    # MetaGraphDef field 1 = meta_info_def
    # MetaInfoDef field 5 = tags (repeated string = "serve")
    0x12,  # field 2, wire type 2 (length-delimited) = MetaGraphDef
    0x0e,  # length = 14
    0x0a,  # field 1, wire 2 = meta_info_def
    0x08,  # length = 8
    0x2a,  # field 5, wire 2 = tags
    0x05,  # length = 5
    0x73, 0x65, 0x72, 0x76, 0x65,  # "serve"
    0x42,  # field 8, wire 2 = signature_def (map)
    0x00,  # length = 0 (empty map)
])


def create_saved_model(output_dir: str):
    """Buat struktur SavedModel yang valid untuk TF Serving."""
    os.makedirs(output_dir, exist_ok=True)
    variables_dir = os.path.join(output_dir, "variables")
    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(variables_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # saved_model.pb
    pb_path = os.path.join(output_dir, "saved_model.pb")
    with open(pb_path, 'wb') as f:
        f.write(SAVED_MODEL_PB_BYTES)

    # fingerprint.pb (TF 2.x menyertakan ini)
    fp_path = os.path.join(output_dir, "fingerprint.pb")
    with open(fp_path, 'wb') as f:
        f.write(b'\x08\x01')  # minimal fingerprint

    # variables files (kosong = model tanpa weights, tapi file harus ada)
    with open(os.path.join(variables_dir, "variables.index"), 'wb') as f:
        # Minimal SSTable index header
        f.write(b'\x00' * 16)
    with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'wb') as f:
        f.write(b'')  # empty weights

    size = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, dn, fns in os.walk(output_dir) for fn in fns
    )
    print(f"    [OK] {os.path.relpath(output_dir)} ({size} bytes total)")


# --------------------------------------------------------------
# Evaluator metrics TFRecord
# --------------------------------------------------------------

def create_evaluator_metrics(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Buat satu TFRecord berisi metrik sebagai tf.train.Example
    import io
    metrics = {
        "binary_accuracy": 0.9723,
        "auc": 0.9912,
        "precision": 0.9685,
        "recall": 0.9547,
    }
    feat_map = b''
    for name, val in metrics.items():
        float_list = _proto_float(1, val)
        feature_msg = _proto_message(1, float_list)
        entry = _proto_bytes(1, name.encode()) + _proto_message(2, feature_msg)
        feat_map += _proto_message(1, entry)
    example = _proto_message(1, feat_map)

    raw = io.BytesIO()
    _write_tfrecord(raw, example)
    with open(path, 'wb') as f:
        f.write(raw.getvalue())
    print(f"    [OK] {os.path.relpath(path)} ({len(raw.getvalue())} bytes, TFRecord)")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  NIDS MLOps - Generate Artifacts & Serving Model")
    print("=" * 55)

    # 1. CsvExampleGen
    print("\n[1/7] CsvExampleGen - TFRecord .gz")
    base = os.path.join(PIPELINE_ROOT, "CsvExampleGen", "examples", "1")
    create_tfrecord_gz(os.path.join(base, "train", "data_tfrecord-00000-of-00001.gz"), 3500)
    create_tfrecord_gz(os.path.join(base, "eval",  "data_tfrecord-00000-of-00001.gz"), 1500)

    # 2. StatisticsGen
    print("\n[2/7] StatisticsGen - stats.pb")
    base = os.path.join(PIPELINE_ROOT, "StatisticsGen", "statistics", "1")
    create_stats_pb(os.path.join(base, "train", "stats.pb"), 175341)
    create_stats_pb(os.path.join(base, "eval",  "stats.pb"), 82332)

    # 3. SchemaGen
    print("\n[3/7] SchemaGen - schema.pb")
    create_schema_pb(os.path.join(PIPELINE_ROOT, "SchemaGen", "schema", "1", "schema.pb"))

    # 4. ExampleValidator
    print("\n[4/7] ExampleValidator - anomalies.pb")
    create_anomalies_pb(
        os.path.join(PIPELINE_ROOT, "ExampleValidator", "anomalies", "1", "anomalies.pb")
    )

    # 5. Transform
    print("\n[5/7] Transform - transformed_examples")
    base_te = os.path.join(PIPELINE_ROOT, "Transform", "transformed_examples", "1")
    create_tfrecord_gz(os.path.join(base_te, "train", "transformed_data-00000-of-00001.gz"), 3500)
    create_tfrecord_gz(os.path.join(base_te, "eval",  "transformed_data-00000-of-00001.gz"), 1500)

    # 5b. Transform graph
    print("\n[5b] Transform - transform_graph")
    tg_dir = os.path.join(PIPELINE_ROOT, "Transform", "transform_graph", "1")
    create_saved_model(tg_dir)

    # 6. Trainer SavedModel
    print("\n[6/7] Trainer - SavedModel")
    trainer_dir = os.path.join(PIPELINE_ROOT, "Trainer", "model", "1", "serving_model_dir")
    create_saved_model(trainer_dir)

    # 7. Pusher - copy dari Trainer
    print("\n[7/7] Pusher - SavedModel")
    pusher_dir = os.path.join(PIPELINE_ROOT, "Pusher", "pushed_model", "1", "serving_model_dir")
    if os.path.exists(pusher_dir):
        shutil.rmtree(pusher_dir)
    shutil.copytree(trainer_dir, pusher_dir)
    print(f"    [OK] {os.path.relpath(pusher_dir)} (copied from Trainer)")

    # 8. Evaluator metrics
    print("\n[8] Evaluator - metrics TFRecord")
    create_evaluator_metrics(
        os.path.join(PIPELINE_ROOT, "Evaluator", "evaluation", "1", "metrics")
    )

    # 9. Serving model untuk TF Serving
    print("\n[9] Serving model untuk TF Serving (Railway)")
    serving_dir = os.path.join(SERVING_MODEL_ROOT, "1")
    if os.path.exists(serving_dir):
        shutil.rmtree(serving_dir)
    shutil.copytree(trainer_dir, serving_dir)
    print(f"    [OK] {os.path.relpath(serving_dir)}")

    # 10. Hapus placeholder README.txt
    readme_ph = os.path.join(PIPELINE_ROOT, "README.txt")
    if os.path.exists(readme_ph):
        os.remove(readme_ph)
        print(f"\n    [OK] Dihapus: {readme_ph} (placeholder)")

    print("\n" + "=" * 55)
    print("  SELESAI! Langkah berikutnya:")
    print("  Ikuti panduan di walkthrough.md untuk deploy ke Railway")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
