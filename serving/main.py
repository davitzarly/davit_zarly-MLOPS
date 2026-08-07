"""FastAPI Serving Application - UNSW-NB15 NIDS Model.

Endpoint:
- POST /predict: Prediksi apakah traffic jaringan normal atau serangan.
- GET /health: Health check endpoint.
- GET /metrics: Prometheus metrics endpoint.
"""

import os
import time
import logging
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Konfigurasi ===
MODEL_PATH = os.environ.get('MODEL_PATH', 'serving_model')

# === Prometheus Metrics ===
PREDICTION_COUNT = Counter(
    'nids_prediction_total',
    'Total predictions made',
    ['prediction_class']
)
REQUEST_LATENCY = Histogram(
    'nids_request_latency_seconds',
    'Request latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
ERROR_COUNT = Counter(
    'nids_error_total',
    'Total prediction errors'
)
MODEL_VERSION = Gauge(
    'nids_model_version',
    'Current model version info'
)

# === App ===
app = FastAPI(
    title='UNSW-NB15 NIDS API',
    description='Network Intrusion Detection System menggunakan model TensorFlow',
    version='1.0.0'
)

# === Model Loading ===
model = None
predict_fn = None

# Fitur yang harus dikirim dalam tf.train.Example, sesuai raw_feature_spec
# hasil Transform (harus sama persis dengan schema training).
FLOAT_FEATURES = [
    'dur', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'tcprtt', 'synack', 'ackdat',
]
INT_FEATURES = [
    'spkts', 'dpkts', 'sbytes', 'dbytes', 'sloss', 'dloss', 'swin',
    'stcpb', 'dtcpb', 'dwin', 'smean', 'dmean', 'trans_depth',
    'response_body_len', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports',
]
STRING_FEATURES = ['proto', 'service', 'state']


@app.on_event('startup')
async def load_model():
    """Load TensorFlow SavedModel (hasil Pusher) saat startup."""
    global model, predict_fn
    try:
        model_dir = MODEL_PATH
        if os.path.isdir(model_dir):
            subdirs = [
                os.path.join(model_dir, d)
                for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d))
            ]
            if subdirs:
                model_dir = max(subdirs, key=os.path.getmtime)

        logger.info(f"Loading model from: {model_dir}")
        model = tf.saved_model.load(model_dir)
        predict_fn = model.signatures['serving_default']
        logger.info("Model loaded successfully!")
        MODEL_VERSION.set(1.0)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


# === Request/Response Models ===
class NetworkTraffic(BaseModel):
    """Model input untuk prediksi traffic jaringan."""
    dur: float = Field(0.0, description='Duration of connection')
    proto: str = Field('tcp', description='Protocol type')
    service: str = Field('-', description='Network service')
    state: str = Field('FIN', description='Connection state')
    spkts: int = Field(0, description='Source to destination packet count')
    dpkts: int = Field(0, description='Destination to source packet count')
    sbytes: int = Field(0, description='Source to destination bytes')
    dbytes: int = Field(0, description='Destination to source bytes')
    rate: float = Field(0.0, description='Packets per second')
    sload: float = Field(0.0, description='Source bits per second')
    dload: float = Field(0.0, description='Destination bits per second')
    sloss: int = Field(0, description='Source retransmitted bytes')
    dloss: int = Field(0, description='Destination retransmitted bytes')
    sinpkt: float = Field(0.0, description='Source inter-packet arrival time mean')
    dinpkt: float = Field(0.0, description='Destination inter-packet arrival time mean')
    sjit: float = Field(0.0, description='Source jitter')
    djit: float = Field(0.0, description='Destination jitter')
    swin: int = Field(0, description='Source TCP window advert')
    stcpb: int = Field(0, description='Source TCP base sequence number')
    dtcpb: int = Field(0, description='Destination TCP base sequence number')
    dwin: int = Field(0, description='Destination TCP window advert')
    tcprtt: float = Field(0.0, description='TCP round trip time')
    synack: float = Field(0.0, description='TCP SYN+ACK round trip time')
    ackdat: float = Field(0.0, description='TCP ACK round trip time')
    smean: int = Field(0, description='Mean of source packet size')
    dmean: int = Field(0, description='Mean of destination packet size')
    trans_depth: int = Field(0, description='Transaction depth')
    response_body_len: int = Field(0, description='HTTP response body length')
    ct_src_dport_ltm: int = Field(0, description='Connection count source to dest port')
    ct_dst_sport_ltm: int = Field(0, description='Connection count dest to source port')
    is_ftp_login: int = Field(0, description='FTP login indicator')
    ct_ftp_cmd: int = Field(0, description='FTP command count')
    ct_flw_http_mthd: int = Field(0, description='HTTP methods count')
    is_sm_ips_ports: int = Field(0, description='Same source/destination ports')


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description='0=Normal, 1=Attack')
    confidence: float = Field(..., description='Confidence score (0-1)')
    label: str = Field(..., description='Label prediksi')


class BatchPredictionRequest(BaseModel):
    data: List[NetworkTraffic]


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total: int
    attack_count: int
    normal_count: int


def _to_tf_example(data: Dict[str, Any]) -> bytes:
    """Ubah satu record request menjadi serialized tf.train.Example.

    Urutan/tipe fitur harus sama persis dengan raw_feature_spec yang
    dipakai saat training (lihat modules/transform.py).
    """
    feature = {}
    for key in FLOAT_FEATURES:
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(data[key])])
        )
    for key in INT_FEATURES:
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(data[key])])
        )
    for key in STRING_FEATURES:
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[str(data[key]).encode('utf-8')])
        )
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def _run_inference(data: Dict[str, Any]):
    """Jalankan inference sungguhan lewat SavedModel signature."""
    serialized = _to_tf_example(data)
    result = predict_fn(examples=tf.constant([serialized]))
    # Nama output key mengikuti dict yang dikembalikan serve_tf_examples
    # di modules/trainer.py, yaitu 'predictions'.
    score = float(tf.squeeze(result['predictions']).numpy())
    prediction = 1 if score > 0.5 else 0
    confidence = score if prediction == 1 else 1.0 - score
    return prediction, confidence


@app.get('/health')
async def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None
    }


@app.get('/metrics')
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post('/predict', response_model=PredictionResponse)
async def predict_single(traffic: NetworkTraffic):
    """Prediksi single traffic jaringan menggunakan model TFX yang di-load."""
    start_time = time.time()
    try:
        if model is None or predict_fn is None:
            raise HTTPException(status_code=503, detail='Model not loaded')

        data = traffic.model_dump()
        prediction, confidence = _run_inference(data)
        label = 'Attack' if prediction == 1 else 'Normal'

        PREDICTION_COUNT.labels(prediction_class=label).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        logger.info(
            f"Prediction: {label} (confidence={confidence:.4f}, "
            f"latency={time.time() - start_time:.4f}s)"
        )

        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            label=label
        )

    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict/batch', response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Prediksi batch traffic jaringan menggunakan model TFX yang di-load."""
    if model is None or predict_fn is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    predictions = []
    attack_count = 0
    normal_count = 0

    for traffic in request.data:
        data = traffic.model_dump()
        prediction, confidence = _run_inference(data)
        label = 'Attack' if prediction == 1 else 'Normal'

        if prediction == 1:
            attack_count += 1
        else:
            normal_count += 1

        predictions.append({
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'label': label
        })
        PREDICTION_COUNT.labels(prediction_class=label).inc()

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        attack_count=attack_count,
        normal_count=normal_count
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
