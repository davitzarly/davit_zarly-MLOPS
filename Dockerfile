FROM tensorflow/serving:latest

# Salin SavedModel ke direktori model TF Serving
# Struktur: /models/<MODEL_NAME>/<version>/saved_model.pb
COPY ./serving_model/davit_zarly-model /models/davit_zarly-model

# Salin konfigurasi Prometheus
COPY ./config /model_config

# Nama model — harus sama dengan nama folder di /models/
ENV MODEL_NAME=davit_zarly-model

# Path ke file konfigurasi monitoring Prometheus
ENV MONITORING_CONFIG="/model_config/prometheus.config"

# Port REST API TF Serving (default 8501)
ENV PORT=8501

# Buat entrypoint script
RUN echo '#!/bin/bash \n\
\n\
env \n\
tensorflow_model_server \
  --port=8500 \
  --rest_api_port=${PORT} \
  --model_name=${MODEL_NAME} \
  --model_base_path=/models/${MODEL_NAME} \
  --monitoring_config_file=${MONITORING_CONFIG} \
  "$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

EXPOSE 8500 8501

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
