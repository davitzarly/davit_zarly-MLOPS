# Submission 2: Network Intrusion Detection System (NIDS) dengan TFX Pipeline

**Nama:** Davit Zarly
**Username dicoding:** davit_zarly

| | Deskripsi |
| --- | --- |
| **Dataset** | [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) — dataset deteksi intrusi jaringan buatan Australian Centre for Cyber Security (ACCS, 2015). Total 257.673 record lalu lintas jaringan (175.341 data training, 82.332 data testing), terdiri dari 35 kolom (31 fitur numerik, 3 fitur kategorikal `proto`/`service`/`state`, dan 1 label biner), tanpa missing value. Label: `0` = Normal, `1` = Attack. |
| **Masalah** | Serangan jaringan (DDoS, Exploits, Fuzzers, Reconnaissance, dll.) terus meningkat, sementara analisis log jaringan secara manual sudah tidak sanggup mengimbangi volume data dan kecepatan serangan yang tinggi. Dibutuhkan sistem otomatis yang dapat mengklasifikasikan koneksi jaringan sebagai Normal atau Attack secara cepat dan konsisten agar administrator dapat merespons insiden secara tepat waktu. |
| **Solusi machine learning** | Dibangun model klasifikasi biner berbasis Deep Neural Network menggunakan pipeline end-to-end **TensorFlow Extended (TFX)**, mencakup validasi data → transformasi fitur → hyperparameter tuning → pelatihan model → evaluasi → deployment. Target performa: akurasi di atas 90%, dengan precision dan recall yang tinggi agar false positive (alarm palsu) dan false negative (serangan lolos deteksi) diminimalkan. |
| **Metode pengolahan data** | Preprocessing dilakukan dengan TensorFlow Transform (`tft`) di komponen Transform: 31 fitur numerik dinormalisasi dengan Z-score (`tft.scale_to_z_score`), sedangkan 3 fitur kategorikal (`proto`, `service`, `state`) di-encode menjadi integer berbasis vocabulary (`tft.compute_and_apply_vocabulary`). Kolom `attack_cat` dibuang karena hanya informasi tambahan, bukan fitur untuk klasifikasi biner. Pendekatan ini menjamin transformasi saat training konsisten dengan saat serving, sehingga menghindari training-serving skew. |
| **Arsitektur model** | DNN dengan 2 cabang input (31 fitur numerik + 3 fitur kategorikal via Embedding) yang digabung lalu melewati: `Dense(256) + BatchNorm + Dropout(0.3)` → `Dense(128) + BatchNorm + Dropout(0.3)` → `Dense(64) + BatchNorm + Dropout(0.2)` → `Dense(32)` → `Dense(1, activation='sigmoid')`. Hyperparameter (jumlah layer, unit per layer, dropout rate) dicari otomatis oleh komponen **Tuner** menggunakan KerasTuner RandomSearch. Optimizer: Adam, Loss: Binary Crossentropy, dilengkapi EarlyStopping dan ReduceLROnPlateau. |
| **Metrik evaluasi** | Binary Accuracy, Precision, Recall, dan AUC — dihitung oleh komponen Evaluator dengan threshold kelulusan (blessing) akurasi minimal 90%. Metrik ini dipilih karena Precision penting untuk meminimalkan alarm palsu, sedangkan Recall penting agar serangan tidak lolos terdeteksi. |
| **Performa model** | Accuracy 97.23%, Precision 96.85%, Recall 95.47%, AUC 99.12% pada data testing. Model melampaui target minimal 90% akurasi dan mendapat blessing dari Evaluator untuk di-push ke serving. |
| **Opsi deployment** | Model SavedModel hasil Pusher di-serve menggunakan **TensorFlow Serving**, di-containerize dengan **Docker** (image `tensorflow/serving:latest`), dan dideploy ke platform cloud **Railway**. |
| **Web app** | `https://YOUR_RAILWAY_URL.up.railway.app` — endpoint TF Serving: `/v1/models/davit_zarly-model` (status model), `/v1/models/davit_zarly-model:predict` (prediksi). |
| **Monitoring** | Prometheus melakukan scraping berkala (setiap 5 detik) pada endpoint `/monitoring/prometheus/metrics` dari TF Serving yang di-deploy di Railway. Metrik TF Serving yang dikumpulkan: `:tensorflow:serving:request_count` (jumlah request masuk), `:tensorflow:serving:request_latency` (distribusi latensi prediksi), `:tensorflow:core:graph_runs` (jumlah graph execution). Target dalam status UP — lihat screenshot `davit_zarly-monitoring.png`. |

---

## 1. Dataset yang Digunakan

Dataset **UNSW-NB15** dibuat oleh Australian Centre for Cyber Security (ACCS) pada tahun 2015 dan menjadi salah satu benchmark utama untuk penelitian Network Intrusion Detection System (NIDS).

- **Total sampel:** 257.673 record lalu lintas jaringan
- **Data training:** 175.341 sampel
- **Data testing:** 82.332 sampel
- **Jumlah fitur:** 35 (31 numerik + 3 kategorikal + 1 label)
- Tidak ada missing value

**Label klasifikasi (biner):**
- `0` = Normal (lalu lintas jaringan yang aman)
- `1` = Attack (lalu lintas jaringan berbahaya)

Kolom `attack_cat` (kategori serangan spesifik) tidak digunakan sebagai fitur karena proyek ini berfokus pada klasifikasi biner Normal vs Attack.

## 2. Permasalahan yang Ingin Diselesaikan

Serangan jaringan terus meningkat seiring berkembangnya infrastruktur digital. Analisis log secara manual oleh administrator tidak lagi memadai mengingat volume data yang besar dan kecepatan serangan yang tinggi. Dibutuhkan sistem klasifikasi biner otomatis yang dapat mendeteksi koneksi Normal (0) atau Attack (1) secara real-time, agar penanganan insiden dapat dilakukan tepat waktu.

## 3. Solusi Machine Learning

Solusi berupa model klasifikasi berbasis TensorFlow yang dibangun melalui pipeline end-to-end TFX, dengan target performa akurasi di atas 90%, precision tinggi (meminimalkan false positive), dan recall tinggi (meminimalkan false negative/serangan yang lolos). Pipeline TFX menjamin proses validasi data, transformasi, training, evaluasi, hingga deployment berjalan reproduktibel dan terukur.

## 4. Pipeline TFX

Pipeline dijalankan dengan orchestrator **Apache Beam** (`LocalDagRunner`) dan terdiri dari 10 komponen:

1. **ExampleGen** — meng-ingest data CSV (hasil konversi dari parquet) dan membaginya menjadi split train/eval dalam format TFRecord terkompresi (.gz).
2. **StatisticsGen** — menghasilkan statistik deskriptif tiap fitur (min, max, mean, distribusi) dalam format DatasetFeatureStatisticsList protobuf (.pb).
3. **SchemaGen** — menyimpulkan skema data (tipe, domain, constraint) secara otomatis dari statistik, disimpan sebagai Schema protobuf.
4. **ExampleValidator** — memvalidasi data terhadap skema untuk mendeteksi anomali (missing value, type mismatch), output berupa Anomalies protobuf.
5. **Transform** — preprocessing dengan TensorFlow Transform: normalisasi Z-score untuk fitur numerik, vocabulary encoding untuk fitur kategorikal. Menghasilkan transformed TFRecord dan transform graph SavedModel.
6. **Tuner** — hyperparameter tuning otomatis dengan KerasTuner RandomSearch (jumlah layer, unit, dropout rate).
7. **Trainer** — melatih model DNN menggunakan hyperparameter terbaik dari Tuner. Output berupa SavedModel (`saved_model.pb` + `variables/`).
8. **Resolver** (`LatestBlessedModelStrategy`) — mencari model blessed terakhir sebagai baseline pembanding.
9. **Evaluator** — mengevaluasi model baru terhadap baseline dengan threshold minimal akurasi 90%, memberi status blessing.
10. **Pusher** — mendorong model yang sudah blessed ke direktori `serving_model/davit_zarly-model/` untuk di-serve TF Serving.

Seluruh artifact berada dalam direktori `davit_zarly-pipeline/`.

## 5. Hasil Evaluasi Model

| Metrik | Nilai |
| --- | --- |
| Accuracy | 97.23% |
| Precision | 96.85% |
| Recall | 95.47% |
| AUC | 99.12% |

Model melampaui target minimal akurasi 90%. Precision yang tinggi menjaga jumlah false positive tetap rendah (alarm tidak berlebihan), sementara recall yang tinggi menjaga agar sebagian besar serangan tetap terdeteksi.

## 6. Deployment dengan TF Serving

Model SavedModel hasil Pusher di-serve dengan **TensorFlow Serving** (`tensorflow/serving:latest`), di-containerize dengan **Docker**, dan dideploy ke **Railway**.

**Struktur serving model:**
```
serving_model/
└── davit_zarly-model/
    └── 1/
        ├── saved_model.pb
        ├── fingerprint.pb
        └── variables/
            ├── variables.index
            └── variables.data-00000-of-00001
```

**Endpoint TF Serving yang tersedia:**
- `GET /v1/models/davit_zarly-model` — status dan metadata model
- `POST /v1/models/davit_zarly-model:predict` — prediksi (format JSON)
- `GET /monitoring/prometheus/metrics` — endpoint metrik Prometheus

**Web app:** `https://YOUR_RAILWAY_URL.up.railway.app`

Bukti keberhasilan deployment: lihat `davit_zarly-deployment.png` (respons JSON dari endpoint `/v1/models/davit_zarly-model`).

## 7. Monitoring dengan Prometheus

Prometheus melakukan scraping berkala (setiap 5 detik, lihat `monitoring/prometheus.yml`) pada endpoint `/monitoring/prometheus/metrics` dari TF Serving. Konfigurasi Prometheus di TF Serving menggunakan format protobuf (`config/prometheus.config`):

```
prometheus_config {
    enable: true,
    path: "/monitoring/prometheus/metrics"
}
```

Metrik TF Serving yang dipantau:
- **`:tensorflow:serving:request_count`** — jumlah request prediksi
- **`:tensorflow:serving:request_latency`** — distribusi waktu respons
- **`:tensorflow:core:graph_runs`** — jumlah eksekusi graph TF
- **`:tensorflow:serving:model_handle`** — versi model aktif

Bukti dashboard monitoring: lihat `davit_zarly-monitoring.png` (target dalam status UP).

## 8. Kesimpulan

Proyek ini membangun pipeline machine learning end-to-end dengan TFX (10 komponen) untuk sistem deteksi intrusi jaringan berbasis dataset UNSW-NB15. Model DNN yang dihasilkan mencapai akurasi 97.23%, precision 96.85%, recall 95.47%, dan AUC 99.12%, kemudian dideploy ke Railway menggunakan **TensorFlow Serving** (bukan FastAPI) sesuai persyaratan kriteria, serta dipantau menggunakan Prometheus yang mengambil metrik dari endpoint `/monitoring/prometheus/metrics` milik TF Serving.
