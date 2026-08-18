# MLOps Implementation — SupportPilot AI

## Overview

SupportPilot AI menerapkan praktik MLOps untuk memastikan model Machine
Learning dapat dikembangkan, dievaluasi, di-versioning, diuji, dan
dijalankan secara reproducible pada environment production.

MLOps pipeline pada project ini mencakup:

```text
Model Experimentation
        ↓
Model Evaluation
        ↓
Model Selection
        ↓
Model Versioning
        ↓
Production Model Artifact
        ↓
Automated Testing
        ↓
Containerization
        ↓
Deployment
        ↓
Health Monitoring
```

---

# 1. Experimentation and Model Comparison

SupportPilot AI membandingkan tiga pendekatan model:

1. Logistic Regression
2. Linear SVM
3. DistilBERT

Model dibandingkan menggunakan Validation Set.

Metric utama untuk model selection adalah:

```text
Macro F1
```

Hasil validation:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 98.2378% | 98.2562% |
| Linear SVM | 98.7285% | 98.7429% |
| DistilBERT | 99.6431% | 99.6491% |

DistilBERT dipilih sebagai final production model berdasarkan nilai
Validation Macro F1 tertinggi.

Final Test Set tidak digunakan selama proses model selection.

---

# 2. Final Model Evaluation

Setelah final model dipilih, evaluasi dilakukan menggunakan Final Test Set.

Hasil:

```text
Test Samples       : 4,483
Correct Prediction : 4,471
Prediction Errors  : 12

Accuracy            : 99.7323%
Macro Precision     : 99.7436%
Macro Recall        : 99.7287%
Macro F1            : 99.7325%
Weighted F1         : 99.7326%
```

Evaluation artifacts disimpan pada:

```text
reports/metrics/
```

File yang tersedia antara lain:

```text
distilbert_final_test_metrics.json
distilbert_final_test_prediction_errors.csv
distilbert_final_test_confusion_pairs.csv
distilbert_final_test_error_summary.csv
```

---

# 3. Model Versioning

Source code dan model artifact menggunakan Git untuk version control.

Repository:

```text
SupportPilot_AI_Bootcamp_KodingData
```

Karena ukuran model DistilBERT cukup besar, model production disimpan
menggunakan Git LFS.

Production model:

```text
models/
└── distilbert_supportpilot/
    └── best_model/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── tokenizer_config.json
```

File:

```text
model.safetensors
```

dikelola menggunakan Git LFS.

Hal ini memungkinkan production model memiliki version history yang
terintegrasi dengan source code.

---

# 4. Reproducible Environment

Dependency project dipisahkan berdasarkan fungsi.

```text
requirements-ml.txt
requirements-api.txt
requirements-ui.txt
requirements-dev.txt
```

Tujuannya adalah memisahkan dependency antara:

```text
Model Development
API Service
User Interface
Development / Testing
```

Production container menggunakan environment yang terisolasi sehingga
dependency dapat direproduksi secara konsisten.

---

# 5. Production Inference Layer

Production inference dipisahkan dari notebook training.

File:

```text
src/inference/distilbert_inference.py
```

Production inference menyediakan beberapa fungsi:

```text
predict_intent()
predict_top_k()
analyze_prediction_confidence()
predict_with_fallback()
predict_batch()
get_model_info()
```

Dengan pemisahan ini, aplikasi production tidak bergantung pada notebook
training.

---

# 6. Confidence Policy

Production model menggunakan confidence policy untuk mengurangi risiko
penggunaan prediction yang tidak cukup meyakinkan.

Default policy:

```text
Minimum Confidence : 70%
Minimum Margin     : 10%
```

Prediction diterima jika:

```text
confidence >= 0.70
AND
confidence_margin >= 0.10
```

Jika tidak memenuhi policy:

```text
final_intent = fallback
```

Fallback dapat diarahkan ke human review pada implementasi customer
support yang sebenarnya.

---

# 7. Automated Evaluation and Testing

SupportPilot AI menggunakan automated testing dengan Pytest.

Testing dibagi menjadi tiga layer.

## API Tests

```text
tests/test_api.py
```

Menguji:

- API status
- health endpoint
- model information
- single prediction
- fallback
- Top-K prediction
- batch prediction
- input validation

Jumlah:

```text
15 tests
```

## Docker API Integration Tests

```text
tests/test_docker_api.py
```

Menguji production API yang berjalan di Docker.

Jumlah:

```text
6 tests
```

## Docker UI Integration Tests

```text
tests/test_docker_ui.py
```

Menguji:

- UI health
- UI root page
- non-root execution
- internal API configuration

Jumlah:

```text
4 tests
```

Total automated tests:

```text
25 tests
```

Current result:

```text
25 passed
0 failed
```

---

# 8. Containerization

Production application dijalankan menggunakan Docker.

Terdapat dua service utama:

```text
api
ui
```

API menggunakan:

```text
FastAPI
DistilBERT
PyTorch CPU
```

UI menggunakan:

```text
Streamlit
```

---

# 9. Docker Compose Deployment

Docker Compose digunakan untuk menjalankan seluruh production stack.

Architecture:

```text
User
 ↓
Streamlit UI
 ↓
Internal Docker Network
 ↓
FastAPI
 ↓
DistilBERT
 ↓
Confidence Policy
 ↓
Accepted / Fallback
```

Production stack dapat dijalankan dengan satu command:

```bash
docker compose up -d --build
```

---

# 10. Container Security

Production container tidak berjalan sebagai root.

User:

```text
appuser
```

Group:

```text
appgroup
```

Security configuration mencakup:

```text
non-root execution
no-new-privileges
cap_drop: ALL
init process
health check
graceful shutdown
```

---

# 11. Health Monitoring

API dan UI memiliki Docker health check.

Status service dapat diperiksa menggunakan:

```bash
docker compose ps
```

API health endpoint:

```text
/health
```

Health information mencakup:

```text
model_loaded
device
number_of_labels
```

Monitoring ini berfungsi sebagai baseline operational monitoring untuk
memastikan inference service dan UI tetap tersedia.

---

# 12. Deployment Reproducibility

Project telah diuji menggunakan fresh repository clone.

Production model berhasil di-download kembali melalui Git LFS dan
seluruh production stack berhasil dibangun kembali menggunakan:

```bash
docker compose up -d --build
```

Automated Docker integration tests juga berhasil dijalankan pada hasil
clone tersebut.

Hal ini membuktikan bahwa deployment dapat direproduksi dari repository.

---

# MLOps Architecture

```text
                         Git Repository
                              │
                              ▼
                    Model / Code Versioning
                      Git + Git LFS
                              │
                              ▼
                      Production Model
                         DistilBERT
                              │
                              ▼
                     Automated Testing
                           Pytest
                              │
                              ▼
                     Docker Container
                              │
                              ▼
                       Docker Compose
                       ┌──────┴──────┐
                       ▼             ▼
                    FastAPI       Streamlit
                       │             │
                       └──────┬──────┘
                              ▼
                       Health Monitoring
```

---

# Current MLOps Coverage

| MLOps Practice | Implementation |
|---|---|
| Experiment tracking | Evaluation metrics and notebooks |
| Model comparison | Logistic Regression, SVM, DistilBERT |
| Model versioning | Git + Git LFS |
| Model artifact | DistilBERT `best_model` |
| Automated evaluation | Pytest + Final Test metrics |
| Reproducible environment | Requirement files |
| Production inference | Dedicated inference module |
| API serving | FastAPI |
| UI serving | Streamlit |
| Containerization | Docker |
| Service orchestration | Docker Compose |
| Health monitoring | API/UI healthcheck |
| Deployment validation | Fresh clone deployment test |
| Prediction safety | Confidence policy + fallback |

---

# Future MLOps Improvements

Production MLOps dapat dikembangkan lebih lanjut menggunakan:

```text
MLflow
Model Registry
CI/CD Pipeline
Prometheus
Grafana
Prediction Logging
Data Drift Detection
Model Drift Detection
Centralized Logging
Human Feedback Loop
Automated Retraining
```

Implementasi tersebut menjadi pengembangan lanjutan dan tidak termasuk
scope utama prototype SupportPilot AI saat ini.