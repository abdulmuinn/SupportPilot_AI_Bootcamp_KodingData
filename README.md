# SupportPilot AI

> Sistem AI Customer Support berbasis Machine Learning, Retrieval-Augmented Generation (RAG), Large Language Model (LLM), FastAPI, dan MLOps/LLMOps.

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-blue)

---

## 1. Gambaran Umum

**SupportPilot AI** adalah sistem customer support berbasis Artificial Intelligence yang dirancang untuk membantu menangani pertanyaan pelanggan secara otomatis.

Sistem dapat:

- memahami pertanyaan pelanggan,
- mengklasifikasikan intent pelanggan,
- mencari informasi yang relevan dari knowledge base perusahaan,
- menghasilkan jawaban menggunakan Large Language Model,
- memberikan sumber informasi yang digunakan,
- mendeteksi kondisi ketika AI tidak cukup yakin,
- dan merekomendasikan Human Escalation.

Project ini dikembangkan sebagai Final Project **Track B — AI Engineering KodingData Bootcamp** sekaligus sebagai project portfolio AI Engineering.

---

## 2. Latar Belakang

Tim customer support pada sebuah perusahaan sering menerima banyak pertanyaan berulang dari pelanggan.

Contoh pertanyaan tersebut antara lain:

- Status pesanan
- Status pengiriman
- Refund
- Pembayaran
- Pembatalan pesanan
- Permasalahan akun
- Invoice
- Informasi layanan
- Permintaan berbicara dengan customer service

Jika seluruh pertanyaan tersebut ditangani secara manual, beberapa masalah dapat muncul:

- Waktu respons customer support menjadi lebih lama
- Beban kerja customer support meningkat
- Jawaban antar customer support dapat tidak konsisten
- Customer support membutuhkan waktu untuk mencari kebijakan perusahaan
- Pelanggan harus menunggu untuk memperoleh jawaban
- Sistem AI biasa dapat menghasilkan jawaban yang tidak sesuai informasi perusahaan

Karena itu dibutuhkan sistem AI yang tidak hanya mampu menghasilkan jawaban, tetapi juga mampu memahami maksud pelanggan dan menggunakan informasi perusahaan sebagai dasar jawaban.

---

## 3. Problem Statement

Project ini mencoba menjawab pertanyaan berikut:

> Bagaimana membangun sistem AI Customer Support yang mampu memahami intent pelanggan, mengambil informasi yang relevan dari knowledge base perusahaan, menghasilkan jawaban yang grounded menggunakan Large Language Model, serta mengalihkan pertanyaan kepada manusia ketika AI tidak memiliki confidence yang cukup?

---

## 4. Tujuan Project

Tujuan utama project SupportPilot AI adalah membangun sistem AI end-to-end yang mampu:

1. Mengklasifikasikan intent dari pertanyaan pelanggan
2. Membandingkan beberapa pendekatan Machine Learning dan Transformer
3. Memilih model Intent Classification terbaik
4. Mencari informasi yang relevan dari knowledge base
5. Mengimplementasikan Retrieval-Augmented Generation (RAG)
6. Menghasilkan jawaban menggunakan Large Language Model
7. Mengurangi risiko hallucination
8. Memberikan informasi sumber jawaban
9. Melakukan Human Escalation untuk pertanyaan tertentu
10. Menyediakan REST API menggunakan FastAPI
11. Menyediakan antarmuka pengguna menggunakan Streamlit
12. Melakukan experiment tracking dan monitoring menggunakan MLflow
13. Menjalankan aplikasi menggunakan container Docker

---

## 5. Business Objective

Dari sisi bisnis, sistem ini bertujuan untuk membantu perusahaan:

- Mengurangi pertanyaan customer support yang bersifat repetitif
- Mempercepat waktu respons kepada pelanggan
- Memberikan jawaban yang lebih konsisten
- Membantu customer support mencari informasi perusahaan
- Mengurangi risiko jawaban AI yang tidak sesuai knowledge base
- Memberikan mekanisme Human Escalation
- Menyediakan data yang dapat digunakan untuk analisis customer support

---

## 6. Target Pengguna

### Customer

Customer dapat mengirimkan pertanyaan melalui antarmuka aplikasi dan menerima jawaban AI berdasarkan knowledge base perusahaan.

### Customer Support Agent

Customer Support Agent dapat menggunakan sistem sebagai AI Copilot untuk mendapatkan rekomendasi jawaban.

### Customer Support Manager

Manager dapat menggunakan informasi yang dihasilkan sistem untuk memantau:

- kategori pertanyaan pelanggan,
- intent pelanggan,
- confidence model,
- kasus Human Escalation,
- response latency,
- feedback,
- dan performa model.

---

## 7. Contoh Penggunaan

Contoh pertanyaan pelanggan:

```text
"I haven't received my refund yet."
```

Sistem melakukan Intent Classification:

```text
Intent:
track_refund

Confidence:
0.94
```

Kemudian sistem mencari informasi yang relevan:

```text
Knowledge Base:
Refund Policy
```

LLM menghasilkan jawaban berdasarkan informasi tersebut.

Contoh output:

```text
Intent:
track_refund

Confidence:
0.94

Response:
Your refund is currently being processed. Refund processing
may take several business days depending on the payment method.

Source:
Refund Policy

Human Escalation:
No
```

---

## 8. Alur Kerja Sistem

Secara sederhana, sistem bekerja dengan alur berikut:

```text
Pertanyaan Customer
        |
        v
   Streamlit UI
        |
        v
      FastAPI
        |
        +-------------------------------+
        |                               |
        v                               v
Intent Classification             RAG Retriever
        |                               |
        |                               v
        |                            Qdrant
        |                               |
        |                               v
        |                        Knowledge Base
        |                               |
        +---------------+---------------+
                        |
                        v
                   LLM Generator
                        |
                        v
                 Confidence Check
                    /        \
                   /          \
                  v            v
            Jawaban AI    Human Escalation
```

MLflow digunakan untuk melakukan experiment tracking dan monitoring terhadap komponen Machine Learning dan AI.

---

## 9. Komponen Utama Sistem

### 9.1 Intent Classification

Intent Classification digunakan untuk mengenali maksud dari pertanyaan pelanggan.

Contoh:

```text
Input:
"Where is my order?"

Output:
track_order
```

Contoh lainnya:

```text
Input:
"I want to cancel my order."

Output:
cancel_order
```

Informasi intent dapat digunakan untuk:

- routing pertanyaan,
- mencari knowledge base yang relevan,
- analytics,
- monitoring,
- dan Human Escalation.

---

### 9.2 Model Comparison

Project ini akan membandingkan minimal tiga pendekatan model untuk Intent Classification.

Model yang direncanakan:

#### Model 1

```text
TF-IDF
   ↓
Logistic Regression
```

#### Model 2

```text
TF-IDF
   ↓
Linear SVM
```

#### Model 3

```text
Tokenizer
   ↓
DistilBERT
   ↓
Classification Head
```

Ketiga model akan dibandingkan untuk mengetahui pendekatan yang paling sesuai.

---

## 10. Evaluasi Model

Model Intent Classification akan dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Inference Time

Model terbaik tidak hanya ditentukan berdasarkan Accuracy.

Pemilihan model juga mempertimbangkan:

- Macro F1-score
- kemampuan generalisasi,
- performa pada setiap class,
- inference speed,
- ukuran model,
- kompleksitas,
- dan kebutuhan deployment.

---

## 11. Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation digunakan agar Large Language Model tidak hanya mengandalkan pengetahuan bawaan model.

Sebelum menghasilkan jawaban, sistem terlebih dahulu mencari informasi yang relevan dari knowledge base perusahaan.

Alur RAG:

```text
Customer Question
        |
        v
Text Embedding
        |
        v
Vector Search
        |
        v
Qdrant Vector Database
        |
        v
Relevant Knowledge
        |
        v
LLM Prompt
        |
        v
Generated Answer
```

Dengan pendekatan ini, jawaban AI diharapkan lebih sesuai dengan kebijakan dan informasi perusahaan.

---

## 12. Knowledge Base

Pada project ini akan dibuat perusahaan e-commerce fiktif sebagai studi kasus.

Knowledge base akan berisi beberapa informasi seperti:

```text
knowledge_base/
│
├── shipping_policy.md
├── refund_policy.md
├── cancellation_policy.md
├── payment_policy.md
├── account_policy.md
├── faq.md
└── human_escalation_policy.md
```

Dokumen tersebut akan diproses menjadi potongan informasi atau chunks.

Setiap chunk akan diubah menjadi embedding dan disimpan dalam Qdrant Vector Database.

---

## 13. Large Language Model

Large Language Model digunakan untuk menghasilkan jawaban customer support.

LLM akan menerima informasi berupa:

```text
Customer Question
+
Predicted Intent
+
Relevant Knowledge
+
System Instructions
```

Kemudian menghasilkan jawaban berdasarkan informasi tersebut.

---

## 14. Grounded Response

Salah satu tujuan utama project adalah mengurangi hallucination.

Karena itu LLM akan diarahkan untuk:

- menggunakan informasi dari knowledge base,
- tidak membuat informasi yang tidak tersedia,
- memberikan sumber informasi,
- dan merekomendasikan Human Escalation jika informasi tidak cukup.

---

## 15. Human Escalation

AI tidak harus menjawab seluruh pertanyaan.

Pertanyaan dapat dialihkan kepada manusia ketika:

- confidence Intent Classification terlalu rendah,
- retrieval tidak menemukan informasi relevan,
- knowledge base tidak memiliki informasi yang dibutuhkan,
- pertanyaan membutuhkan tindakan manual,
- pertanyaan membutuhkan keputusan manusia,
- atau model tidak mampu memberikan jawaban yang grounded.

Contoh:

```text
Intent:
unknown

Confidence:
0.38

Status:
Human Escalation Required
```

---

## 16. Evaluasi RAG

Sistem RAG akan dievaluasi menggunakan beberapa aspek:

- Retrieval Relevance
- Answer Relevance
- Groundedness
- Hallucination Rate
- Response Latency

Evaluasi dilakukan untuk memastikan bahwa dokumen yang diambil relevan dan jawaban LLM sesuai dengan knowledge base.

---

## 17. MLOps dan LLMOps

Project menggunakan MLflow untuk membantu proses experiment tracking dan monitoring.

Informasi yang dapat dicatat antara lain:

### Machine Learning

- Model
- Hyperparameter
- Accuracy
- Precision
- Recall
- Macro F1-score
- Training Time
- Model Artifact

### LLM / RAG

- Input
- Output
- Retrieval Result
- Response Latency
- Evaluation Result

---

## 18. REST API

Backend project akan dibuat menggunakan FastAPI.

Endpoint yang direncanakan:

```text
GET  /health

POST /predict-intent

POST /ask

POST /feedback

GET  /metrics
```

Contoh:

```text
POST /predict-intent
```

Input:

```json
{
  "message": "Where is my order?"
}
```

Output:

```json
{
  "intent": "track_order",
  "confidence": 0.94
}
```

---

## 19. Streamlit Application

Streamlit digunakan sebagai antarmuka pengguna.

Fitur yang direncanakan:

### Customer Support Chat

Digunakan untuk berinteraksi dengan SupportPilot AI.

### Model Information

Menampilkan informasi model Intent Classification.

### Sources

Menampilkan knowledge base yang digunakan untuk menghasilkan jawaban.

### Feedback

User dapat memberikan feedback terhadap jawaban AI.

### Monitoring Dashboard

Menampilkan beberapa informasi seperti:

- jumlah pertanyaan,
- intent distribution,
- Human Escalation,
- confidence,
- dan response latency.

---

## 20. Technology Stack

### Programming Language

```text
Python 3.11
```

### Data Processing

```text
Pandas
NumPy
```

### Machine Learning

```text
Scikit-learn
PyTorch
Hugging Face Transformers
```

### NLP dan Embedding

```text
Sentence Transformers
```

### Vector Database

```text
Qdrant
```

### Backend

```text
FastAPI
```

### Frontend

```text
Streamlit
```

### MLOps / LLMOps

```text
MLflow
```

### Infrastructure

```text
Docker
Docker Compose
```

### Version Control

```text
Git
GitHub
```

---

## 21. Struktur Project

```text
SupportPilot_AI_Bootcamp_KodingData/
│
├── app/
│   ├── api/
│   └── ui/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── knowledge_base/
│   └── evaluation/
│
├── docs/
│   └── project_charter.md
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── rag/
│   └── evaluation/
│
├── tests/
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## 22. Dataset

Dataset utama akan digunakan untuk membangun model Intent Classification.

Dataset harus memenuhi kebutuhan:

```text
Jumlah data:
>= 2.000 records

Task:
Text Classification

Input:
Customer Question

Target:
Customer Intent
```

Informasi dataset secara lengkap akan ditambahkan setelah tahap Dataset Acquisition selesai.

Dokumentasi dataset nantinya mencakup:

- nama dataset,
- sumber dataset,
- lisensi,
- jumlah records,
- jumlah kategori,
- jumlah intent,
- struktur kolom,
- dan alasan pemilihan dataset.

---

## 23. Tahapan Pengerjaan

### Phase 1 — Project Planning

- [x] Menentukan project
- [x] Menentukan business problem
- [x] Menentukan objective
- [x] Menentukan scope
- [x] Menentukan architecture awal
- [x] Membuat Project Charter
- [x] Membuat README awal

### Phase 2 — Dataset & Data Understanding

- [ ] Dataset Acquisition
- [ ] Load Dataset
- [ ] Memahami struktur dataset
- [ ] Data Quality Check
- [ ] Exploratory Data Analysis
- [ ] Class Distribution Analysis
- [ ] Text Length Analysis
- [ ] Duplicate Analysis
- [ ] Missing Value Analysis

### Phase 3 — Data Preprocessing

- [ ] Data Cleaning
- [ ] Duplicate Handling
- [ ] Train / Validation / Test Split
- [ ] TF-IDF Preparation
- [ ] Transformer Tokenization

### Phase 4 — Machine Learning

- [ ] Logistic Regression
- [ ] Linear SVM
- [ ] DistilBERT
- [ ] Model Evaluation
- [ ] Model Comparison
- [ ] Error Analysis
- [ ] Best Model Selection

### Phase 5 — MLOps

- [ ] MLflow Setup
- [ ] Experiment Tracking
- [ ] Metrics Logging
- [ ] Model Artifact Logging
- [ ] Model Versioning

### Phase 6 — Knowledge Base

- [ ] Membuat perusahaan studi kasus
- [ ] Shipping Policy
- [ ] Refund Policy
- [ ] Cancellation Policy
- [ ] Payment Policy
- [ ] Account Policy
- [ ] FAQ
- [ ] Human Escalation Policy

### Phase 7 — RAG

- [ ] Document Loading
- [ ] Document Cleaning
- [ ] Chunking
- [ ] Embedding
- [ ] Qdrant Setup
- [ ] Vector Storage
- [ ] Retrieval
- [ ] Retrieval Evaluation

### Phase 8 — LLM

- [ ] LLM Integration
- [ ] Prompt Engineering
- [ ] RAG + LLM Integration
- [ ] Grounded Response
- [ ] Source Citation
- [ ] Confidence Checking
- [ ] Human Escalation

### Phase 9 — Backend

- [ ] FastAPI Setup
- [ ] Health Endpoint
- [ ] Intent Prediction Endpoint
- [ ] RAG Endpoint
- [ ] Feedback Endpoint
- [ ] API Error Handling

### Phase 10 — Frontend

- [ ] Streamlit Setup
- [ ] Customer Chat Interface
- [ ] Intent Information
- [ ] Source Information
- [ ] Feedback System
- [ ] Monitoring Dashboard

### Phase 11 — Testing

- [ ] Unit Test
- [ ] API Test
- [ ] Model Test
- [ ] Retrieval Test
- [ ] Error Handling Test
- [ ] End-to-End Test

### Phase 12 — Docker & Deployment

- [ ] Dockerfile FastAPI
- [ ] Dockerfile Streamlit
- [ ] Qdrant Service
- [ ] MLflow Service
- [ ] Docker Compose
- [ ] End-to-End Container Test
- [ ] Deployment

### Phase 13 — Finalisasi

- [ ] README Final
- [ ] Architecture Diagram
- [ ] Screenshot MLflow
- [ ] Screenshot Application
- [ ] PowerPoint
- [ ] Demo Video
- [ ] Final Quality Check

---

## 24. Deliverables

Final project akan menghasilkan:

```text
Notebook Data Understanding dan EDA

Notebook Preprocessing

Notebook Model Comparison

Trained Intent Classification Model

MLflow Experiment Tracking

Knowledge Base

RAG Pipeline

LLM Integration

FastAPI Backend

Streamlit Application

Qdrant Vector Database

Human Escalation Mechanism

Docker Configuration

README

Architecture Documentation

PowerPoint Presentation

Demo Video
```

---

## 25. Success Criteria

Project dianggap berhasil jika:

- Dataset memiliki minimal 2.000 records
- Dataset berhasil dianalisis dan diproses
- Minimal tiga model berhasil dibandingkan
- Metrik evaluasi model terdokumentasi
- Model terbaik dapat melakukan Intent Classification
- MLflow dapat mencatat experiment
- Knowledge Base berhasil dibuat
- Qdrant dapat melakukan vector search
- RAG dapat menemukan informasi relevan
- LLM dapat menghasilkan jawaban berdasarkan knowledge base
- Sistem dapat melakukan Human Escalation
- FastAPI dapat digunakan
- Streamlit dapat digunakan
- Sistem dapat berjalan secara end-to-end
- Project dapat direproduksi menggunakan dokumentasi README

---

## 26. Project Scope

### Termasuk dalam Project

```text
Intent Classification
Machine Learning
Transformer
RAG
LLM
Knowledge Base
Vector Database
MLOps / LLMOps
FastAPI
Streamlit
Human Escalation
Docker
Evaluation
Monitoring
```

### Tidak Termasuk pada Versi Pertama

```text
WhatsApp Integration
Real CRM Integration
Real Customer Database
Payment Gateway
Voice AI
Production Authentication
Multi-Tenant SaaS
Real-Time Human Agent Platform
```

Fitur tersebut dapat dikembangkan setelah versi utama selesai.

---

## 27. Portfolio Development

Setelah Final Project KodingData selesai dan dipresentasikan, project akan dikembangkan menjadi portfolio internasional.

Tahapan lanjutan meliputi:

```text
README Bahasa Inggris

Professional GitHub Documentation

Live Demo

English Demo Video

Architecture Case Study

API Documentation

Upwork Portfolio Description

Freelancer Portfolio Description
```

---

## 28. Instalasi

> Bagian ini masih akan diperbarui seiring perkembangan project.

Clone repository:

```bash
git clone <repository-url>
```

Masuk ke directory project:

```bash
cd SupportPilot_AI_Bootcamp_KodingData
```

Buat Conda environment:

```bash
conda create -n supportpilot-ai python=3.11 -y
```

Aktifkan environment:

```bash
conda activate supportpilot-ai
```

Install dependency:

```bash
pip install -r requirements.txt
```

---

## 29. Menjalankan Project

> Project masih dalam tahap pengembangan.

Cara menjalankan FastAPI, Streamlit, MLflow, Qdrant, dan Docker akan ditambahkan setelah masing-masing komponen selesai dikembangkan.

---

## 30. Status Saat Ini

```text
Current Phase:
Project Planning

Next Phase:
Dataset Acquisition & Data Understanding
```

---

## 31. Author

**Abdul Muin**

Data Science Student  
Machine Learning & AI Engineering

---

## 32. Catatan

Project ini dibuat untuk:

```text
Final Project KodingData Bootcamp
+
AI Engineering Portfolio
+
Pembelajaran end-to-end AI Engineering
```

Dokumentasi saat ini menggunakan **Bahasa Indonesia** agar proses pembelajaran dan presentasi Final Project lebih mudah dipahami.

Setelah Final Project selesai, dokumentasi portfolio akan dikembangkan menggunakan Bahasa Inggris.