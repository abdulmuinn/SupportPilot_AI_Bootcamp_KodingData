# Project Charter — SupportPilot AI

## 1. Informasi Project

### Nama Project

**SupportPilot AI**

### Judul Lengkap

**SupportPilot AI: Sistem Intelligent Customer Support Menggunakan Intent Classification, Retrieval-Augmented Generation (RAG), Large Language Model, dan MLOps/LLMOps**

### Track

**Track B — AI Engineering**

### Jenis Project

```text
Machine Learning
+
Natural Language Processing
+
Large Language Model
+
Retrieval-Augmented Generation
+
AI Engineering
+
MLOps / LLMOps
```

### Status

```text
Dalam Pengembangan
```

---

## 2. Latar Belakang

Customer support merupakan salah satu bagian penting dalam sebuah bisnis karena berhubungan langsung dengan pelanggan.

Dalam operasionalnya, customer support sering menerima banyak pertanyaan yang bersifat repetitif.

Contohnya:

```text
"Where is my order?"

"How can I cancel my order?"

"I haven't received my refund."

"How can I change my password?"

"Why was my payment declined?"
```

Pertanyaan-pertanyaan tersebut sebenarnya dapat dikelompokkan berdasarkan maksud atau **intent** pelanggan.

Contohnya:

```text
"Where is my order?"
        ↓
track_order
```

atau:

```text
"I haven't received my refund."
        ↓
track_refund
```

Jika seluruh pertanyaan harus ditangani secara manual, perusahaan dapat menghadapi beberapa masalah seperti:

- waktu respons yang lebih lama,
- meningkatnya workload customer support,
- jawaban yang tidak konsisten,
- waktu yang dibutuhkan untuk mencari kebijakan perusahaan,
- serta meningkatnya biaya operasional customer support.

Large Language Model dapat membantu menghasilkan jawaban secara otomatis.

Namun menggunakan LLM secara langsung juga memiliki risiko.

Salah satunya adalah **hallucination**, yaitu kondisi ketika AI menghasilkan jawaban yang terlihat meyakinkan tetapi tidak berdasarkan informasi perusahaan.

Karena itu, project ini tidak hanya menggunakan LLM.

SupportPilot AI menggabungkan:

```text
Intent Classification
+
Knowledge Retrieval
+
RAG
+
LLM
+
Confidence Checking
+
Human Escalation
+
Monitoring
```

---

## 3. Problem Statement

Masalah utama yang ingin diselesaikan dalam project ini adalah:

> Bagaimana membangun sistem AI Customer Support yang dapat memahami maksud pertanyaan pelanggan, mencari informasi perusahaan yang relevan, menghasilkan jawaban berdasarkan informasi tersebut, serta mengalihkan pertanyaan kepada manusia ketika AI tidak memiliki confidence yang cukup?

Permasalahan tersebut dapat dipecah menjadi beberapa technical problem.

### Problem 1 — Memahami Pertanyaan

Sistem harus memahami maksud dari pertanyaan pelanggan.

Solusi:

```text
Intent Classification
```

---

### Problem 2 — Mencari Informasi yang Relevan

Sistem harus mencari informasi perusahaan yang berhubungan dengan pertanyaan.

Solusi:

```text
Information Retrieval
+
Vector Search
```

---

### Problem 3 — Menghasilkan Jawaban

Sistem harus mampu memberikan jawaban yang natural kepada pelanggan.

Solusi:

```text
Large Language Model
```

---

### Problem 4 — Mengurangi Hallucination

LLM tidak boleh bebas menghasilkan informasi yang tidak tersedia.

Solusi:

```text
Retrieval-Augmented Generation
+
Grounded Prompt
```

---

### Problem 5 — Menangani Ketidakpastian

Tidak semua pertanyaan harus dijawab AI.

Solusi:

```text
Confidence Checking
+
Human Escalation
```

---

### Problem 6 — Mengetahui Performa AI

Performa model dan sistem harus dapat dipantau.

Solusi:

```text
MLOps
+
LLMOps
+
MLflow
```

---

## 4. Business Objective

Tujuan bisnis dari project SupportPilot AI adalah membantu proses customer support menjadi lebih efisien.

Sistem diharapkan dapat:

- mengurangi pertanyaan berulang yang harus dijawab manual,
- mempercepat respons kepada pelanggan,
- meningkatkan konsistensi jawaban,
- membantu customer support menemukan informasi lebih cepat,
- meningkatkan efisiensi operasional,
- dan tetap mempertahankan Human-in-the-Loop untuk kasus tertentu.

---

## 5. Technical Objective

Dari sisi teknis, project bertujuan membangun sistem AI end-to-end yang memiliki kemampuan:

1. Melakukan Intent Classification
2. Melatih beberapa model Machine Learning / Transformer
3. Membandingkan performa minimal tiga model
4. Memilih model terbaik berdasarkan evaluation metrics
5. Mengimplementasikan experiment tracking
6. Membuat knowledge base perusahaan
7. Mengubah knowledge base menjadi vector embedding
8. Menyimpan embedding pada vector database
9. Melakukan semantic retrieval
10. Mengimplementasikan RAG
11. Mengintegrasikan Large Language Model
12. Melakukan confidence checking
13. Melakukan Human Escalation
14. Menyediakan REST API
15. Menyediakan user interface
16. Melakukan monitoring
17. Menjalankan aplikasi menggunakan Docker

---

## 6. Target Pengguna

### 6.1 Customer

Customer dapat menggunakan sistem untuk mendapatkan jawaban terhadap pertanyaan umum.

---

### 6.2 Customer Support Agent

Customer Support Agent dapat menggunakan sistem sebagai **AI Copilot**.

AI membantu agent:

- memahami intent pelanggan,
- menemukan informasi,
- dan menghasilkan rekomendasi jawaban.

---

### 6.3 Customer Support Manager

Manager dapat menggunakan data dari sistem untuk melihat:

```text
Intent Distribution

Human Escalation Rate

Model Confidence

Response Latency

Customer Feedback

Model Performance
```

---

## 7. Input Sistem

Input utama berupa teks pertanyaan pelanggan.

Contoh:

```text
"I ordered my laptop five days ago but it has not arrived yet."
```

---

## 8. Output Sistem

Sistem akan memberikan beberapa informasi.

Contoh:

```text
Predicted Intent:
track_order

Confidence:
0.94

Retrieved Knowledge:
Shipping Policy

AI Response:
Your order can be tracked through the Orders section...

Source:
Shipping Policy

Human Escalation:
No
```

---

## 9. Studi Kasus

Untuk menghindari penggunaan data internal perusahaan nyata, project akan menggunakan studi kasus perusahaan e-commerce fiktif.

Nama sementara:

```text
NovaCart
```

NovaCart digunakan sebagai konteks bisnis untuk knowledge base.

Contoh informasi perusahaan:

```text
Shipping Policy

Refund Policy

Cancellation Policy

Payment Policy

Account Policy

Frequently Asked Questions

Human Escalation Policy
```

Nama perusahaan masih dapat diubah pada tahap pengembangan berikutnya.

---

## 10. Core Features

### 10.1 Intent Classification

Sistem mampu menentukan intent dari pertanyaan pelanggan.

Contoh:

```text
Customer:
"Can I cancel my order?"

Intent:
cancel_order
```

---

### 10.2 Model Comparison

Tiga model awal yang akan dibandingkan:

```text
TF-IDF + Logistic Regression

TF-IDF + Linear SVM

DistilBERT
```

---

### 10.3 Knowledge Base

Sistem memiliki database pengetahuan perusahaan.

---

### 10.4 Semantic Search

Pertanyaan pengguna diubah menjadi embedding dan digunakan untuk mencari informasi relevan.

---

### 10.5 Retrieval-Augmented Generation

Informasi relevan dari knowledge base diberikan kepada LLM sebagai context.

---

### 10.6 LLM Response Generation

LLM menghasilkan jawaban customer support.

---

### 10.7 Source Reference

Sistem menampilkan sumber knowledge base yang digunakan.

---

### 10.8 Confidence Checking

Sistem menentukan tingkat keyakinan terhadap proses tertentu.

---

### 10.9 Human Escalation

Kasus yang tidak dapat ditangani AI dialihkan kepada customer support manusia.

---

### 10.10 Monitoring

Aktivitas model dan sistem dapat dipantau.

---

### 10.11 REST API

Kemampuan AI tersedia melalui API sehingga dapat diintegrasikan dengan aplikasi lain.

---

### 10.12 Web Application

User dapat mencoba sistem melalui aplikasi Streamlit.

---

## 11. Machine Learning Problem

Jenis Machine Learning problem:

```text
Supervised Learning
```

Task:

```text
Multiclass Text Classification
```

Input:

```text
Customer Question
```

Target:

```text
Customer Intent
```

Contoh:

```text
Input:
"Where is my refund?"

Target:
track_refund
```

---

## 12. Model yang Akan Dibandingkan

### 12.1 Logistic Regression

Pipeline:

```text
Raw Text
   |
   v
TF-IDF
   |
   v
Logistic Regression
   |
   v
Predicted Intent
```

Model ini digunakan sebagai baseline.

---

### 12.2 Linear SVM

Pipeline:

```text
Raw Text
   |
   v
TF-IDF
   |
   v
Linear SVM
   |
   v
Predicted Intent
```

Linear SVM akan dibandingkan dengan Logistic Regression pada representasi TF-IDF.

---

### 12.3 DistilBERT

Pipeline:

```text
Raw Text
   |
   v
Tokenizer
   |
   v
DistilBERT
   |
   v
Classification Head
   |
   v
Predicted Intent
```

DistilBERT digunakan sebagai pendekatan Transformer untuk memahami konteks teks.

---

## 13. Evaluation Metrics

### 13.1 Classification Metrics

Model akan dibandingkan menggunakan:

```text
Accuracy

Precision

Recall

Macro F1-score

Confusion Matrix

Inference Time
```

Macro F1-score menjadi salah satu metric penting karena setiap intent perlu mendapatkan perhatian yang seimbang.

---

### 13.2 Model Selection

Model terbaik tidak otomatis merupakan model dengan Accuracy tertinggi.

Pemilihan model mempertimbangkan:

```text
Predictive Performance

Generalization

Per-Class Performance

Inference Speed

Model Size

Deployment Complexity
```

---

## 14. RAG Architecture

Alur Retrieval-Augmented Generation:

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
Qdrant
        |
        v
Relevant Chunks
        |
        v
Prompt Construction
        |
        v
Large Language Model
        |
        v
Grounded Response
```

---

## 15. Knowledge Base Pipeline

Knowledge base akan diproses menggunakan pipeline:

```text
Knowledge Documents
        |
        v
Document Loading
        |
        v
Text Cleaning
        |
        v
Chunking
        |
        v
Embedding
        |
        v
Qdrant Vector Database
```

---

## 16. RAG Evaluation

Beberapa aspek yang akan dievaluasi:

```text
Retrieval Relevance

Answer Relevance

Groundedness

Hallucination Rate

Response Latency
```

---

## 17. Human Escalation Logic

Human Escalation dilakukan ketika kondisi tertentu terpenuhi.

High-level logic:

```text
Customer Question
        |
        v
Intent Classification
        |
        v
Confidence Check
        |
   +----+----+
   |         |
High       Low
   |         |
   v         v
Continue   Escalate
   |
   v
RAG Retrieval
   |
   v
Retrieval Quality
   |
   +-------------------+
   |                   |
Relevant          Not Relevant
   |                   |
   v                   v
LLM Response       Escalate
```

Threshold confidence akan ditentukan berdasarkan hasil experiment, bukan ditetapkan secara sembarangan dari awal.

---

## 18. High-Level System Architecture

```text
                         +----------------------+
                         |      Customer        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Streamlit UI      |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |       FastAPI        |
                         +----------+-----------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
        +----------------------+          +----------------------+
        | Intent Classifier    |          |    RAG Retriever     |
        +----------+-----------+          +----------+-----------+
                   |                                 |
                   |                                 v
                   |                       +----------------------+
                   |                       |       Qdrant         |
                   |                       +----------+-----------+
                   |                                  |
                   |                                  v
                   |                       +----------------------+
                   |                       |   Knowledge Base     |
                   |                       +----------+-----------+
                   |                                  |
                   +----------------+-----------------+
                                    |
                                    v
                         +----------------------+
                         |    LLM Generator     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |  Confidence Check    |
                         +----------+-----------+
                                    |
                              +-----+-----+
                              |           |
                              v           v
                         AI Response   Human Agent
```

MLflow digunakan untuk experiment tracking dan monitoring.

---

## 19. Technology Stack

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

### NLP / Embeddings

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

## 20. Project Scope

### 20.1 In Scope

Project versi utama mencakup:

```text
Dataset

EDA

Data Preprocessing

Intent Classification

Model Comparison

MLflow Experiment Tracking

Knowledge Base

Embedding

Vector Database

RAG

LLM Integration

Grounded Response

Human Escalation

FastAPI

Streamlit

Evaluation

Testing

Docker
```

---

### 20.2 Out of Scope

Versi pertama tidak mencakup:

```text
Real WhatsApp Integration

Real CRM Integration

Real Customer Database

Payment Gateway

Voice Customer Support

Production Authentication

Multi-Tenant SaaS

Real-Time Human Support Dashboard

Enterprise Infrastructure
```

Fitur tersebut dapat dikembangkan pada versi portfolio berikutnya.

---

## 21. Success Criteria

Project dianggap berhasil jika memenuhi kriteria berikut.

### Data

```text
Dataset >= 2.000 records
```

Dataset berhasil dianalisis dan diproses.

---

### Machine Learning

```text
Minimal tiga model berhasil dibandingkan
```

Model terbaik dapat dipilih berdasarkan evaluation metrics.

---

### MLOps

Experiment dapat dilacak menggunakan MLflow.

---

### RAG

Knowledge base berhasil:

```text
Loaded
   ↓
Chunked
   ↓
Embedded
   ↓
Stored
   ↓
Retrieved
```

---

### LLM

LLM dapat menghasilkan jawaban berdasarkan context yang diberikan.

---

### Human Escalation

Sistem dapat mendeteksi kondisi yang membutuhkan bantuan manusia.

---

### Backend

FastAPI dapat menerima request dan memberikan response.

---

### Frontend

Streamlit dapat digunakan untuk melakukan interaksi dengan sistem.

---

### Infrastructure

Project dapat dijalankan menggunakan Docker.

---

### Documentation

Project dapat direproduksi berdasarkan README.

---

## 22. Project Deliverables

Output akhir project direncanakan terdiri dari:

```text
Jupyter Notebook

EDA

Data Preprocessing

Model Training

Model Comparison

Best Model

MLflow Experiment

Knowledge Base

RAG Pipeline

LLM Integration

FastAPI Application

Streamlit Application

Qdrant Database

Docker Configuration

Architecture Diagram

README

PowerPoint

Demo Video
```

---

## 23. Risiko Project

### Risiko 1 — Dataset terlalu mudah

Dataset yang terlalu mudah dapat menghasilkan evaluation score tinggi tetapi kurang menunjukkan kemampuan generalisasi.

Mitigasi:

```text
EDA
+
Error Analysis
+
Train/Validation/Test Split
```

---

### Risiko 2 — Data Leakage

Data yang sangat mirip dapat masuk ke train dan test dataset.

Mitigasi:

```text
Duplicate Analysis
+
Careful Dataset Split
```

---

### Risiko 3 — LLM Hallucination

Model menghasilkan informasi yang tidak ada dalam knowledge base.

Mitigasi:

```text
RAG
+
Grounded Prompt
+
Human Escalation
```

---

### Risiko 4 — Retrieval Tidak Relevan

Vector search mengambil dokumen yang tidak sesuai.

Mitigasi:

```text
Retrieval Evaluation
+
Chunking Experiment
+
Embedding Evaluation
```

---

### Risiko 5 — Scope Terlalu Besar

Project dapat menjadi terlalu kompleks.

Mitigasi:

```text
Bootcamp MVP terlebih dahulu
        ↓
Portfolio Enhancement setelah Final Project
```

---

## 24. Project Development Strategy

Project dikembangkan dalam dua tahap besar.

### Version 1 — Bootcamp MVP

Prioritas:

```text
Data
↓
EDA
↓
Preprocessing
↓
Machine Learning
↓
Model Comparison
↓
MLflow
↓
RAG
↓
LLM
↓
FastAPI
↓
Streamlit
↓
Docker
```

Tujuan utama Version 1 adalah menghasilkan project Final Project yang berfungsi secara end-to-end.

---

### Version 2 — Portfolio Enhancement

Setelah Final Project selesai:

```text
Better UI

Advanced Monitoring

Deployment

API Documentation

CI/CD

Improved RAG Evaluation

English Documentation

English Demo Video

Portfolio Case Study

Upwork Portfolio
```

---

## 25. Roadmap

### Phase 1

```text
Project Planning
```

Status:

```text
DONE
```

---

### Phase 2

```text
Dataset Acquisition
+
Data Understanding
+
EDA
```

Status:

```text
NEXT
```

---

### Phase 3

```text
Data Preprocessing
```

---

### Phase 4

```text
Model Training
+
Model Comparison
```

---

### Phase 5

```text
MLOps
```

---

### Phase 6

```text
Knowledge Base
+
RAG
```

---

### Phase 7

```text
LLM Integration
```

---

### Phase 8

```text
FastAPI
```

---

### Phase 9

```text
Streamlit
```

---

### Phase 10

```text
Testing
```

---

### Phase 11

```text
Docker
+
Deployment
```

---

### Phase 12

```text
Documentation
+
Presentation
+
Demo Video
```

---

## 26. Portfolio Objective

Selain memenuhi Final Project, SupportPilot AI dirancang untuk menunjukkan kemampuan AI Engineering secara end-to-end.

Project diharapkan dapat menunjukkan kemampuan dalam:

```text
Business Problem Understanding

Data Analysis

Machine Learning

NLP

Transformer

LLM

RAG

Vector Database

FastAPI

MLOps / LLMOps

Docker

AI Application Development

Model Evaluation

System Design
```

Setelah presentasi Final Project selesai, dokumentasi project akan dikembangkan ke dalam Bahasa Inggris untuk kebutuhan portfolio internasional.

---

## 27. Final Project Strategy

Prioritas pengerjaan project adalah:

```text
Correctness
    ↓
Functionality
    ↓
Understanding
    ↓
Evaluation
    ↓
Documentation
    ↓
UI Enhancement
```

Project tidak akan berfokus pada tampilan aplikasi terlebih dahulu.

Model dan pipeline harus bekerja dengan benar sebelum dilakukan pengembangan UI lanjutan.

---

## 28. Catatan Pembelajaran

Selama pengerjaan project, setiap bagian akan dipelajari berdasarkan logika:

```text
WHY
↓
WHAT
↓
HOW
↓
CODE
↓
RESULT
↓
INTERPRETATION
```

Artinya, sebelum menulis kode perlu dipahami terlebih dahulu:

```text
Kenapa langkah ini diperlukan?

Apa yang sedang kita kerjakan?

Bagaimana prosesnya bekerja?

Bagaimana implementasi kodenya?

Apa hasilnya?

Apa arti hasil tersebut?
```

Pendekatan ini digunakan agar project tidak hanya selesai, tetapi konsep dan logika AI Engineering juga dapat dipahami.

---

## 29. Project Owner

**Abdul Muin**

Final Project  
Track B — AI Engineering  
KodingData Bootcamp

---

## 30. Status Project

```text
Phase 1:
Project Planning ✅

Current:
README & Project Charter ✅

Next:
Dataset Acquisition & Data Understanding
```