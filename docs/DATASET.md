# Dataset Documentation — SupportPilot AI

## Dataset Source

SupportPilot AI menggunakan dataset:

**Bitext - Retail (eCommerce) Tagged Training Dataset for LLM-based Virtual Assistants**

Dataset disediakan oleh **Bitext** melalui Hugging Face.

Official dataset page:

```text
https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset
```

Dataset file:

```text
bitext-retail-ecommerce-llm-chatbot-training-dataset.csv
```

---

## Dataset Overview

Dataset berisi customer request dan response pada domain **Retail / eCommerce**.

Dataset original memiliki:

```text
Rows       : 44,884
Columns    : 5
Intents    : 46
Categories : 13
Language   : English
```

Kolom dataset:

| Column | Description |
|---|---|
| `instruction` | Customer request atau message |
| `intent` | Intent spesifik dari customer request |
| `category` | High-level category dari intent |
| `tags` | Informasi variasi bahasa pada instruction |
| `response` | Contoh response dari virtual assistant |

Untuk project SupportPilot AI:

```text
Feature (X) : instruction
Target (y)  : intent
```

Kolom `response` tidak digunakan sebagai target karena tujuan utama
project adalah **Intent Classification**.

---

## Why This Dataset Was Selected

Dataset ini dipilih karena memenuhi kebutuhan project SupportPilot AI.

Alasan pemilihan:

1. Memiliki jumlah data yang cukup besar untuk eksperimen Machine Learning
   dan Deep Learning.

2. Memiliki **46 intent classes**, sehingga memberikan multiclass
   classification problem yang lebih menantang dibanding dataset dengan
   jumlah class yang sedikit.

3. Berasal dari domain **Retail / eCommerce**, sehingga sesuai dengan use
   case customer support automation.

4. Memiliki customer message dengan variasi bahasa, typo, expression, dan
   pola kalimat yang beragam.

5. Mendukung eksperimen beberapa pendekatan model, mulai dari classical
   Machine Learning sampai Transformer-based model.

6. Dataset tidak termasuk dataset pembelajaran dasar yang sangat umum
   seperti Iris, Titanic, atau MNIST.

---

## Dataset Acquisition

Dataset tidak disimpan langsung di repository Git karena file dataset
berukuran relatif besar.

Dataset dapat di-download menggunakan:

```bash
python src/data/download_dataset.py
```

Script:

```text
src/data/download_dataset.py
```

melakukan:

```text
Download dataset
      ↓
Save to data/raw/
      ↓
Validate required columns
      ↓
Dataset ready for Data Understanding
```

Raw dataset disimpan sebagai:

```text
data/raw/bitext_retail_ecommerce.csv
```

---

## Data Quality Check

Data Understanding dilakukan pada:

```text
notebooks/01_data_understanding.ipynb
```

Pemeriksaan mencakup:

- dataset shape
- column structure
- data types
- missing values
- duplicate records
- category distribution
- intent distribution
- text length analysis
- dataset quality assessment

Dataset original memiliki:

```text
44,884 rows
```

---

## Data Cleaning

Preprocessing dilakukan pada:

```text
notebooks/02_data_preprocessing.ipynb
```

Duplicate detection dilakukan menggunakan normalized instruction.

Hasil cleaning:

```text
Original dataset      : 44,884
Normalized duplicates : 57
Rows removed           : 57
Clean dataset          : 44,827
```

Teks asli pada kolom `instruction` tetap dipertahankan untuk modeling.

Normalisasi digunakan untuk **duplicate detection**, bukan untuk mengganti
teks asli yang digunakan oleh model.

---

## Dataset Split

Clean dataset kemudian dibagi menggunakan **stratified split** agar
distribusi intent tetap terjaga.

Hasil split:

```text
Training Set   : 35,861
Validation Set : 4,483
Final Test Set : 4,483
--------------------------------
Total          : 44,827
```

Pembagian dataset:

```text
Clean Dataset
44,827
     │
     ├── Training
     │   35,861
     │
     ├── Validation
     │   4,483
     │
     └── Final Test
         4,483
```

---

## Data Leakage Prevention

Training, Validation, dan Final Test Set dibuat sebelum model development.

Penggunaan dataset:

```text
Training Set
    ↓
Model Training

Validation Set
    ↓
Model Evaluation
    ↓
Model Comparison
    ↓
Final Model Selection

Final Test Set
    ↓
Final Evaluation Only
```

Final Test Set tidak digunakan untuk:

- model selection
- hyperparameter tuning
- retraining
- architecture selection

Hal ini dilakukan agar Final Test tetap menjadi unseen dataset untuk
mengukur kemampuan generalisasi model.

---

## Dataset Files

Dataset CSV tidak disimpan di Git repository.

File berikut di-ignore melalui `.gitignore`:

```text
data/raw/*.csv
data/processed/*.csv
```

File metadata yang tetap disimpan:

```text
data/processed/label_mapping.json
```

Label mapping digunakan untuk menjaga konsistensi mapping antara:

```text
Intent Name ↔ Class ID
```

pada training dan production inference.

---

## License

Dataset menggunakan:

```text
Community Data License Agreement
CDLA-Sharing-1.0
```

SupportPilot AI memberikan attribution kepada **Bitext** sebagai penyedia
dataset original.

Penggunaan dan redistribusi dataset harus mengikuti ketentuan lisensi dari
dataset original.

---

## Dataset Attribution

Dataset Provider:

```text
Bitext
```

Dataset:

```text
Bitext - Retail (eCommerce) Tagged Training Dataset
for LLM-based Virtual Assistants
```

Source:

```text
https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset
```

License:

```text
CDLA-Sharing-1.0
```

Dataset digunakan untuk tujuan pengembangan dan evaluasi
**SupportPilot AI — Customer Support Intent Classification**.