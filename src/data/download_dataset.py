from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


# ==========================================================
# KONFIGURASI DATASET
# ==========================================================

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "bitext/"
    "Bitext-retail-ecommerce-llm-chatbot-training-dataset/"
    "resolve/main/"
    "bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
)

OUTPUT_PATH = Path(
    "data/raw/bitext_retail_ecommerce.csv"
)

REQUIRED_COLUMNS = {
    "instruction",
    "intent",
    "category",
    "tags",
    "response",
}


# ==========================================================
# DOWNLOAD DATASET
# ==========================================================

def download_dataset():
    """
    Mengunduh dataset Bitext Retail eCommerce dari Hugging Face.

    Dataset disimpan dalam folder data/raw agar data asli
    tetap terpisah dari data yang nantinya sudah diproses.
    """

    print("=" * 60)
    print("SUPPORTPILOT AI - DATASET ACQUISITION")
    print("=" * 60)

    # Pastikan folder data/raw tersedia.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Hindari download ulang jika file sudah tersedia.
    if OUTPUT_PATH.exists():
        print("\nDataset sudah tersedia.")
        print(f"Lokasi: {OUTPUT_PATH}")

        return

    print("\nMengunduh dataset dari Hugging Face...")
    print(f"Sumber:\n{DATASET_URL}")

    urlretrieve(
        DATASET_URL,
        OUTPUT_PATH,
    )

    print("\nDataset berhasil diunduh.")
    print(f"Lokasi: {OUTPUT_PATH}")


# ==========================================================
# VALIDASI DATASET
# ==========================================================

def validate_dataset():
    """
    Membaca dataset menggunakan Pandas dan melakukan
    pemeriksaan struktur dasar.
    """

    print("\n" + "=" * 60)
    print("VALIDASI DATASET")
    print("=" * 60)

    df = pd.read_csv(
        OUTPUT_PATH,
    )

    print(
        f"\nJumlah baris : {len(df):,}"
    )

    print(
        f"Jumlah kolom : {len(df.columns)}"
    )

    print(
        f"Nama kolom   : {list(df.columns)}"
    )

    # Mengecek apakah seluruh kolom yang dibutuhkan tersedia.
    missing_columns = (
        REQUIRED_COLUMNS - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset tidak memiliki seluruh kolom "
            f"yang dibutuhkan: {missing_columns}"
        )

    print(
        "\nValidasi struktur dataset: BERHASIL"
    )

    print("\nContoh 3 data pertama:")

    print(
        df.head(3).to_string(
            index=False,
        )
    )


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():
    """
    Menjalankan proses Dataset Acquisition dan validasi.
    """

    download_dataset()

    validate_dataset()

    print("\n" + "=" * 60)
    print("DATASET ACQUISITION SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()