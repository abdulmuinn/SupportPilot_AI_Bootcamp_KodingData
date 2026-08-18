"""
SupportPilot AI
Production Inference Module - DistilBERT Intent Classifier
"""

from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "distilbert_supportpilot"
    / "best_model"
)


# ============================================================
# INFERENCE CONFIGURATION
# ============================================================

MAX_LENGTH = 64

MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_MARGIN = 0.10

FALLBACK_INTENT = "fallback"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL DIRECTORY VALIDATION
# ============================================================

if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Model DistilBERT tidak ditemukan:\n{MODEL_DIR}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
)

model.to(DEVICE)
model.eval()


# ============================================================
# LABEL MAPPING
# ============================================================

NUM_LABELS = int(
    model.config.num_labels
)

ID2LABEL = {
    int(idx): str(label)
    for idx, label in model.config.id2label.items()
}

LABEL2ID = {
    str(label): int(idx)
    for label, idx in model.config.label2id.items()
}


# ============================================================
# LABEL MAPPING VALIDATION
# ============================================================

def validate_label_mapping() -> bool:
    """
    Memastikan ID2LABEL dan LABEL2ID konsisten dengan
    jumlah output class pada model.
    """

    expected_ids = set(
        range(NUM_LABELS)
    )

    if len(ID2LABEL) != NUM_LABELS:

        raise ValueError(
            "Jumlah ID2LABEL tidak sesuai "
            "dengan jumlah class model."
        )

    if len(LABEL2ID) != NUM_LABELS:

        raise ValueError(
            "Jumlah LABEL2ID tidak sesuai "
            "dengan jumlah class model."
        )

    if set(ID2LABEL.keys()) != expected_ids:

        raise ValueError(
            "ID2LABEL memiliki ID yang tidak lengkap."
        )

    if set(LABEL2ID.values()) != expected_ids:

        raise ValueError(
            "LABEL2ID memiliki ID yang tidak lengkap."
        )

    for idx, label in ID2LABEL.items():

        if LABEL2ID.get(label) != idx:

            raise ValueError(
                f"Mapping label tidak konsisten: "
                f"{idx} -> {label}"
            )

    return True


validate_label_mapping()


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_text(
    text: str,
) -> str:
    """
    Membersihkan dan memvalidasi satu input text.
    """

    if not isinstance(text, str):

        raise TypeError(
            "Input text harus berupa string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Input text tidak boleh kosong."
        )

    return text


# ============================================================
# SINGLE INTENT PREDICTION
# ============================================================

def predict_intent(
    text: str,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Prediksi satu intent tanpa fallback policy.
    """

    text = _validate_text(
        text
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        logits = model(
            **encoded
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        confidence_tensor, predicted_tensor = torch.max(
            probabilities,
            dim=-1,
        )

    predicted_id = int(
        predicted_tensor.item()
    )

    confidence = float(
        confidence_tensor.item()
    )

    return {
        "text": text,
        "predicted_id": predicted_id,
        "predicted_intent": ID2LABEL[
            predicted_id
        ],
        "confidence": confidence,
        "confidence_percent": (
            confidence * 100
        ),
    }


# ============================================================
# TOP-K PREDICTION
# ============================================================

def predict_top_k(
    text: str,
    top_k: int = 5,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Mengembalikan Top-K intent.
    """

    text = _validate_text(
        text
    )

    if not isinstance(top_k, int):

        raise TypeError(
            "top_k harus berupa integer."
        )

    if top_k < 1:

        raise ValueError(
            "top_k minimal 1."
        )

    top_k = min(
        top_k,
        NUM_LABELS,
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        logits = model(
            **encoded
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        top_probs, top_ids = torch.topk(
            probabilities,
            k=top_k,
            dim=-1,
        )

    top_probs = (
        top_probs[0]
        .detach()
        .cpu()
        .tolist()
    )

    top_ids = (
        top_ids[0]
        .detach()
        .cpu()
        .tolist()
    )

    predictions = []

    for rank, (
        class_id,
        probability,
    ) in enumerate(
        zip(
            top_ids,
            top_probs,
        ),
        start=1,
    ):

        class_id = int(
            class_id
        )

        probability = float(
            probability
        )

        predictions.append(
            {
                "rank": rank,
                "predicted_id": class_id,
                "predicted_intent": ID2LABEL[
                    class_id
                ],
                "confidence": probability,
                "confidence_percent": (
                    probability * 100
                ),
            }
        )

    return {
        "text": text,
        "top_k": top_k,
        "predictions": predictions,
    }


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def analyze_prediction_confidence(
    text: str,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Mengambil Top-1, Top-2 dan confidence margin.
    """

    result = predict_top_k(
        text=text,
        top_k=2,
        max_length=max_length,
    )

    top1 = result[
        "predictions"
    ][0]

    top2 = result[
        "predictions"
    ][1]

    margin = (
        top1["confidence"]
        - top2["confidence"]
    )

    return {
        "text": result["text"],

        "predicted_id":
            top1["predicted_id"],

        "predicted_intent":
            top1["predicted_intent"],

        "confidence":
            top1["confidence"],

        "confidence_percent":
            top1["confidence_percent"],

        "second_best_id":
            top2["predicted_id"],

        "second_best_intent":
            top2["predicted_intent"],

        "second_best_confidence":
            top2["confidence"],

        "second_best_confidence_percent":
            top2["confidence_percent"],

        "confidence_margin":
            margin,

        "confidence_margin_percent":
            margin * 100,
    }


# ============================================================
# PREDICTION WITH FALLBACK
# ============================================================

def predict_with_fallback(
    text: str,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_CONFIDENCE_MARGIN,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """
    Intent prediction dengan confidence policy.
    """

    analysis = analyze_prediction_confidence(
        text=text,
        max_length=max_length,
    )

    confidence = analysis[
        "confidence"
    ]

    margin = analysis[
        "confidence_margin"
    ]

    confidence_pass = (
        confidence >= min_confidence
    )

    margin_pass = (
        margin >= min_margin
    )

    accepted = (
        confidence_pass
        and margin_pass
    )

    if accepted:

        final_intent = analysis[
            "predicted_intent"
        ]

        status = "accepted"

        reason = (
            "Prediction memenuhi "
            "confidence policy."
        )

    else:

        final_intent = FALLBACK_INTENT
        status = "fallback"

        reasons = []

        if not confidence_pass:

            reasons.append(
                "confidence di bawah threshold"
            )

        if not margin_pass:

            reasons.append(
                "confidence margin terlalu kecil"
            )

        reason = ", ".join(
            reasons
        )

    return {
        **analysis,

        "final_intent":
            final_intent,

        "min_confidence":
            min_confidence,

        "min_margin":
            min_margin,

        "accepted":
            accepted,

        "status":
            status,

        "reason":
            reason,
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(
    texts: List[str],
    batch_size: int = 32,
    min_confidence: float = MIN_CONFIDENCE,
    min_margin: float = MIN_CONFIDENCE_MARGIN,
    max_length: int = MAX_LENGTH,
) -> List[Dict[str, Any]]:
    """
    Melakukan inference terhadap banyak text sekaligus.
    """

    if not isinstance(
        texts,
        (list, tuple),
    ):

        raise TypeError(
            "texts harus berupa list atau tuple."
        )

    if len(texts) == 0:

        raise ValueError(
            "texts tidak boleh kosong."
        )

    if not isinstance(
        batch_size,
        int,
    ):

        raise TypeError(
            "batch_size harus berupa integer."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size minimal 1."
        )

    clean_texts = [
        _validate_text(text)
        for text in texts
    ]

    results = []

    for start_idx in range(
        0,
        len(clean_texts),
        batch_size,
    ):

        batch_texts = clean_texts[
            start_idx:
            start_idx + batch_size
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.inference_mode():

            logits = model(
                **encoded
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            top2_probs, top2_ids = torch.topk(
                probabilities,
                k=2,
                dim=-1,
            )

        top2_probs = (
            top2_probs
            .detach()
            .cpu()
            .tolist()
        )

        top2_ids = (
            top2_ids
            .detach()
            .cpu()
            .tolist()
        )

        for text, probs, ids in zip(
            batch_texts,
            top2_probs,
            top2_ids,
        ):

            top1_id = int(
                ids[0]
            )

            top2_id = int(
                ids[1]
            )

            confidence = float(
                probs[0]
            )

            second_confidence = float(
                probs[1]
            )

            margin = (
                confidence
                - second_confidence
            )

            predicted_intent = ID2LABEL[
                top1_id
            ]

            second_best_intent = ID2LABEL[
                top2_id
            ]

            accepted = (
                confidence >= min_confidence
                and
                margin >= min_margin
            )

            final_intent = (
                predicted_intent
                if accepted
                else FALLBACK_INTENT
            )

            results.append(
                {
                    "text": text,

                    "predicted_id":
                        top1_id,

                    "predicted_intent":
                        predicted_intent,

                    "final_intent":
                        final_intent,

                    "confidence":
                        confidence,

                    "confidence_percent":
                        confidence * 100,

                    "second_best_intent":
                        second_best_intent,

                    "second_best_confidence":
                        second_confidence,

                    "confidence_margin":
                        margin,

                    "confidence_margin_percent":
                        margin * 100,

                    "accepted":
                        accepted,

                    "status":
                        (
                            "accepted"
                            if accepted
                            else "fallback"
                        ),
                }
            )

    return results


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info() -> Dict[str, Any]:
    """
    Informasi model untuk health check API.
    """

    return {
        "model_name":
            model.__class__.__name__,

        "model_path":
            str(MODEL_DIR),

        "device":
            str(DEVICE),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "num_labels":
            NUM_LABELS,

        "max_length":
            MAX_LENGTH,

        "min_confidence":
            MIN_CONFIDENCE,

        "min_margin":
            MIN_CONFIDENCE_MARGIN,
    }


# ============================================================
# DIRECT EXECUTION SELF-CHECK
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "SUPPORTPILOT AI — INFERENCE SELF CHECK"
    )

    print("=" * 60)

    info = get_model_info()

    print(
        "Model      :",
        info["model_name"],
    )

    print(
        "Device     :",
        info["device"],
    )

    print(
        "GPU        :",
        info["gpu"],
    )

    print(
        "Classes    :",
        info["num_labels"],
    )

    print(
        "Mapping OK :",
        validate_label_mapping(),
    )

    print()

    result = predict_with_fallback(
        "Where is my order?"
    )

    print(
        "Test input :",
        result["text"],
    )

    print(
        "Intent     :",
        result["final_intent"],
    )

    print(
        "Confidence :",
        f"{result['confidence_percent']:.2f}%",
    )

    print(
        "Status     :",
        result["status"],
    )

    print()
    print(
        "✅ Inference module siap digunakan."
    )