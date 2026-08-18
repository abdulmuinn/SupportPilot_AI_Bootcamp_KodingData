"""
SupportPilot AI
FastAPI Application
"""

from fastapi import (
    FastAPI,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from src.inference.distilbert_inference import (
    get_model_info,
    predict_with_fallback,
    predict_top_k,
    predict_batch,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SupportPilot AI API",
    description=(
        "REST API untuk customer support "
        "intent classification menggunakan DistilBERT."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    """
    Request body untuk intent prediction.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Customer support message.",
        examples=[
            "Where is my order?"
        ],
    )


class PredictResponse(BaseModel):
    """
    Response prediction dari DistilBERT.
    """

    text: str

    predicted_id: int
    predicted_intent: str

    confidence: float
    confidence_percent: float

    second_best_id: int
    second_best_intent: str
    second_best_confidence: float
    second_best_confidence_percent: float

    confidence_margin: float
    confidence_margin_percent: float

    final_intent: str

    min_confidence: float
    min_margin: float

    accepted: bool
    status: str
    reason: str

# ============================================================
# TOP-K REQUEST / RESPONSE SCHEMAS
# ============================================================

class TopKRequest(BaseModel):
    """
    Request untuk mendapatkan beberapa kandidat intent.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Customer support message.",
        examples=[
            "Where is my package?"
        ],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Jumlah kandidat intent yang dikembalikan.",
    )


class TopKPrediction(BaseModel):
    """
    Satu kandidat intent pada hasil Top-K.
    """

    rank: int
    predicted_id: int
    predicted_intent: str
    confidence: float
    confidence_percent: float


class TopKResponse(BaseModel):
    """
    Response Top-K intent prediction.
    """

    text: str
    top_k: int
    predictions: list[TopKPrediction]
    
# ============================================================
# BATCH REQUEST / RESPONSE SCHEMAS
# ============================================================

class BatchPredictRequest(BaseModel):
    """
    Request untuk memprediksi beberapa customer message.
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Daftar customer support message "
            "yang akan diprediksi."
        ),
        examples=[
            [
                "Where is my order?",
                "I want to cancel my order.",
                "What is the weather today?",
            ]
        ],
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Ukuran batch inference.",
    )


class BatchPrediction(BaseModel):
    """
    Hasil prediction untuk satu customer message.
    """

    text: str

    predicted_id: int
    predicted_intent: str
    final_intent: str

    confidence: float
    confidence_percent: float

    second_best_intent: str
    second_best_confidence: float

    confidence_margin: float
    confidence_margin_percent: float

    accepted: bool
    status: str


class BatchPredictResponse(BaseModel):
    """
    Response batch prediction.
    """

    total: int
    accepted: int
    fallback: int

    predictions: list[BatchPrediction]

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "SupportPilot AI",
        "status": "running",
        "message": "SupportPilot AI API is running.",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    model_info = get_model_info()

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": model_info["device"],
        "num_labels": model_info["num_labels"],
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_information():

    model_info = get_model_info()

    # Jangan expose absolute local path ke client.
    model_info.pop(
        "model_path",
        None,
    )

    return model_info


# ============================================================
# PREDICT INTENT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict_intent_endpoint(
    request: PredictRequest,
):

    text = request.text.strip()

    if not text:

        raise HTTPException(
            status_code=422,
            detail="Text tidak boleh kosong.",
        )

    try:

        result = predict_with_fallback(
            text
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Terjadi kesalahan saat "
                "melakukan inference."
            ),
        ) from error
        
# ============================================================
# TOP-K INTENT PREDICTION
# ============================================================

@app.post(
    "/predict/top-k",
    response_model=TopKResponse,
)
def predict_top_k_endpoint(
    request: TopKRequest,
):

    text = request.text.strip()

    if not text:

        raise HTTPException(
            status_code=422,
            detail="Text tidak boleh kosong.",
        )

    try:

        result = predict_top_k(
            text=text,
            top_k=request.top_k,
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Terjadi kesalahan saat "
                "melakukan Top-K inference."
            ),
        ) from error
        
# ============================================================
# BATCH INTENT PREDICTION
# ============================================================

@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
)
def predict_batch_endpoint(
    request: BatchPredictRequest,
):

    # --------------------------------------------------------
    # Validasi text
    # --------------------------------------------------------

    clean_texts = [
        text.strip()
        for text in request.texts
    ]

    if any(
        not text
        for text in clean_texts
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Semua text dalam batch "
                "harus berisi karakter."
            ),
        )

    try:

        predictions = predict_batch(
            texts=clean_texts,
            batch_size=request.batch_size,
        )

        accepted_count = sum(
            item["accepted"]
            for item in predictions
        )

        fallback_count = (
            len(predictions)
            - accepted_count
        )

        return {
            "total": len(predictions),
            "accepted": accepted_count,
            "fallback": fallback_count,
            "predictions": predictions,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Terjadi kesalahan saat "
                "melakukan batch inference."
            ),
        ) from error