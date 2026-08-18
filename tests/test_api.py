"""
Automated API Tests
SupportPilot AI
"""

from fastapi.testclient import TestClient

from src.api.main import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# BASIC API TESTS
# ============================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == "SupportPilot AI"
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_labels"] == 46


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["num_labels"] == 46
    assert data["max_length"] == 64

    # absolute local path tidak boleh terekspos
    assert "model_path" not in data


# ============================================================
# SINGLE PREDICTION TESTS
# ============================================================

def test_predict_track_order():
    response = client.post(
        "/predict",
        json={
            "text": "Where is my order?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_intent"] == "track_order"
    assert data["final_intent"] == "track_order"

    assert data["accepted"] is True
    assert data["status"] == "accepted"

    assert data["confidence"] > 0.90


def test_predict_fallback():
    response = client.post(
        "/predict",
        json={
            "text": "What is the weather today?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["final_intent"] == "fallback"
    assert data["accepted"] is False
    assert data["status"] == "fallback"


# ============================================================
# TOP-K TEST
# ============================================================

def test_predict_top_k():
    response = client.post(
        "/predict/top-k",
        json={
            "text": "Where is my package?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["top_k"] == 3
    assert len(data["predictions"]) == 3

    assert data["predictions"][0]["rank"] == 1

    assert (
        data["predictions"][0]["predicted_intent"]
        == "track_delivery"
    )


# ============================================================
# BATCH TEST
# ============================================================

def test_predict_batch():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "Where is my order?",
                "I want to cancel my order.",
                "My payment is not working.",
                "I want to return this product.",
                "What is the weather today?",
            ],
            "batch_size": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 5
    assert data["accepted"] == 3
    assert data["fallback"] == 2

    assert len(
        data["predictions"]
    ) == 5


# ============================================================
# NEGATIVE VALIDATION TESTS
# ============================================================

def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": ""
        },
    )

    assert response.status_code == 422


def test_predict_whitespace():
    response = client.post(
        "/predict",
        json={
            "text": "   "
        },
    )

    assert response.status_code == 422


def test_predict_missing_text():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_top_k_below_minimum():
    response = client.post(
        "/predict/top-k",
        json={
            "text": "Where is my order?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_top_k_above_maximum():
    response = client.post(
        "/predict/top-k",
        json={
            "text": "Where is my order?",
            "top_k": 11,
        },
    )

    assert response.status_code == 422


def test_empty_batch():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [],
            "batch_size": 32,
        },
    )

    assert response.status_code == 422


def test_blank_text_inside_batch():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "Where is my order?",
                "   ",
                "I want to cancel my order.",
            ],
            "batch_size": 3,
        },
    )

    assert response.status_code == 422


def test_invalid_batch_size():
    response = client.post(
        "/predict/batch",
        json={
            "texts": [
                "Where is my order?"
            ],
            "batch_size": 0,
        },
    )

    assert response.status_code == 422