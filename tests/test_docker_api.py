"""
SupportPilot AI
Docker Integration Tests

Tests ini menguji API yang benar-benar berjalan
di dalam Docker container.
"""

import json
import os
from urllib import request
from urllib.error import HTTPError


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = os.getenv(
    "SUPPORTPILOT_DOCKER_API_URL",
    "http://127.0.0.1:8001",
)


# ============================================================
# HTTP HELPERS
# ============================================================

def get_json(path: str):
    """
    GET request dan parse JSON response.
    """

    url = f"{BASE_URL}{path}"

    with request.urlopen(
        url,
        timeout=10,
    ) as response:

        return (
            response.status,
            json.loads(
                response.read().decode(
                    "utf-8"
                )
            ),
        )


def post_json(
    path: str,
    payload: dict,
):
    """
    POST JSON request dan parse JSON response.
    """

    url = f"{BASE_URL}{path}"

    body = json.dumps(
        payload
    ).encode("utf-8")

    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

    with request.urlopen(
        req,
        timeout=30,
    ) as response:

        return (
            response.status,
            json.loads(
                response.read().decode(
                    "utf-8"
                )
            ),
        )


# ============================================================
# HEALTH TEST
# ============================================================

def test_docker_health():

    status, data = get_json(
        "/health"
    )

    assert status == 200

    assert (
        data["status"]
        == "healthy"
    )

    assert (
        data["model_loaded"]
        is True
    )

    assert (
        data["device"]
        == "cpu"
    )

    assert (
        data["num_labels"]
        == 46
    )


# ============================================================
# MODEL INFO TEST
# ============================================================

def test_docker_model_info():

    status, data = get_json(
        "/model-info"
    )

    assert status == 200

    assert (
        data["model_name"]
        ==
        "DistilBertForSequenceClassification"
    )

    assert (
        data["device"]
        == "cpu"
    )

    assert data["gpu"] is None

    assert (
        data["num_labels"]
        == 46
    )

    assert (
        data["max_length"]
        == 64
    )


# ============================================================
# PREDICTION TEST
# ============================================================

def test_docker_prediction():

    status, data = post_json(
        "/predict",
        {
            "text":
                "Where is my order?"
        },
    )

    assert status == 200

    assert (
        data["predicted_intent"]
        == "track_order"
    )

    assert (
        data["final_intent"]
        == "track_order"
    )

    assert (
        data["accepted"]
        is True
    )

    assert (
        data["status"]
        == "accepted"
    )

    assert (
        data["confidence"]
        > 0.90
    )


# ============================================================
# FALLBACK TEST
# ============================================================

def test_docker_fallback():

    status, data = post_json(
        "/predict",
        {
            "text":
                "What is the weather today?"
        },
    )

    assert status == 200

    assert (
        data["final_intent"]
        == "fallback"
    )

    assert (
        data["accepted"]
        is False
    )

    assert (
        data["status"]
        == "fallback"
    )


# ============================================================
# TOP-K TEST
# ============================================================

def test_docker_top_k():

    status, data = post_json(
        "/predict/top-k",
        {
            "text":
                "Where is my package?",
            "top_k": 3,
        },
    )

    assert status == 200

    assert (
        data["top_k"]
        == 3
    )

    assert (
        len(
            data["predictions"]
        )
        == 3
    )

    assert (
        data["predictions"][0]
        ["predicted_intent"]
        == "track_delivery"
    )


# ============================================================
# BATCH TEST
# ============================================================

def test_docker_batch():

    status, data = post_json(
        "/predict/batch",
        {
            "texts": [
                "Where is my order?",
                "I want to cancel my order.",
                "What is the weather today?",
            ],
            "batch_size": 3,
        },
    )

    assert status == 200

    assert (
        data["total"]
        == 3
    )

    assert (
        len(
            data["predictions"]
        )
        == 3
    )