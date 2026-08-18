"""
SupportPilot AI
Docker Streamlit UI integration tests
"""

import os
import urllib.request

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE_UI = (
    PROJECT_ROOT
    / "dockerfile.ui"
)

COMPOSE_FILE = (
    PROJECT_ROOT
    / "docker-compose.yml"
)


UI_BASE_URL = os.getenv(
    "SUPPORTPILOT_UI_TEST_URL",
    "http://127.0.0.1:8501",
)


# ============================================================
# HELPERS
# ============================================================

def http_get(
    endpoint: str,
    timeout: int = 15,
):
    url = f"{UI_BASE_URL}{endpoint}"

    with urllib.request.urlopen(
        url,
        timeout=timeout,
    ) as response:

        body = response.read()

        return (
            response.status,
            response.headers,
            body,
        )


# ============================================================
# UI HEALTH
# ============================================================

def test_docker_ui_health():

    status, _, body = http_get(
        "/_stcore/health"
    )

    assert status == 200

    assert (
        body
        .decode("utf-8")
        .strip()
        .lower()
        == "ok"
    )


# ============================================================
# UI ROOT PAGE
# ============================================================

def test_docker_ui_root():

    status, headers, body = http_get(
        "/"
    )

    assert status == 200

    content_type = (
        headers
        .get("Content-Type", "")
        .lower()
    )

    assert "text/html" in content_type

    html = (
        body
        .decode(
            "utf-8",
            errors="ignore",
        )
        .lower()
    )

    assert "<html" in html


# ============================================================
# NON-ROOT CONTAINER CONFIGURATION
# ============================================================

def test_docker_ui_non_root():

    dockerfile_content = (
        DOCKERFILE_UI
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "USER appuser"
        in dockerfile_content
    )
# ============================================================
# INTERNAL API CONFIGURATION
# ============================================================

def test_docker_ui_api_url():

    compose_content = (
        COMPOSE_FILE
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "SUPPORTPILOT_API_URL: http://api:8000"
        in compose_content
    )