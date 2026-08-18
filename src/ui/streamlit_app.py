"""
SupportPilot AI
Streamlit Production UI
"""

import json
import os

from urllib import request
from urllib.error import (
    HTTPError,
    URLError,
)

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv(
    "SUPPORTPILOT_API_URL",
    "http://127.0.0.1:8000",
)

# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_intent(intent: str) -> str:
    """
    Mengubah machine-readable intent
    menjadi label yang nyaman dibaca.
    """

    if not intent:
        return "-"

    if intent == "fallback":
        return "Fallback / Human Review"

    return (
        intent
        .replace("_", " ")
        .title()
    )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SupportPilot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .hero-subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.7rem;
    }

    .status-card {
        padding: 14px 16px;
        border-radius: 12px;
        background: rgba(0, 180, 100, 0.08);
        border: 1px solid rgba(0, 180, 100, 0.20);
        margin-bottom: 15px;
    }

    .result-card {
        border: 1px solid rgba(120,120,120,0.22);
        border-radius: 14px;
        padding: 18px;
        margin-top: 5px;
        min-height: 95px;
    }

    .result-label {
        font-size: 0.85rem;
        color: #777;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 1.45rem;
        font-weight: 650;
    }

    .small-muted {
        color: #888;
        font-size: 0.86rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120,120,120,0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTTP HELPER
# ============================================================

def api_get(
    endpoint: str,
    timeout: int = 10,
):
    """
    GET request ke FastAPI.
    """

    url = f"{API_BASE_URL}{endpoint}"

    try:

        with request.urlopen(
            url,
            timeout=timeout,
        ) as response:

            return json.loads(
                response
                .read()
                .decode("utf-8")
            )

    except (
        HTTPError,
        URLError,
        TimeoutError,
    ):

        return None


def api_post(
    endpoint: str,
    payload: dict,
    timeout: int = 30,
):
    """
    POST JSON request ke FastAPI.
    """

    url = f"{API_BASE_URL}{endpoint}"

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

    try:

        with request.urlopen(
            req,
            timeout=timeout,
        ) as response:

            return json.loads(
                response
                .read()
                .decode("utf-8")
            )

    except HTTPError as error:

        try:

            detail = json.loads(
                error
                .read()
                .decode("utf-8")
            )

        except Exception:

            detail = {
                "detail":
                    "API request gagal."
            }

        return {
            "_error": True,
            "_status_code":
                error.code,
            "_detail":
                detail,
        }

    except (
        URLError,
        TimeoutError,
    ):

        return {
            "_error": True,
            "_status_code": None,
            "_detail": {
                "detail":
                    "Tidak dapat terhubung ke API."
            },
        }

# ============================================================
# BATCH INFERENCE HELPER
# ============================================================

def run_batch_inference(
    texts: list[str],
    inference_batch_size: int = 32,
):
    """
    Menjalankan batch inference.

    FastAPI membatasi maksimal 100 message
    per request, sehingga input besar akan
    diproses dalam beberapa request.
    """

    all_predictions = []

    max_request_size = 100

    for start in range(
        0,
        len(texts),
        max_request_size,
    ):

        chunk = texts[
            start:
            start + max_request_size
        ]

        result = api_post(
            "/predict/batch",
            {
                "texts": chunk,
                "batch_size":
                    inference_batch_size,
            },
            timeout=60,
        )

        if (
            result is None
            or
            result.get("_error")
        ):
            return None

        all_predictions.extend(
            result["predictions"]
        )

    return all_predictions

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-title">
        🤖 SupportPilot AI
    </div>

    <div class="hero-subtitle">
        AI-powered Customer Support Intent Classification
        using DistilBERT
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Status"
    )

    health = api_get(
        "/health"
    )

    model_info = None

    if (
        health
        and
        health.get("status")
        == "healthy"
    ):

        model_info = api_get(
            "/model-info"
        )

        st.success(
            "● API Online"
        )

        col_a, col_b = st.columns(
            2
        )

        with col_a:

            st.metric(
                "Classes",
                health.get(
                    "num_labels",
                    "-"
                ),
            )

        with col_b:

            st.metric(
                "Device",
                str(
                    health.get(
                        "device",
                        "-"
                    )
                ).upper(),
            )

        if model_info:

            st.caption(
                "Production Model"
            )

            st.code(
                model_info.get(
                    "model_name",
                    "Unknown"
                ),
                language=None,
            )

    else:

        st.error(
            "● API Offline"
        )

        st.warning(
            "Jalankan Docker Compose "
            "terlebih dahulu."
        )

    st.divider()

    st.caption(
        "API Endpoint"
    )

    st.code(
        API_BASE_URL,
        language=None,
    )

    st.divider()

    st.caption(
        "Production Policy"
    )

    if model_info:

        st.write(
            "Confidence threshold:",
            f"{model_info['min_confidence'] * 100:.0f}%"
        )

        st.write(
            "Margin threshold:",
            f"{model_info['min_margin'] * 100:.0f}%"
        )


# ============================================================
# NAVIGATION
# ============================================================

tab_analyze, tab_top_k, tab_batch = st.tabs(
    [
        "🎯 Intent Analyzer",
        "📊 Top-K Analysis",
        "📁 Batch Analysis",
    ]
)


# ============================================================
# TAB 1 — INTENT ANALYZER
# ============================================================

with tab_analyze:

    st.subheader(
        "Analyze Customer Message"
    )

    st.caption(
        "Klasifikasikan pesan customer dan "
        "lihat confidence model."
    )

    with st.form(
        "single_prediction_form"
    ):

        customer_message = st.text_area(
            "Customer Message",
            placeholder=(
                "Example: "
                "Where is my order?"
            ),
            height=135,
            max_chars=500,
        )

        submitted = st.form_submit_button(
            "Analyze Message",
            type="primary",
            use_container_width=True,
        )


    if submitted:

        text = customer_message.strip()

        if not text:

            st.warning(
                "Masukkan pesan customer "
                "terlebih dahulu."
            )

        else:

            with st.spinner(
                "DistilBERT sedang "
                "menganalisis pesan..."
            ):

                result = api_post(
                    "/predict",
                    {
                        "text": text
                    },
                )

            if (
                result is None
                or
                result.get("_error")
            ):

                st.error(
                    "Prediction gagal. "
                    "Pastikan FastAPI aktif."
                )

            else:

                st.divider()

                # ============================================
                # RESULT STATUS
                # ============================================

                if result["accepted"]:

                    st.success(
                        "✓ Prediction Accepted"
                    )

                else:

                    st.warning(
                        "⚠ Prediction Fallback — "
                        "Human review recommended"
                    )


                # ============================================
                # MAIN METRICS
                # ============================================

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    st.metric(
                        "Final Intent",
                        format_intent(
                            result[
                                "final_intent"
                            ]
                        ),
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        (
                            f"{result['confidence_percent']:.2f}%"
                        ),
                    )

                with col3:

                    st.metric(
                        "Confidence Margin",
                        (
                            f"{result['confidence_margin_percent']:.2f}%"
                        ),
                    )


                # ============================================
                # CONFIDENCE BAR
                # ============================================

                st.progress(
                    min(
                        max(
                            float(
                                result[
                                    "confidence"
                                ]
                            ),
                            0.0,
                        ),
                        1.0,
                    ),
                    text=(
                        "Model Confidence "
                        f"{result['confidence_percent']:.2f}%"
                    ),
                )


                # ============================================
                # PREDICTION DETAILS
                # ============================================
                
                st.markdown(
                    "### Prediction Details"
                )
                
                detail1, detail2 = st.columns(2)
                
                with detail1:
                
                    st.metric(
                        "Raw Prediction",
                        format_intent(
                            result["predicted_intent"]
                        ),
                    )
                
                with detail2:
                
                    second_best_percent = float(
                        result[
                            "second_best_confidence_percent"
                        ]
                    )
                
                    st.metric(
                        "Second-best Intent",
                        format_intent(
                            result[
                                "second_best_intent"
                            ]
                        ),
                    )
                
                    st.caption(
                        (
                            "Model confidence: "
                            f"{second_best_percent:.4f}%"
                        )
                    )


                # ============================================
                # POLICY
                # ============================================

                with st.expander(
                    "Confidence Policy & Decision"
                ):

                    p1, p2 = st.columns(
                        2
                    )

                    with p1:

                        st.metric(
                            "Minimum Confidence",
                            (
                                f"{result['min_confidence'] * 100:.0f}%"
                            ),
                        )

                    with p2:

                        st.metric(
                            "Minimum Margin",
                            (
                                f"{result['min_margin'] * 100:.0f}%"
                            ),
                        )

                    st.write(
                        "**Status:**",
                        result["status"],
                    )

                    st.write(
                        "**Decision reason:**",
                        result["reason"],
                    )


# ============================================================
# TAB 2 — TOP-K ANALYSIS
# ============================================================

with tab_top_k:

    st.subheader(
        "Top-K Intent Analysis"
    )

    st.caption(
        "Lihat beberapa intent dengan "
        "probabilitas tertinggi."
    )

    with st.form(
        "top_k_form"
    ):

        top_k_text = st.text_area(
            "Customer Message",
            placeholder=(
                "Example: "
                "Where is my package?"
            ),
            height=120,
            max_chars=500,
            key="top_k_text",
        )

        top_k_value = st.slider(
            "Number of candidates",
            min_value=2,
            max_value=10,
            value=5,
        )

        top_k_submitted = (
            st.form_submit_button(
                "Analyze Top-K",
                type="primary",
                use_container_width=True,
            )
        )


    if top_k_submitted:

        text = top_k_text.strip()

        if not text:

            st.warning(
                "Masukkan pesan customer "
                "terlebih dahulu."
            )

        else:

            with st.spinner(
                "Mengambil Top-K intent..."
            ):

                top_result = api_post(
                    "/predict/top-k",
                    {
                        "text": text,
                        "top_k":
                            top_k_value,
                    },
                )

            if (
                top_result is None
                or
                top_result.get(
                    "_error"
                )
            ):

                st.error(
                    "Top-K prediction gagal."
                )

            else:

                predictions = (
                    top_result[
                        "predictions"
                    ]
                )

                st.divider()

                # ============================================
                # TOP RESULT
                # ============================================

                top_prediction = (
                    predictions[0]
                )

                m1, m2 = st.columns(
                    2
                )

                with m1:
                
                    st.metric(
                        "Top Intent",
                        format_intent(
                            top_prediction[
                                "predicted_intent"
                            ]
                        ),
                    )

                with m2:

                    st.metric(
                        "Top Confidence",
                        (
                            f"{top_prediction['confidence_percent']:.2f}%"
                        ),
                    )


                # ============================================
                # CHART DATA
                # ============================================

                chart_df = pd.DataFrame(
                    {
                        "Intent": [
                            format_intent(
                                item[
                                    "predicted_intent"
                                ]
                            )
                            for item in predictions
                        ],
                        "Confidence": [
                            float(
                                item[
                                    "confidence_percent"
                                ]
                            )
                            for item in predictions
                        ],
                    }
                )
                
                st.markdown(
                    "### Intent Confidence Ranking"
                )
                
                st.vega_lite_chart(
                    chart_df,
                    {
                        "mark": {
                            "type": "bar",
                            "cornerRadiusEnd": 5,
                        },
                        "encoding": {
                            "x": {
                                "field": "Confidence",
                                "type": "quantitative",
                                "title": "Confidence (%)",
                                "scale": {
                                    "domain": [0, 100]
                                },
                            },
                            "y": {
                                "field": "Intent",
                                "type": "nominal",
                                "title": None,
                                "sort": "-x",
                            },
                            "tooltip": [
                                {
                                    "field": "Intent",
                                    "type": "nominal",
                                    "title": "Intent",
                                },
                                {
                                    "field": "Confidence",
                                    "type": "quantitative",
                                    "title": "Confidence",
                                    "format": ".4f",
                                },
                            ],
                        },
                    },
                    width="stretch",
                    height=280,
                )


                # ============================================
                # TABLE
                # ============================================

                table_df = pd.DataFrame(
                    [
                        {
                            "Rank":
                                item["rank"],

                            "Intent":
                                format_intent(
                                    item[
                                        "predicted_intent"
                                    ]
                                ),

                            "Confidence":
                                (
                                    f"{item['confidence_percent']:.4f}%"
                                ),
                        }

                        for item
                        in predictions
                    ]
                )

                st.markdown(
                    "### Candidate Details"
                )

                st.dataframe(
                    table_df,
                    use_container_width=True,
                    hide_index=True,
                )
                
# ============================================================
# TAB 3 — BATCH ANALYSIS
# ============================================================

with tab_batch:

    st.subheader(
        "Batch Customer Message Analysis"
    )

    st.caption(
        "Klasifikasikan banyak customer message "
        "sekaligus menggunakan DistilBERT."
    )

    input_mode = st.radio(
        "Input Method",
        [
            "Paste Messages",
            "Upload CSV",
        ],
        horizontal=True,
    )


    # ========================================================
    # MODE 1 — PASTE MESSAGE
    # ========================================================

    if input_mode == "Paste Messages":

        batch_text = st.text_area(
            "Customer Messages",
            placeholder=(
                "Masukkan satu pesan per baris:\n\n"
                "Where is my order?\n"
                "I want to cancel my order.\n"
                "My payment is not working.\n"
                "What is the weather today?"
            ),
            height=220,
        )

        st.caption(
            "Satu baris = satu customer message."
        )

        analyze_batch = st.button(
            "Analyze Batch",
            type="primary",
            use_container_width=True,
        )

        messages = []

        if analyze_batch:

            messages = [
                line.strip()
                for line
                in batch_text.splitlines()
                if line.strip()
            ]


    # ========================================================
    # MODE 2 — UPLOAD CSV
    # ========================================================

    else:

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help=(
                "CSV harus memiliki minimal "
                "satu kolom berisi customer message."
            ),
        )

        messages = []
        analyze_batch = False

        if uploaded_file is not None:

            try:

                upload_df = pd.read_csv(
                    uploaded_file
                )

                st.success(
                    (
                        f"CSV loaded: "
                        f"{len(upload_df):,} rows"
                    )
                )

                st.dataframe(
                    upload_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                column_name = st.selectbox(
                    "Select message column",
                    options=list(
                        upload_df.columns
                    ),
                )

                analyze_batch = st.button(
                    "Analyze Uploaded CSV",
                    type="primary",
                    use_container_width=True,
                )

                if analyze_batch:

                    messages = (
                        upload_df[column_name]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )

                    messages = [
                        text
                        for text
                        in messages.tolist()
                        if text
                    ]

            except Exception as error:

                st.error(
                    (
                        "CSV tidak dapat dibaca: "
                        f"{error}"
                    )
                )


    # ========================================================
    # RUN BATCH INFERENCE
    # ========================================================

    if analyze_batch:

        if not messages:

            st.warning(
                "Tidak ada customer message "
                "yang dapat dianalisis."
            )

        else:

            st.info(
                (
                    f"Total messages: "
                    f"{len(messages):,}"
                )
            )

            with st.spinner(
                (
                    "DistilBERT sedang "
                    f"menganalisis "
                    f"{len(messages):,} messages..."
                )
            ):

                batch_predictions = (
                    run_batch_inference(
                        messages,
                        inference_batch_size=32,
                    )
                )

            if batch_predictions is None:

                st.error(
                    "Batch inference gagal. "
                    "Pastikan FastAPI aktif."
                )

            else:

                # ============================================
                # CREATE DATAFRAME
                # ============================================

                result_df = pd.DataFrame(
                    batch_predictions
                )

                total_messages = len(
                    result_df
                )

                accepted_count = int(
                    result_df[
                        "accepted"
                    ].sum()
                )

                fallback_count = (
                    total_messages
                    - accepted_count
                )

                accepted_percent = (
                    accepted_count
                    / total_messages
                    * 100
                )


                # ============================================
                # SUMMARY
                # ============================================

                st.divider()

                st.markdown(
                    "### Batch Summary"
                )

                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                with c1:

                    st.metric(
                        "Total Messages",
                        f"{total_messages:,}",
                    )

                with c2:

                    st.metric(
                        "Accepted",
                        f"{accepted_count:,}",
                    )

                with c3:

                    st.metric(
                        "Fallback",
                        f"{fallback_count:,}",
                    )

                with c4:

                    st.metric(
                        "Acceptance Rate",
                        (
                            f"{accepted_percent:.2f}%"
                        ),
                    )


                # ============================================
                # CLEAN RESULT TABLE
                # ============================================

                display_df = result_df[
                    [
                        "text",
                        "predicted_intent",
                        "final_intent",
                        "confidence_percent",
                        "confidence_margin_percent",
                        "status",
                    ]
                ].copy()

                display_df.columns = [
                    "Customer Message",
                    "Raw Intent",
                    "Final Intent",
                    "Confidence (%)",
                    "Margin (%)",
                    "Status",
                ]
                
                display_df[
                    "Raw Intent"
                ] = (
                    display_df[
                        "Raw Intent"
                    ]
                    .apply(
                        format_intent
                    )
                )
                
                display_df[
                    "Final Intent"
                ] = (
                    display_df[
                        "Final Intent"
                    ]
                    .apply(
                        format_intent
                    )
                )
                
                display_df[
                    "Status"
                ] = (
                    display_df[
                        "Status"
                    ]
                    .replace(
                        {
                            "accepted":
                                "Accepted",
                
                            "fallback":
                                "Human Review",
                        }
                    )
                )

                display_df[
                    "Confidence (%)"
                ] = (
                    display_df[
                        "Confidence (%)"
                    ]
                    .round(2)
                )

                display_df[
                    "Margin (%)"
                ] = (
                    display_df[
                        "Margin (%)"
                    ]
                    .round(2)
                )


                # ============================================
                # RESULTS
                # ============================================

                st.markdown(
                    "### Classification Results"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )


                # ============================================
                # INTENT DISTRIBUTION
                # ============================================

                intent_distribution = (
                    result_df[
                        "final_intent"
                    ]
                    .apply(
                        format_intent
                    )
                    .value_counts()
                    .rename_axis(
                        "Intent"
                    )
                    .reset_index(
                        name="Count"
                    )
                )

                st.markdown(
                    "### Intent Distribution"
                )

                st.bar_chart(
                    intent_distribution,
                    x="Intent",
                    y="Count",
                )


                # ============================================
                # DOWNLOAD
                # ============================================

                csv_result = (
                    display_df
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    label=(
                        "⬇ Download Classification Results"
                    ),
                    data=csv_result,
                    file_name=(
                        "supportpilot_batch_predictions.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )