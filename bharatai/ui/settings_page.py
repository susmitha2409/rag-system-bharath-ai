"""
BharatAI — Settings Page (Page 6)
Configuration of Groq LLM credentials, connection diagnostics, embedding models,
crawler parameters, and retrieval thresholds.
"""

import streamlit as st
from config import (
    GROQ_API_URL,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    GROQ_MAX_RETRIES,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
    RELEVANCE_THRESHOLD,
    CRAWLER_MAX_PAGES_PER_SITE,
    CRAWLER_MAX_DEPTH,
    CRAWLER_REQUEST_DELAY,
)
from core.llm_client import get_groq_client
from database.db import get_db


def render_settings_page():
    st.markdown("## ⚙️ System Settings & Groq Configuration")
    st.caption("Configure credentials, inference parameters, chunking windows, and crawler limits.")

    db = get_db()
    groq_client = get_groq_client()

    # -------------------------------------------------------------
    # SECTION 1: GROQ API CONFIGURATION (MANDATORY & SOLE HOSTED LLM)
    # -------------------------------------------------------------
    st.markdown("### 🔑 Groq LLM Host Configuration")
    st.info("ℹ️ **BharatAI Architecture Policy:** Groq is the sole hosted LLM provider. No other hosted provider is permitted.")

    col1, col2 = st.columns(2)
    with col1:
        # Load persisted key from DB if any
        stored_key = db.get_setting("GROQ_API_KEY", groq_client.api_key)
        new_api_key = st.text_input(
            "Groq API Key",
            value=stored_key if stored_key else "",
            type="password",
            help="Enter your personal Groq API key (starts with gsk_). Never committed to source.",
        )

        groq_model = st.text_input(
            "Groq Model ID",
            value=db.get_setting("GROQ_MODEL", groq_client.model),
            help="Supported models: llama-3.3-70b-versatile, llama3-70b-8192, llama-3.1-8b-instant, mixtral-8x7b-32768",
        )

    with col2:
        groq_url = st.text_input("Groq API Endpoint", value=GROQ_API_URL, disabled=True)
        timeout_sec = st.number_input("Request Timeout (seconds)", min_value=5, max_value=120, value=GROQ_TIMEOUT_SECONDS)

    # Save and Test Buttons
    c_btn1, c_btn2 = st.columns([2, 3])
    with c_btn1:
        if st.button("💾 Save Credentials", type="primary", use_container_width=True):
            if new_api_key.strip():
                groq_client.set_credentials(api_key=new_api_key.strip(), model=groq_model.strip())
                db.set_setting("GROQ_API_KEY", new_api_key.strip())
                db.set_setting("GROQ_MODEL", groq_model.strip())
                st.success("Credentials saved securely in session and local settings store!")
            else:
                st.warning("Please provide a valid API key.")

    with c_btn2:
        test_conn_btn = st.button("🔌 Test Groq Connection", use_container_width=True)

    if test_conn_btn:
        if new_api_key.strip():
            groq_client.set_credentials(api_key=new_api_key.strip(), model=groq_model.strip())

        with st.spinner("Testing connection to Groq API..."):
            diag = groq_client.test_connection()

        if diag["success"]:
            st.success(f"✅ {diag['message']} (Latency: {diag['latency_ms']} ms)")
        else:
            st.error(f"❌ Connection Failed: {diag['message']} (Status: {diag['status']})")

    st.markdown("---")

    # -------------------------------------------------------------
    # SECTION 2: INFERENCE PARAMETERS
    # -------------------------------------------------------------
    st.markdown("### 🎛️ Groq Generation Parameters")
    p1, p2, p3 = st.columns(3)
    with p1:
        temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=GROQ_TEMPERATURE, step=0.05)
    with p2:
        max_tokens = st.slider("Max Output Tokens", min_value=256, max_value=4096, value=GROQ_MAX_TOKENS, step=128)
    with p3:
        max_retries = st.slider("API Retry Limit", min_value=1, max_value=5, value=GROQ_MAX_RETRIES)

    # -------------------------------------------------------------
    # SECTION 3: EMBEDDING & VECTOR DATABASE
    # -------------------------------------------------------------
    st.markdown("### 🧠 Local Embedding & Vector Store")
    e1, e2 = st.columns(2)
    with e1:
        st.text_input("Embedding Model (Local Sentence-Transformers)", value=EMBEDDING_MODEL_NAME, disabled=True)
        st.caption("Runs 100% locally on CPU/GPU without third-party embedding APIs.")
    with e2:
        st.text_input("Vector Database Path", value="./data/chroma_db", disabled=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        chunk_sz = st.number_input("Chunk Size (characters)", min_value=200, max_value=2000, value=CHUNK_SIZE, step=50)
    with c2:
        chunk_ov = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=400, value=CHUNK_OVERLAP, step=25)
    with c3:
        def_k = st.number_input("Default Top-K Chunks", min_value=1, max_value=10, value=RETRIEVAL_TOP_K)

    # -------------------------------------------------------------
    # SECTION 4: WEB CRAWLER SETTINGS
    # -------------------------------------------------------------
    st.markdown("### 🕷️ Crawler Policy & Rate Limits")
    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        st.number_input("Max Pages Per Site", min_value=5, max_value=100, value=CRAWLER_MAX_PAGES_PER_SITE)
    with cr2:
        st.number_input("Max Crawl Depth", min_value=1, max_value=5, value=CRAWLER_MAX_DEPTH)
    with cr3:
        st.number_input("Request Delay (seconds)", min_value=0.5, max_value=5.0, value=CRAWLER_REQUEST_DELAY, step=0.5)

    st.caption("The crawler strictly adheres to robots.txt and restricts requests to the 10 approved Government of India domains.")
