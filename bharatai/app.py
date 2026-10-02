"""
BharatAI — Government Website Intelligence Platform
Main Streamlit Application Entrypoint
Agentic RAG with Multiple Knowledge Sources and Three JEV Decision Layers.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
from config import GROQ_MODEL
from core.llm_client import get_groq_client
from core.vector_store import get_vector_store
from database.db import get_db

# UI Page Components
from ui.chat_page import render_chat_page
from ui.website_page import render_website_page
from ui.knowledge_page import render_knowledge_page
from ui.evaluation_page import render_evaluation_page
from ui.test_lab_page import render_test_lab_page
from ui.settings_page import render_settings_page

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="BharatAI — Government Website Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Professional Indian Tricolor Theme: Navy Blue #0A2540, Saffron #FF9933, Green #138808)
st.markdown(
    """
    <style>
    /* Main container styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* Top Header Bar */
    .bharat-header {
        background: linear-gradient(90deg, #0A2540 0%, #1e3a5f 70%, #ff9933 100%);
        padding: 18px 24px;
        border-radius: 8px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .bharat-title {
        font-size: 24px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .bharat-subtitle {
        font-size: 13px;
        color: #e2e8f0;
        margin-top: 4px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #0A2540;
    }

    /* Primary Buttons */
    button[kind="primary"] {
        background-color: #0A2540 !important;
        border-color: #0A2540 !important;
    }
    button[kind="primary"]:hover {
        background-color: #ff9933 !important;
        border-color: #ff9933 !important;
        color: white !important;
    }

    /* Badge Pills */
    .badge-accept {
        background-color: #138808;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    db = get_db()
    groq_client = get_groq_client()
    vector_store = get_vector_store()

    # Load stored API key if present in DB and not currently configured
    if not groq_client.is_configured():
        stored_key = db.get_setting("GROQ_API_KEY")
        if stored_key:
            groq_client.set_credentials(stored_key, db.get_setting("GROQ_MODEL", GROQ_MODEL))

    # Top Application Header
    st.markdown(
        """
        <div class="bharat-header">
            <div class="bharat-title">🏛️ BharatAI — Government Website Intelligence Platform</div>
            <div class="bharat-subtitle">Agentic Retrieval-Augmented Generation (RAG) with Three JEV Decision Layers • Powered by Groq</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg",
            width=60,
        )
        st.markdown("### **BharatAI Navigation**")

        page_choice = st.radio(
            "Select Module",
            options=[
                "💬 Chat Assistant",
                "🌐 Website Management",
                "📚 Knowledge Base",
                "📊 Evaluation Dashboard",
                "🔬 Test Lab",
                "⚙️ Settings",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### **System Telemetry**")

        # Groq connection status indicator
        if groq_client.is_configured():
            st.markdown("🟢 **Groq LLM:** Configured")
            st.caption(f"Model: `{groq_client.model}`")
        else:
            st.markdown("🟡 **Groq LLM:** Key Required")
            st.caption("Please configure in Settings")

        st.markdown(f"📦 **Vector Chunks:** `{vector_store.count()}`")
        st.markdown("🏛️ **Registry:** `10 Portals Active`")

        st.markdown("---")
        st.caption("Government of India RAG Architecture  \n*Strictly Hosted on Groq*")

    # Route to Selected Page
    if page_choice == "💬 Chat Assistant":
        render_chat_page()
    elif page_choice == "🌐 Website Management":
        render_website_page()
    elif page_choice == "📚 Knowledge Base":
        render_knowledge_page()
    elif page_choice == "📊 Evaluation Dashboard":
        render_evaluation_page()
    elif page_choice == "🔬 Test Lab":
        render_test_lab_page()
    elif page_choice == "⚙️ Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
