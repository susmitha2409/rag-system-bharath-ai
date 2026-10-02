"""
BharatAI — Chat Assistant Page (Page 1)
Interactive citizen interface with source selection, mode toggles, expandable citations,
translation indicators, and feedback capture.
"""

import json
import streamlit as st
from config import OFFICIAL_WEBSITES
from core.rag_pipeline import get_rag_pipeline
from database.db import get_db


def render_chat_page():
    st.markdown("## 🏛️ BharatAI Government Information Assistant")
    st.caption("Authoritative Retrieval-Augmented Generation across 10 official Government of India portals.")

    db = get_db()
    pipeline = get_rag_pipeline()

    # Session State initialization for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None

    # Sidebar / Column Controls
    with st.expander("⚙️ Query Configuration & Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            retrieval_mode = st.radio(
                "Retrieval Mode",
                options=["HYBRID", "INDEXED", "LIVE"],
                index=0,
                help="HYBRID combines indexed vector database with live official web feeds.",
            )
        with col2:
            response_lang = st.selectbox(
                "Response Language",
                options=["en", "hi", "ta", "te", "bn", "mr"],
                format_func=lambda x: {
                    "en": "English (Default)",
                    "hi": "Hindi (हिंदी)",
                    "ta": "Tamil (தமிழ்)",
                    "te": "Telugu (తెలుగు)",
                    "bn": "Bengali (বাংলা)",
                    "mr": "Marathi (मराठी)",
                }.get(x, x),
            )
        with col3:
            top_k = st.slider("Top Sources (K)", min_value=1, max_value=8, value=4)
            relevance_threshold = st.slider("Relevance Cutoff", min_value=0.1, max_value=0.8, value=0.30, step=0.05)

        site_options = ["All Websites"] + [s["name"] for s in OFFICIAL_WEBSITES]
        selected_site_names = st.multiselect("Filter by Ministry / Portal", options=site_options, default=["All Websites"])

    # Map selected site names to site IDs
    selected_site_ids = []
    if "All Websites" not in selected_site_names:
        for s in OFFICIAL_WEBSITES:
            if s["name"] in selected_site_names:
                selected_site_ids.append(s["id"])

    # Clear chat button
    c_col1, c_col2 = st.columns([6, 1])
    with c_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_query_result = None
            st.rerun()

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander(f"📚 Verified Sources ({len(msg['citations'])})", expanded=False):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"**{cite['citation_id']} [{cite['title']}]({cite['url']})**  \n"
                            f"🏢 *{cite['department']}* | 🗓️ *{cite['publication_date']}* | 🎯 Score: `{cite.get('similarity_score', 0):.2f}`"
                        )
                        if cite.get("is_translated"):
                            st.caption("🌐 *Machine-translated from original Indian language text into English.*")
                        st.text(cite.get("passage_snippet", ""))

    # Query Input
    user_query = st.chat_input("Ask a question about government policies, schemes, notifications, or current news...")

    if user_query:
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute RAG Pipeline with spinner
        with st.chat_message("assistant"):
            with st.spinner("Retrieving official records & generating grounded response..."):
                result = pipeline.run(
                    query=user_query,
                    user_selected_sites=selected_site_ids if selected_site_ids else None,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    relevance_threshold=relevance_threshold,
                    response_language=response_lang,
                )

            st.session_state.last_query_result = result
            answer_text = result["answer"]
            citations = result.get("citations", [])
            decision = result.get("quality_decision", "ACCEPT")

            # Decision Badge
            badge_color = "#138808" if decision == "ACCEPT" else ("#ff9933" if decision == "RETRY" else "#d9534f")
            st.markdown(
                f"<span style='background-color:{badge_color}; color:white; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:bold;'>"
                f"JEV GATE: {decision}</span> &nbsp; "
                f"<span style='color:#666; font-size:12px;'>Latency: {result['trace']['total_latency_ms']:.0f} ms | Model: {result['trace'].get('groq_model', 'Groq')}</span>",
                unsafe_allow_html=True,
            )

            st.markdown(answer_text)

            # Display citations and source passages
            if citations:
                with st.expander(f"📚 Verified Citations ({len(citations)})", expanded=True):
                    for cite in citations:
                        st.markdown(
                            f"**{cite['citation_id']} [{cite['title']}]({cite['url']})**  \n"
                            f"🏢 *{cite['department']}* | 🗓️ *{cite['publication_date']}* | 🎯 Score: `{cite.get('similarity_score', 0):.2f}`"
                        )
                        if cite.get("is_translated"):
                            st.info("🌐 *Passage translated from Indian language original by Groq.*")
                        st.code(cite.get("passage_snippet", ""), language=None)

            # Record in session state
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_text,
                "citations": citations,
                "request_id": result["request_id"],
            })

    # Feedback and Download for last query
    if st.session_state.last_query_result:
        res = st.session_state.last_query_result
        req_id = res["request_id"]
        st.markdown("---")
        fb_col1, fb_col2, fb_col3, dl_col = st.columns([1.5, 1.5, 4, 3])

        with fb_col1:
            if st.button("👍 Helpful", key=f"fb_pos_{req_id}"):
                db.record_user_feedback(req_id, 1, "User marked response as helpful")
                st.toast("Thank you for your feedback!", icon="✅")

        with fb_col2:
            if st.button("👎 Not Helpful", key=f"fb_neg_{req_id}"):
                db.record_user_feedback(req_id, -1, "User marked response as unhelpful")
                st.toast("Feedback recorded.", icon="📝")

        with dl_col:
            download_data = json.dumps(res, indent=2)
            st.download_button(
                label="📥 Export Result (JSON)",
                data=download_data,
                file_name=f"bharatai_answer_{req_id[:8]}.json",
                mime="application/json",
            )
