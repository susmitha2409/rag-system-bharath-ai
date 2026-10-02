"""
BharatAI — Test Lab (Page 5)
Deep-dive inspection suite to step through each of the 3 JEV Decision Layers,
review raw embeddings, candidate chunks, similarity scores, prompts, and gate rationale.
"""

import json
import streamlit as st
from config import OFFICIAL_WEBSITES
from core.evaluation import BENCHMARK_DATASET
from core.rag_pipeline import get_rag_pipeline


def render_test_lab_page():
    st.markdown("## 🔬 JEV Architecture Test Lab")
    st.caption("Inspect each decision layer, prompt construction, vector similarity scores, and gate verdicts.")

    pipeline = get_rag_pipeline()

    # Preset Sample Queries from 10 Ministries
    sample_options = ["Custom Query"] + [f"[{q['target_website_id'].upper()}] {q['question']}" for q in BENCHMARK_DATASET]
    chosen_preset = st.selectbox("Select Sample Official Question", options=sample_options)

    default_q = ""
    default_site = []
    if chosen_preset != "Custom Query":
        idx = sample_options.index(chosen_preset) - 1
        default_q = BENCHMARK_DATASET[idx]["question"]
        default_site = [BENCHMARK_DATASET[idx]["target_website_id"]]

    # Input Form
    with st.form("test_lab_form"):
        query = st.text_area("Question under test", value=default_q, height=70)
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox("Forced Retrieval Mode", options=["HYBRID", "INDEXED", "LIVE"], index=0)
        with c2:
            k = st.slider("Top K", min_value=1, max_value=8, value=4)
        with c3:
            threshold = st.slider("Relevance Cutoff", min_value=0.1, max_value=0.7, value=0.30, step=0.05)

        run_btn = st.form_submit_button("🧪 Execute Test Run", type="primary")

    if run_btn and query.strip():
        with st.spinner("Executing full JEV multi-stage pipeline..."):
            out = pipeline.run(
                query=query.strip(),
                user_selected_sites=default_site if default_site else None,
                retrieval_mode=mode,
                top_k=k,
                relevance_threshold=threshold,
            )

        st.success(f"Execution complete in {out['trace']['total_latency_ms']:.1f} ms!")

        # -------------------------------------------------------------
        # STEP 1: JEV DECISION LAYER 1
        # -------------------------------------------------------------
        with st.expander("🔹 JEV Decision Layer 1: Tool & Source Selection", expanded=True):
            st.markdown(f"**Detected Intent:** `{out['trace']['detected_intent']}`")
            st.markdown(f"**Identified Department:** `{out['trace']['detected_department']}`")
            st.markdown(f"**Time Sensitive Flag:** `{bool(out['trace']['is_time_sensitive'])}`")
            st.markdown(f"**Selected Knowledge Sources:** `{out['selected_websites']}`")
            st.markdown(f"**Selection Rationale:** {out['trace']['routing_rationale']}")
            st.caption(f"⏱️ Stage Latency: {out['trace']['source_selection_latency_ms']:.1f} ms")

        # -------------------------------------------------------------
        # STEP 2: MULTI-SOURCE RETRIEVAL & LAYER 2 FILTERING
        # -------------------------------------------------------------
        with st.expander("🔹 JEV Decision Layer 2: Document Relevance & Evidence Ranking", expanded=True):
            st.markdown(f"**Retrieved Candidates:** `{out['trace']['retrieved_chunk_count']}`")
            st.markdown(f"**Retained Evidence Chunks:** `{out['trace']['retained_chunk_count']}`")
            st.caption(
                f"⏱️ Vector Search: {out['trace']['vector_search_latency_ms']:.1f} ms | "
                f"Live Retrieval: {out['trace']['live_retrieval_latency_ms']:.1f} ms | "
                f"Relevance Filter: {out['trace']['relevance_filter_latency_ms']:.1f} ms"
            )

            st.markdown("##### Retained Evidence Passages:")
            for idx, c in enumerate(out.get("evidence_chunks", [])):
                score = c.get("similarity_score", 0.0)
                st.markdown(f"**Chunk #{idx+1}** | Score: `{score:.3f}` | Domain: `{c.get('source_domain')}`")
                st.caption(f"Source: {c.get('source_url')}")
                st.code(c.get("english_text") or c.get("original_text", ""), language=None)

        # -------------------------------------------------------------
        # STEP 3: GROQ GENERATION & LAYER 3 QUALITY GATE
        # -------------------------------------------------------------
        with st.expander("🔹 JEV Decision Layer 3: Answer Quality Gate & Verdict", expanded=True):
            verdict = out["quality_decision"]
            badge_color = "#138808" if verdict == "ACCEPT" else ("#ff9933" if verdict == "RETRY" else "#d9534f")
            st.markdown(
                f"### Gate Verdict: <span style='color:{badge_color};'>{verdict}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Decision Rationale:** {out['decision_rationale']}")
            st.markdown(f"**Retries Performed:** `{out['trace']['retry_count']}`")

            q_metrics = out.get("quality_metrics", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faithfulness Score", f"{q_metrics.get('faithfulness_score', 0):.2f}")
            m2.metric("Answer Relevance", f"{q_metrics.get('answer_relevance_score', 0):.2f}")
            m3.metric("Citation Correctness", f"{q_metrics.get('citation_correctness_score', 0):.2f}")
            m4.metric("Unsupported Claims", q_metrics.get("unsupported_claim_count", 0))

            st.caption(
                f"⏱️ Groq Generation: {out['trace']['groq_generation_latency_ms']:.1f} ms | "
                f"Quality Gate: {out['trace']['quality_gate_latency_ms']:.1f} ms | "
                f"Model: {out['trace'].get('groq_model', 'Groq')}"
            )

        # -------------------------------------------------------------
        # STEP 4: FINAL GENERATED ANSWER
        # -------------------------------------------------------------
        st.markdown("### 📝 Final Generated Response")
        st.markdown(out["answer"])

        # Citations Table
        if out.get("citations"):
            st.markdown("#### Citations Verification Map")
            c_data = []
            for c in out["citations"]:
                c_data.append({
                    "Citation": c["citation_id"],
                    "Source Title": c["title"],
                    "Ministry": c["department"],
                    "URL": c["url"],
                    "Similarity": f"{c.get('similarity_score', 0):.3f}",
                })
            st.dataframe(c_data, use_container_width=True)

        # Download full trace
        st.download_button(
            label="📥 Download Detailed Execution Trace (JSON)",
            data=json.dumps(out, indent=2),
            file_name=f"test_lab_trace_{out['request_id']}.json",
            mime="application/json",
        )
