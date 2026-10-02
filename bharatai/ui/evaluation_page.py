"""
BharatAI — Evaluation Dashboard (Page 4)
Comprehensive analytics, stage-by-stage latency breakdowns, retrieval & answer quality metrics,
interactive Plotly visualizations, and automated benchmark execution.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import GROQ_MODEL, GROQ_API_URL
from core.evaluation import EvaluationRunner
from database.db import get_db


def render_evaluation_page():
    st.markdown("## 📊 Evaluation & Observability Dashboard")
    st.caption("Measure stage-by-stage latencies, model performance, retrieval accuracy, and groundedness.")

    db = get_db()
    summary = db.get_analytics_summary()
    logs = db.get_query_logs(limit=100)

    # Top KPI Metrics Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Inquiries", summary["total_queries"])
    k2.metric("Avg Latency", f"{summary['avg_total_latency_ms']:.0f} ms")
    k3.metric("P95 Latency", f"{summary['p95_total_latency_ms']:.0f} ms")
    k4.metric("Avg Faithfulness", f"{summary['avg_faithfulness'] * 100:.1f}%")
    k5.metric("Accept Rate", f"{(summary['accept_count'] / max(summary['total_queries'], 1)) * 100:.1f}%")

    st.markdown("---")

    # Automated Benchmark Execution Section
    with st.expander("⚡ Run Automated Benchmark Evaluation (20 Standard Questions)", expanded=False):
        b_col1, b_col2 = st.columns([3, 1])
        with b_col1:
            st.write(
                "Run end-to-end evaluation across standardized questions covering all 10 official ministries. "
                "Calculates Precision@5, Recall@5, MRR, groundedness, and response times."
            )
            num_q = st.slider("Number of benchmark questions to evaluate", min_value=1, max_value=20, value=5)
        with b_col2:
            st.write("")
            run_bench_btn = st.button("▶️ Execute Benchmark", type="primary", use_container_width=True)

        if run_bench_btn:
            eval_runner = EvaluationRunner()
            p_bar = st.progress(0.0, text="Initializing benchmark...")

            def bench_cb(msg, cur, tot):
                p_bar.progress(cur / max(tot, 1), text=msg)

            bench_result = eval_runner.run_benchmark(num_questions=num_q, progress_callback=bench_cb)
            p_bar.progress(1.0, text="Benchmark completed!")
            st.success(
                f"Benchmark Complete! Avg Precision@5: {bench_result['avg_precision_at_5']:.2f} | "
                f"Avg MRR: {bench_result['avg_mrr']:.2f} | Avg Faithfulness: {bench_result['avg_faithfulness']:.2f}"
            )
            if bench_result.get("detailed_results"):
                st.dataframe(pd.DataFrame(bench_result["detailed_results"]), use_container_width=True)

    # Charts Section
    if logs:
        df_logs = pd.DataFrame(logs)

        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            st.markdown("#### ⏱️ Latency Breakdown by Processing Stage")
            # Calculate average latency across stages
            avg_stages = {
                "Source Selection": df_logs["source_selection_latency_ms"].mean(),
                "Embedding": df_logs["embedding_latency_ms"].mean(),
                "Vector Search": df_logs["vector_search_latency_ms"].mean(),
                "Live Retrieval": df_logs["live_retrieval_latency_ms"].mean(),
                "Relevance Filter": df_logs["relevance_filter_latency_ms"].mean(),
                "Groq Generation": df_logs["groq_generation_latency_ms"].mean(),
                "Quality Gate": df_logs["quality_gate_latency_ms"].mean(),
            }
            fig_lat = px.bar(
                x=list(avg_stages.keys()),
                y=list(avg_stages.values()),
                labels={"x": "Stage", "y": "Avg Latency (ms)"},
                title="Average Latency per Component (ms)",
                color=list(avg_stages.values()),
                color_continuous_scale="Viridis",
            )
            fig_lat.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_lat, use_container_width=True)

        with ch_col2:
            st.markdown("#### 🛡️ JEV Decision Layer 3 Gate Distribution")
            decision_counts = df_logs["final_decision"].value_counts().reset_index()
            decision_counts.columns = ["Decision", "Count"]
            colors = {"ACCEPT": "#138808", "RETRY": "#ff9933", "ABSTAIN": "#d9534f"}
            fig_dec = px.pie(
                decision_counts,
                values="Count",
                names="Decision",
                title="Quality Gate Decisions",
                color="Decision",
                color_discrete_map=colors,
            )
            fig_dec.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dec, use_container_width=True)

        # Latency over time line chart
        st.markdown("#### 📈 End-to-End Latency Trend Over Recent Queries")
        fig_trend = px.line(
            df_logs,
            x="timestamp",
            y=["total_latency_ms", "groq_generation_latency_ms"],
            labels={"value": "Latency (ms)", "variable": "Metric", "timestamp": "Timestamp"},
            title="Total vs Groq LLM Latency (ms)",
        )
        fig_trend.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    # Groq Model Performance Panel
    st.markdown("### 🤖 Groq LLM Host Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LLM Provider", "Groq (Hosted)")
    m2.metric("Active Model", GROQ_MODEL)
    m3.metric("API Endpoint", "api.groq.com/v1")
    m4.metric("Success Rate", f"{(summary['successful_requests'] / max(summary['total_queries'], 1)) * 100:.1f}%")

    # Detailed Logs Table
    st.markdown("### 🔍 Detailed Query Traces & Latency Logs")
    if logs:
        st.dataframe(
            df_logs[
                [
                    "request_id",
                    "timestamp",
                    "user_query",
                    "retrieval_mode",
                    "final_decision",
                    "total_latency_ms",
                    "groq_generation_latency_ms",
                    "retrieved_chunk_count",
                    "retained_chunk_count",
                    "api_status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Export buttons
        exp_col1, exp_col2, _ = st.columns([2, 2, 4])
        with exp_col1:
            csv_data = df_logs.to_csv(index=False)
            st.download_button(
                label="📥 Export Traces (CSV)",
                data=csv_data,
                file_name="bharatai_query_traces.csv",
                mime="text/csv",
            )
        with exp_col2:
            json_data = json.dumps(logs, indent=2)
            st.download_button(
                label="📥 Export Traces (JSON)",
                data=json_data,
                file_name="bharatai_query_traces.json",
                mime="application/json",
            )
    else:
        st.info("No query logs recorded yet. Ask a question on the Chat Assistant page to view live analytics.")
