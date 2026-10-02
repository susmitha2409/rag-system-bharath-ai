"""
BharatAI — Knowledge Base Explorer (Page 3)
Explore vector store index, search indexed government records, upload official documents (PDF/DOCX),
and inspect chunk-level metadata.
"""

import pandas as pd
import streamlit as st
from core.vector_store import get_vector_store
from core.document_processor import get_document_processor
from core.security import validate_file_upload
from database.db import get_db


def render_knowledge_page():
    st.markdown("## 📚 Knowledge Base & Document Explorer")
    st.caption("Search, inspect, and manage indexed official Government of India documents.")

    db = get_db()
    vector_store = get_vector_store()
    doc_processor = get_document_processor()

    # Metrics Summary
    total_docs = len(db.get_all_documents(limit=500))
    total_chunks = vector_store.count()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Indexed Documents", total_docs)
    m_col2.metric("Vector Database Chunks", total_chunks)
    m_col3.metric("Vector Store Type", "ChromaDB Persistent")

    st.markdown("---")

    # Document Upload Section
    with st.expander("📤 Upload Official Government Document (PDF / DOCX / TXT)", expanded=False):
        uploaded_file = st.file_uploader("Upload Gazette, Policy Document, or Press Release", type=["pdf", "docx", "txt", "md"])
        up_dept = st.selectbox(
            "Issuing Ministry / Department",
            options=[
                "Prime Minister's Office",
                "Ministry of Electronics & IT",
                "Cabinet Secretariat",
                "Ministry of Health & Family Welfare",
                "Ministry of Home Affairs",
                "Ministry of Information & Broadcasting",
                "Ministry of Finance - Department of Expenditure",
                "Ministry of Finance - Department of Economic Affairs",
                "Press Information Bureau",
                "National Portal of India",
            ],
        )

        if uploaded_file and st.button("📥 Ingest and Index Document", type="primary"):
            file_bytes = uploaded_file.read()
            valid, msg = validate_file_upload(uploaded_file.name, len(file_bytes))
            if not valid:
                st.error(msg)
            else:
                filename = uploaded_file.name
                ext = filename.rsplit(".", 1)[-1].lower()

                with st.spinner("Processing text and generating embeddings..."):
                    if ext == "pdf":
                        doc_rec = doc_processor.process_pdf(
                            pdf_bytes=file_bytes,
                            url=f"upload://{filename}",
                            department=up_dept,
                            domain="uploaded.gov.in",
                            title=filename,
                        )
                    elif ext == "docx":
                        doc_rec = doc_processor.process_docx(
                            docx_bytes=file_bytes,
                            url=f"upload://{filename}",
                            department=up_dept,
                            domain="uploaded.gov.in",
                            title=filename,
                        )
                    else:
                        doc_rec = doc_processor.process_plain_text(
                            text=file_bytes.decode("utf-8", errors="ignore"),
                            url=f"upload://{filename}",
                            department=up_dept,
                            domain="uploaded.gov.in",
                            title=filename,
                            doc_type=ext,
                        )

                    # Save to DB and Vector Store
                    db.record_document(
                        doc_id=doc_rec["id"],
                        page_id=None,
                        website_id=None,
                        title=doc_rec["title"],
                        url=doc_rec["url"],
                        domain=doc_rec["domain"],
                        department=doc_rec["department"],
                        publication_date=None,
                        doc_type=doc_rec["document_type"],
                        source_language=doc_rec["source_language"],
                        original_text=doc_rec["original_text"][:5000],
                        english_text=doc_rec["english_text"][:5000],
                        is_translated=doc_rec["is_translated"],
                        content_hash=doc_rec["content_hash"],
                        num_chunks=doc_rec["num_chunks"],
                    )
                    vector_store.add_chunks(doc_rec["chunks"])

                st.success(f"Successfully processed and indexed '{filename}' into {doc_rec['num_chunks']} vector chunks!")
                st.rerun()

    # Search & Inspection
    st.markdown("### 🔎 Semantic & Keyword Document Search")
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        search_query = st.text_input("Enter search query or topic keywords...", placeholder="e.g. Semiconductor capital subsidy or PM-KISAN guidelines")
    with s_col2:
        top_k_search = st.number_input("Max Results", min_value=1, max_value=20, value=5)

    if search_query:
        with st.spinner("Searching vector index..."):
            results = vector_store.similarity_search(query=search_query, top_k=top_k_search)

        if not results:
            st.info("No matching chunks found in the index for this query.")
        else:
            st.markdown(f"Found **{len(results)}** matching chunks:")
            for i, r in enumerate(results):
                with st.expander(f"Chunk #{i+1} — {r.get('source_title', 'Document')} (Score: {r.get('similarity_score', 0):.3f})"):
                    st.markdown(f"**URL:** [{r.get('source_url')}]({r.get('source_url')})")
                    st.markdown(f"**Department:** `{r.get('department')}` | **Language:** `{r.get('source_language')}`")
                    st.code(r.get("english_text") or r.get("original_text", ""), language=None)

    # Document Table List
    st.markdown("### 📑 Recent Indexed Documents")
    all_docs = db.get_all_documents(limit=50)
    if all_docs:
        docs_df = pd.DataFrame(all_docs)
        st.dataframe(docs_df[["title", "department", "document_type", "num_chunks", "created_at", "url"]], use_container_width=True)
    else:
        st.info("No documents indexed yet. Use the Website Crawler or upload a document to build your knowledge base.")
