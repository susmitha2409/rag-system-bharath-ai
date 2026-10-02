"""
BharatAI — Official Government Website Management (Page 2)
Monitor registry status, initiate crawl jobs, inspect crawl logs, and add official portals.
"""

import time
import pandas as pd
import streamlit as st
from config import OFFICIAL_WEBSITES
from core.crawler import GovernmentCrawler
from database.db import get_db


def render_website_page():
    st.markdown("## 🌐 Official Government Website Registry & Crawler")
    st.caption("Manage indexing and discovery for the 10 official Government of India portals.")

    db = get_db()
    crawler = GovernmentCrawler()

    # Fetch current website registry from SQLite
    websites = db.get_websites()

    # Action Toolbar
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        selected_site_id = st.selectbox(
            "Select Website",
            options=[w["id"] for w in websites],
            format_func=lambda x: next((w["name"] for w in websites if w["id"] == x), x),
        )

    with col2:
        st.write("") # spacing
        st.write("")
        crawl_btn = st.button("🚀 Crawl Selected Website", type="primary", use_container_width=True)

    with col3:
        st.write("")
        st.write("")
        crawl_all_btn = st.button("⚡ Crawl All Enabled Websites", use_container_width=True)

    # Progress placeholder
    progress_placeholder = st.empty()

    if crawl_btn and selected_site_id:
        def update_progress(msg, current, total):
            pct = min(1.0, current / max(total, 1))
            progress_placeholder.progress(pct, text=msg)

        with st.spinner(f"Crawling website..."):
            result = crawler.crawl_website(selected_site_id, progress_callback=update_progress)
            progress_placeholder.empty()
            st.success(
                f"Crawl {result['status']}! Indexed {result['pages_indexed']} pages and {result['indexed_chunks']} chunks in {result['duration_seconds']}s."
            )
            st.rerun()

    if crawl_all_btn:
        progress_bar = st.progress(0.0, text="Starting batch crawl...")
        enabled_sites = [w for w in websites if w.get("enabled")]
        for i, s in enumerate(enabled_sites):
            progress_bar.progress((i) / len(enabled_sites), text=f"Crawling {s['name']}...")
            crawler.crawl_website(s["id"])
        progress_bar.progress(1.0, text="Batch crawl completed!")
        st.success(f"Crawled all {len(enabled_sites)} enabled government portals!")
        st.rerun()

    # Table of Websites
    st.markdown("### 📋 Configured Portals Registry")
    table_data = []
    for w in websites:
        table_data.append({
            "ID": w["id"],
            "Website Name": w["name"],
            "Domain": w["canonical_domain"],
            "Department": w["department"],
            "Status": "🟢 Enabled" if w["enabled"] else "🔴 Disabled",
            "Last Crawled": (w["last_crawl_timestamp"] or "Never")[:19],
            "Crawl Status": w["last_crawl_status"],
            "Discovered": w["pages_discovered"],
            "Indexed Pages": w["pages_indexed"],
            "Indexed Chunks": w["indexed_chunks"],
            "Languages": w.get("supported_languages", "['en']"),
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Toggle Enable/Disable
    st.markdown("---")
    t_col1, t_col2 = st.columns([3, 2])
    with t_col1:
        st.markdown("#### ⚙️ Toggle Website Activation")
        target_site = st.selectbox("Choose Portal to Toggle", options=[w["id"] for w in websites], format_func=lambda x: next((w["name"] for w in websites if w["id"] == x), x), key="toggle_sel")
        site_obj = next((w for w in websites if w["id"] == target_site), None)
        if site_obj:
            curr_state = bool(site_obj["enabled"])
            new_state = st.toggle("Portal Enabled for Retrieval", value=curr_state, key="site_toggle_btn")
            if new_state != curr_state:
                db.update_website_status(target_site, new_state)
                st.toast(f"Updated {site_obj['name']} status to {'Enabled' if new_state else 'Disabled'}")
                st.rerun()

    with t_col2:
        st.markdown("#### ➕ Add Official Portal")
        with st.expander("Register New Government Website", expanded=False):
            with st.form("add_site_form"):
                new_name = st.text_input("Portal Name", placeholder="e.g. Ministry of Finance")
                new_url = st.text_input("Base URL", placeholder="https://finmin.gov.in/")
                new_domain = st.text_input("Canonical Domain", placeholder="finmin.gov.in")
                new_dept = st.text_input("Department Name", placeholder="Department of Financial Services")
                submit_add = st.form_submit_button("Register Portal")
                if submit_add:
                    if new_domain.endswith(".gov.in") or new_domain.endswith(".nic.in"):
                        added = db.add_custom_website(new_name, new_url, new_domain, new_dept)
                        if added:
                            st.success("Official website registered successfully!")
                            st.rerun()
                        else:
                            st.error("Domain already exists in registry.")
                    else:
                        st.error("Only official .gov.in or .nic.in domains can be registered.")

    # Crawl Errors Section
    with st.expander("⚠️ Recent Crawl Errors & Logs", expanded=False):
        errors = db.get_recent_crawl_errors(limit=20)
        if errors:
            err_df = pd.DataFrame(errors)
            st.dataframe(err_df[["timestamp", "url", "error_type", "status_code", "error_message"]], use_container_width=True)
        else:
            st.info("No crawler errors recorded. Crawler is operating cleanly.")
