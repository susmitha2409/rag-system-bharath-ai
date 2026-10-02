#!/usr/bin/env bash
# BharatAI Launcher Script
cd "$(dirname "$0")/bharatai" || exit 1
echo "Starting BharatAI — Government Website Intelligence Platform..."
streamlit run app.py
