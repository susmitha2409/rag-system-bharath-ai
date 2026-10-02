BharatAI — Agentic RAG Platform for Government of India Websites
![Image](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)
![Image](https://img.shields.io/badge/Hosted%20LLM-Groq%20Cloud-orange.svg)
![Image](https://img.shields.io/badge/Vector%20Store-ChromaDB%20Persistent-brightgreen.svg)
![Image](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
BharatAI is an enterprise-grade, end-to-end Retrieval-Augmented Generation (RAG) platform designed specifically for official Government of India portals. It autonomously crawls, parses, indexes, translates, and retrieves official policy documentation, schemes, and breaking announcements, providing verified citizen answers backed by strict citation evidence.
🏛️ 1. Official Government Registry (10 Portals)
BharatAI includes a centralized, pre-seeded registry of the 10 core central government portals:
PM Office (PMINDIA) — https://www.pmindia.gov.in/ (Prime Minister's Office)
Ministry of Electronics and Information Technology (MeitY) — https://www.meity.gov.in/
Cabinet Secretariat — https://cabsec.gov.in/
Ministry of Health and Family Welfare (MoHFW) — https://mohfw.gov.in/
Ministry of Home Affairs (MHA) — https://www.mha.gov.in/
Ministry of Information and Broadcasting (MIB) — https://mib.gov.in/
Department of Expenditure (DoE) — https://doe.gov.in/ (Ministry of Finance)
Department of Economic Affairs (DEA) — https://dea.gov.in/ (Ministry of Finance)
Press Information Bureau (PIB) — https://www.pib.gov.in/
National Portal of India — https://www.india.gov.in/
🧠 2. Architecture & Three JEV Decision Layers
BharatAI implements a multi-agent reasoning architecture with three JEV (Judgment, Evidence, Verification) decision layers:
code
Code
[Citizen Question]
        │
        ▼
[Reasoning Agent (Groq / Deterministic Fallback)]
        │
        ▼
[JEV Decision Layer 1: Tool & Source Selection Agent]
   ├─ Intent & Topic Extraction
   ├─ Time-Sensitivity Check (Historical vs Latest Breaking News)
   └─ Mode Decision (Mode A: Indexed RAG | Mode B: Live Web Retrieval | Mode C: Hybrid)
        │
        ├─────────────────────────────┬─────────────────────────────┐
        ▼                             ▼                             ▼
[ChromaDB Vector Store]   [Live Official Web Crawler]    [Document Upload Store]
  (Local MiniLM Embeddings) (Direct official domain fetch)   (PDF, DOCX, TXT)
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
             [JEV Decision Layer 2: Document Relevance Filter]
                ├─ Content Deduplication (SHA-256 fingerprinting)
                ├─ Vector Similarity Thresholding
                ├─ Multilingual Script Detection & Groq Translation to English
                └─ Grounded Evidence Context Construction
                                      │
                                      ▼
                      [Answer Generator (Groq Cloud LLM)]
                ├─ Strict Grounded Prompting (No Hallucinations)
                ├─ Mandatory Inline Bracket Citations [1], [2]
                └─ Official Entity & Figure Preservation
                                      │
                                      ▼
             [JEV Decision Layer 3: Answer Quality Gate]
                ├─ Groundedness Verification (Faithfulness score)
                ├─ Citation Legitimacy Map (Checks for phantom citations)
                ├─ Unsupported Claim Detection
                └─ Decision Verdict:
                     ├─ ACCEPT: Present verified answer with clickable source citations
                     ├─ RETRY: Reformulate query and re-execute (max 2 retries)
                     └─ ABSTAIN: Declare insufficient official evidence
⚡ 3. Mandatory Groq API Configuration
Groq is the sole hosted LLM provider for BharatAI. No calls are made to Gemini, OpenAI, Anthropic, or any other hosted provider.
Endpoint: https://api.groq.com/openai/v1/chat/completions
Default Model: llama-3.3-70b-versatile (fully configurable to llama3-70b-8192, llama-3.1-8b-instant, etc.)
Error Handling: Bounded exponential backoff with HTTP 429 Retry-After adherence.
Security: Secret redaction engine prevents API keys from appearing in UI logs, traces, or exported JSON/CSV files.
💻 4. Installation & Local Setup on Ubuntu
Prerequisites
Ubuntu 20.04 LTS or 22.04 LTS
Python 3.10 or 3.11
Git
Step 1: Clone or Navigate to Project
code
Bash
cd bharatai
Step 2: Create and Activate Virtual Environment
code
Bash
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
code
Bash
pip install --upgrade pip
pip install -r requirements.txt
Step 4: Configure Environment Variables
Copy the .env.example file and configure your Groq API credentials:
code
Bash
cp .env.example .env
Edit .env and set:
code
Ini
GROQ_API_KEY=gsk_your_actual_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
(Alternatively, you can launch the app and input your key in the Settings page in the UI!)
Step 5: Launch the Streamlit Application
code
Bash
streamlit run app.py
Open your browser at http://localhost:8501.
🚀 5. Application Walkthrough & Features
The Streamlit UI provides 6 dedicated modules accessible via the sidebar:
1. 💬 Chat Assistant
Natural language questions answered with official citations [1], [2].
Retrieval modes: Indexed, Live, or Hybrid.
Response language selector: English (default), Hindi, Tamil, Telugu, Bengali, Marathi.
Expandable source cards displaying official URL, issuing ministry, publication date, and original text / machine-translation badges.
Thumbs up/down feedback collection persisted in SQLite.
Instant JSON export of the full answer and citations.
2. 🌐 Website Management
Live telemetry for all 10 registered official portals.
One-click crawling for individual portals or batch crawling across all enabled portals.
Crawl progress bar, discovered/indexed page counters, and error logs.
Form to safely register additional .gov.in and .nic.in departmental portals.
3. 📚 Knowledge Base Explorer
Direct semantic similarity search and keyword lookup across indexed records.
Document uploader: Ingest official Gazettes or policy circulars in PDF, DOCX, TXT, or MD formats.
Real-time chunk inspector displaying cosine similarity scores.
4. 📊 Evaluation & Observability Dashboard
Latency Breakdown: Precise microsecond timings using time.perf_counter() for Source Selection, Embeddings, Vector Search, Live Fetch, Relevance Filter, Groq API, and Quality Gate.
Quality Metrics: Faithfulness, Answer Relevance, Precision@5, Recall@5, MRR, and NDCG.
Interactive Visualizations: Plotly charts showing stage-by-stage latency distributions, gate decision ratios, and latency trends.
Automated Benchmark Runner: One-click batch testing over 20 standardized questions.
Data Export: Clean CSV and JSON exports with automated secret redaction.
5. 🔬 Test Lab
Granular step-by-step diagnostic workspace.
Step through JEV Layer 1, Layer 2, and Layer 3 individually.
Inspect raw model prompts, vector distance metrics, and gate justification strings.
6. ⚙️ Settings
Live Groq API connection test button with latency diagnostic.
Runtime model switching (llama-3.3-70b-versatile, llama-3.1-8b-instant, etc.).
Sliders for temperature, max tokens, chunk size, chunk overlap, and crawler rate limits.
🧪 6. Running Automated Tests
Run the complete test suite using pytest:
code
Bash
pytest tests/ -v
The test suite covers:
test_llm_client.py: Groq client configuration, authentication handling, and secret redaction.
test_crawler.py: Approved domain allowlisting and SSRF protections.
test_document_processor.py: Text chunking, sentence boundary detection, and Indian language script detection.
test_retriever.py: Persistent vector database insertions, cosine search, and fallback store.
test_rag_pipeline.py: The 3 JEV Decision Layers and citation validation.
test_evaluation.py: Precision@K, Recall@K, MRR, NDCG calculation engines.
🔒 7. Security & Governance
Domain Allowlisting: The crawler strictly permits access to approved .gov.in and .nic.in domains.
SSRF Defenses: Private IP ranges (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.169.254) and metadata endpoints are blocked.
Prompt Injection Shielding: Retrieved web content is sanitized to prevent prompt injections from escaping evidence context.
Secret Sanitization: All logs, traces, and exports automatically mask API keys.
robots.txt Adherence: Crawling strictly honors robot parser rules and crawl delay settings.
📜 8. License & Disclaimer
BharatAI is developed for educational, analytical, and governance intelligence purposes. Retrieved documents and citations remain the official crown copyright of their respective ministries of the Government of India.
