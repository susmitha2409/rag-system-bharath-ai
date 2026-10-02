# BharatAI — Government Website Intelligence Platform

**Agentic Retrieval-Augmented Generation (RAG) for official Government of India websites**

BharatAI is a Streamlit-based application designed to retrieve information from selected Government of India websites and generate evidence-grounded answers using **Groq**. It combines website crawling, document processing, local embeddings, persistent vector search, source citations, and evaluation tools in one interface.

> **Important:** BharatAI is an independent software project and is not an official Government of India service. Answers are AI-generated and should be checked against the linked official sources, especially for legal, financial, medical, or time-sensitive matters.

---

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Supported government sources](#supported-government-sources)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Run BharatAI](#run-bharatai)
- [Using the application](#using-the-application)
- [Evaluation and observability](#evaluation-and-observability)
- [Project structure](#project-structure)
- [Running tests](#running-tests)
- [Security and responsible use](#security-and-responsible-use)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License and attribution](#license-and-attribution)

---

## Highlights

- **Government-source RAG:** Retrieve content from a configurable registry of official government portals.
- **Groq-powered generation:** Uses a Groq Chat Completions API model for LLM-based operations.
- **Multiple retrieval modes:** Indexed retrieval, live official-site retrieval, and hybrid retrieval.
- **Three JEV decision layers:** Source selection, document relevance filtering, and answer-quality/retrieval gating.
- **Evidence and citations:** Answers can include source references linked to retrieved documents.
- **English-first workflow:** Prioritizes English content where available and supports processing of other languages.
- **Persistent knowledge base:** Uses ChromaDB for vector search and SQLite for application records.
- **Document ingestion:** Supports uploading reference documents such as PDF, DOCX, TXT, and Markdown, subject to the formats enabled in the application.
- **Evaluation dashboard:** Inspect retrieval and answer metrics, request traces, and measured latency.
- **Test Lab:** Examine retrieval results, intermediate decisions, and answer-generation behavior.
- **Operational visibility:** Review crawl status, indexed content, errors, and model connectivity.

---

## Architecture

```text
                         ┌─────────────────────────┐
                         │   Streamlit Interface   │
                         │ Chat • Admin • Evaluate │
                         └────────────┬────────────┘
                                      │ User question
                                      ▼
                         ┌─────────────────────────┐
                         │ Reasoning / Query Plan  │
                         │       Groq LLM          │
                         └────────────┬────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────┐
                   │ JEV Layer 1: Source Selection    │
                   │ Indexed • Live • Hybrid • Files  │
                   └────────────────┬─────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
    ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
    │ ChromaDB Vector │   │ Official Website │   │ Uploaded Files  │
    │ Store + Local   │   │ Crawler / Fetch  │   │ & Document Text │
    │ Embeddings      │   │                  │   │                 │
    └────────┬────────┘   └─────────┬────────┘   └────────┬────────┘
             └──────────────────────┼──────────────────────┘
                                    ▼
                   ┌──────────────────────────────────┐
                   │ JEV Layer 2: Relevance Filtering │
                   │ Ranking • Deduplication • Context│
                   └────────────────┬─────────────────┘
                                    ▼
                         ┌─────────────────────────┐
                         │ Answer Generation       │
                         │       Groq LLM          │
                         └────────────┬────────────┘
                                      ▼
                   ┌──────────────────────────────────┐
                   │ JEV Layer 3: Answer Quality Gate │
                   │ ACCEPT • RETRY • ABSTAIN         │
                   └────────────────┬─────────────────┘
                                    ▼
                         ┌─────────────────────────┐
                         │ Answer + Source Links   │
                         │ Traces + Evaluation     │
                         └─────────────────────────┘
```

### JEV decision layers

1. **Layer 1 — Tool and source selection:** Determines which configured knowledge sources and retrieval mode are appropriate for a question.
2. **Layer 2 — Document relevance filter:** Deduplicates and filters candidate passages, using retrieval scores and configured relevance logic.
3. **Layer 3 — Answer quality and retrieval gate:** Checks evidence and citations, then returns an `ACCEPT`, `RETRY`, or `ABSTAIN` decision according to the implemented quality-gate logic.

The application uses local Python components for crawling, document processing, embeddings, and vector search. Groq is used for hosted LLM inference; it is not the crawler or vector database.

---

## Supported government sources

The initial registry contains these ten portals:

| # | Source | Website |
|---:|---|---|
| 1 | Prime Minister's Office | [pmindia.gov.in](https://www.pmindia.gov.in/) |
| 2 | Ministry of Electronics and Information Technology (MeitY) | [meity.gov.in](https://www.meity.gov.in/) |
| 3 | Cabinet Secretariat | [cabsec.gov.in](https://cabsec.gov.in/) |
| 4 | Ministry of Health and Family Welfare | [mohfw.gov.in](https://mohfw.gov.in/) |
| 5 | Ministry of Home Affairs | [mha.gov.in](https://www.mha.gov.in/) |
| 6 | Ministry of Information and Broadcasting | [mib.gov.in](https://mib.gov.in/) |
| 7 | Department of Expenditure | [doe.gov.in](https://doe.gov.in/) |
| 8 | Department of Economic Affairs | [dea.gov.in](https://dea.gov.in/) |
| 9 | Press Information Bureau (PIB) | [pib.gov.in](https://www.pib.gov.in/) |
| 10 | National Portal of India | [india.gov.in](https://www.india.gov.in/) |

Website availability, crawl permissions, page structure, and language options may change. The number of pages successfully indexed depends on site accessibility and crawler settings.

---

## Technology stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Hosted LLM | Groq API |
| Vector database | ChromaDB |
| Embeddings | Sentence Transformers, running locally |
| Application metadata and traces | SQLite |
| Web extraction | Requests, BeautifulSoup, Trafilatura |
| PDF processing | PyMuPDF |
| DOCX processing | python-docx |
| Data handling | Pandas |
| Visualizations | Plotly |
| Tests | pytest |

---

## Requirements

- Python **3.10 or 3.11** (recommended; use a version compatible with the pinned dependencies in your environment)
- `pip` and `venv`
- A Groq API key
- Internet access for Groq requests, website crawling, and initial embedding-model downloads
- Sufficient disk space for the embedding model, ChromaDB index, SQLite database, and downloaded content

The first run may take longer while the local embedding model is downloaded and initialized.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

If the Python application is inside a `bharatai/` directory in your repository, enter it before continuing:

```bash
cd bharatai
```

Run the following commands from the directory containing `app.py`, `config.py`, and `requirements.txt`.

### 2. Create a virtual environment

**Ubuntu / Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

### 1. Create your environment file

Copy the example configuration:

```bash
cp .env.example .env
```

Open `.env` and set your own Groq API key and a model ID currently available to your Groq account:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
GROQ_MODEL=your_supported_groq_model
```

Get or manage your key from the [Groq Console](https://console.groq.com/keys), and check available models in the [Groq documentation](https://console.groq.com/docs/models).

The project’s `.env.example` may include additional options for the model, retrieval, crawler, chunking, and storage. Review those values before running the application.

**Never commit `.env` or publish your API key.** Keep the real key in your local environment or your deployment platform’s secret manager.

### 2. Configuration notes

- `GROQ_API_KEY`: Secret used to authenticate with Groq.
- `GROQ_API_URL`: Groq Chat Completions endpoint.
- `GROQ_MODEL`: Model ID supported by Groq and enabled for your account.
- `EMBEDDING_MODEL_NAME`: Local Sentence Transformers model used for embeddings.
- `CHROMA_PERSIST_DIRECTORY`: Persistent ChromaDB storage location.
- `DATABASE_PATH`: SQLite database location.
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Document chunking settings.
- `RETRIEVAL_TOP_K`: Number of candidate chunks to retrieve.
- Crawler settings: Control crawl limits, depth, request delay, and timeout.
- Quality-gate settings: Control the maximum number of retrieval retries.

Use the variable names in the checked-in `.env.example` as the source of truth for your current project version.

---

## Run BharatAI

From the application directory:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

If the app does not start, check the terminal output and the [Troubleshooting](#troubleshooting) section.

---

## Using the application

### 1. Configure and test Groq

Open **Settings**, confirm the configured model, and use the connection-test control if available. A successful connection confirms that the application can reach Groq with the configured credentials; it does not by itself validate retrieval quality.

### 2. Crawl official websites

Open **Website Management**:

1. Choose one website to test first.
2. Start a crawl.
3. Review discovered, indexed, and failed pages.
4. Check crawl logs for blocked pages or extraction errors.
5. Expand to additional sites after confirming the initial crawl works.

Respect each website's terms, `robots.txt`, and access restrictions. Crawling is subject to site availability and configured limits.

### 3. Ask a question

Open **Chat Assistant** and ask a question about the indexed sources. Where available, select a retrieval mode:

- **Indexed:** Search the local knowledge base.
- **Live:** Retrieve relevant content from accessible official pages.
- **Hybrid:** Combine indexed content and live retrieval.

For questions about the latest announcements, use live or hybrid retrieval when available. Always check the publication date shown with the source.

### 4. Inspect the evidence

Review the answer's citations and open the linked official pages. Expand source passages and metadata to verify that the retrieved evidence supports the answer.

### 5. Upload documents

Use the Knowledge Base interface to add supported files such as PDF, DOCX, TXT, or Markdown. Uploaded documents supplement the configured website sources; they should not be mistaken for official government sources unless their provenance has been verified.

### 6. Inspect pipeline decisions

Use **Test Lab** to inspect retrieved chunks, relevance scores, JEV decisions, citations, and latency details available for a query.

---

## Evaluation and observability

The Evaluation Dashboard is intended to help inspect system behavior and compare runs.

Depending on available labels and recorded traces, it can report:

- End-to-end and per-stage latency
- Groq request latency and API failures
- Retrieved and retained chunk counts
- Source and citation information
- Retry, accept, and abstain decisions
- Answer-quality estimates
- Retrieval metrics such as Precision@K, Recall@K, MRR, and NDCG when relevance labels or ground truth are available
- Batch evaluation results and downloadable reports

### Interpret metrics carefully

- **Latency** should be calculated from actual measurements; it varies with network conditions, model load, crawl behavior, and hardware.
- **LLM-as-a-judge scores** are automated estimates, not ground truth.
- **Precision, recall, MRR, and NDCG** require relevance labels or a suitable reference dataset. They should not be interpreted as meaningful benchmark results without one.
- **A successful API response is not the same as a correct answer.**
- Evaluate answers against the linked source material, particularly for time-sensitive or consequential questions.

---

## Project structure

```text
bharatai/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── core/
│   ├── llm_client.py
│   ├── embeddings.py
│   ├── crawler.py
│   ├── language_processor.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── source_selection.py
│   ├── relevance_filter.py
│   ├── answer_generator.py
│   ├── quality_gate.py
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   ├── tracing.py
│   └── security.py
├── database/
│   ├── db.py
│   └── schema.sql
├── ui/
│   ├── chat_page.py
│   ├── website_page.py
│   ├── knowledge_page.py
│   ├── evaluation_page.py
│   ├── test_lab_page.py
│   └── settings_page.py
├── data/
└── tests/
    ├── test_llm_client.py
    ├── test_crawler.py
    ├── test_document_processor.py
    ├── test_retriever.py
    ├── test_rag_pipeline.py
    └── test_evaluation.py
```

Some generated files or modules may differ slightly by project version. Refer to the actual repository tree if it has changed.

---

## Running tests

From the application directory:

```bash
pytest tests/ -v
```

If `pytest` is not installed in the active environment:

```bash
python -m pip install pytest
python -m pytest tests/ -v
```

Tests that contact external websites or Groq may require network access and valid configuration. Prefer mocked API and HTTP responses for repeatable unit tests. Do not describe tests as passing unless they have been run in the current environment.

---

## Security and responsible use

- Keep API keys in environment variables or a secrets manager.
- Ensure `.env` is ignored by Git.
- Do not commit database files, downloaded documents, logs, or vector-index data unless you intentionally want to publish them.
- Restrict crawling to approved domains and respect website policies.
- Validate URLs and uploaded files.
- Treat retrieved website content as untrusted input; content may contain misleading information or prompt-injection attempts.
- Do not expose the app publicly without suitable authentication, rate limiting, and resource controls.
- Use HTTPS for public deployment.
- The configured Groq endpoint uses HTTPS. Protect credentials and review your organization's data-handling requirements before sending sensitive content to any hosted model.
- Review generated answers and citations before relying on them.

---

## Troubleshooting

### Groq authentication or model error

- Confirm that `GROQ_API_KEY` is set correctly.
- Verify that the key is active.
- Confirm that `GROQ_MODEL` is a model ID currently available to your account.
- Check Groq service status, rate limits, and error messages.
- Restart Streamlit after changing environment variables.

### No results from retrieval

- Confirm that at least one website crawl completed successfully.
- Check the indexed-page and chunk counts.
- Try a broader question or increase the retrieval top-k.
- Verify that the selected website is enabled.
- Check whether the source page is accessible and contains extractable text.

### A website fails to crawl

- Review the crawl error and HTTP status.
- Check the site's `robots.txt`, redirects, and access restrictions.
- Try a smaller crawl limit or a single-page test.
- Some pages may require JavaScript rendering or may block automated requests; do not bypass access controls.

### Embedding model download or memory issues

- Ensure the machine has internet access for the initial model download.
- Confirm that enough disk space and RAM are available.
- Use a compatible Python and PyTorch installation.
- Allow the first initialization to complete before retrying.

### Data does not persist

- Confirm that `CHROMA_PERSIST_DIRECTORY` and `DATABASE_PATH` point to writable, persistent locations.
- For cloud deployment, use persistent storage. Ephemeral container filesystems may be cleared when an instance restarts or is redeployed.

---

## Limitations

- Results depend on what the crawler can access and extract from each website.
- Government websites may change their page layouts, URLs, language selectors, or access policies.
- Live retrieval is limited to pages the application can discover and fetch; it is not a guarantee of exhaustive web search.
- Publication dates may be missing or inconsistently represented on source pages.
- Machine translation can introduce errors. Check the original-language source for important details.
- The model can still produce mistakes. Citations and quality checks reduce risk but do not guarantee correctness.
- Evaluation metrics depend on the quality and coverage of the reference dataset.
- Production deployment requires operational decisions about authentication, persistence, monitoring, backups, and resource limits.

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Keep secrets and local data out of commits.
4. Add or update tests for changes.
5. Run the test suite.
6. Submit a pull request describing the change and any relevant limitations.

For changes to crawling behavior, preserve domain restrictions and respectful request rates.

---

## License and attribution

Add the license that applies to your repository before publishing it. Unless a license file is included, reuse and redistribution permissions are not explicitly granted by this README.

BharatAI is an independent project. Government websites and their content remain the responsibility of their respective publishers. Use official source pages for authoritative information and attribution.
