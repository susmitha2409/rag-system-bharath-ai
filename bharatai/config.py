"""
BharatAI — Configuration Management
Handles environment variables, default settings, and official government website registry.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k.strip() not in os.environ:
                        os.environ[_k.strip()] = _v.strip().strip("'\"")

# --------------------------------------------------------------------------
# GROQ API SETTINGS (MANDATORY & SOLE HOSTED LLM PROVIDER)
# --------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "45"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2048"))

# --------------------------------------------------------------------------
# EMBEDDINGS & VECTOR DATABASE
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma_db"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))

# --------------------------------------------------------------------------
# CRAWLER CONFIGURATION
# --------------------------------------------------------------------------
CRAWLER_MAX_PAGES_PER_SITE = int(os.getenv("CRAWLER_MAX_PAGES_PER_SITE", "25"))
CRAWLER_MAX_DEPTH = int(os.getenv("CRAWLER_MAX_DEPTH", "2"))
CRAWLER_REQUEST_DELAY = float(os.getenv("CRAWLER_REQUEST_DELAY", "1.0"))
CRAWLER_TIMEOUT_SECONDS = int(os.getenv("CRAWLER_TIMEOUT_SECONDS", "15"))
CRAWLER_USER_AGENT = os.getenv(
    "CRAWLER_USER_AGENT",
    "BharatAI-Government-Information-Bot/1.0 (+https://github.com/gov-bharatai)"
)

# --------------------------------------------------------------------------
# AGENT & PIPELINE CONFIGURATION
# --------------------------------------------------------------------------
QUALITY_GATE_MAX_RETRIES = int(os.getenv("QUALITY_GATE_MAX_RETRIES", "2"))
DEFAULT_RESPONSE_LANGUAGE = os.getenv("DEFAULT_RESPONSE_LANGUAGE", "en")
DATABASE_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "bharatai.db"))

# Ensure data directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# --------------------------------------------------------------------------
# OFFICIAL 10 GOVERNMENT OF INDIA WEBSITES REGISTRY
# --------------------------------------------------------------------------
OFFICIAL_WEBSITES: List[Dict[str, Any]] = [
    {
        "id": "pmindia",
        "name": "PM Office (PMINDIA)",
        "base_url": "https://www.pmindia.gov.in/",
        "canonical_domain": "pmindia.gov.in",
        "department": "Prime Minister's Office",
        "description": "Official portal of the Prime Minister of India with speeches, press releases, major national initiatives, and cabinet decisions.",
        "enabled": True,
        "allowed_domains": ["www.pmindia.gov.in", "pmindia.gov.in"],
        "news_sections": ["/en/news-updates/", "/en/speeches/"]
    },
    {
        "id": "meity",
        "name": "Ministry of Electronics and Information Technology (MeitY)",
        "base_url": "https://www.meity.gov.in/",
        "canonical_domain": "meity.gov.in",
        "department": "Ministry of Electronics & IT",
        "description": "Formulates IT policy, Digital India framework, cybersecurity regulations, semiconductor missions, and electronics governance.",
        "enabled": True,
        "allowed_domains": ["www.meity.gov.in", "meity.gov.in"],
        "news_sections": ["/notifications", "/press-release"]
    },
    {
        "id": "cabsec",
        "name": "Cabinet Secretariat",
        "base_url": "https://cabsec.gov.in/",
        "canonical_domain": "cabsec.gov.in",
        "department": "Cabinet Secretariat",
        "description": "Responsible for the administration of the Government of India (Transaction of Business) Rules and coordination among ministries.",
        "enabled": True,
        "allowed_domains": ["cabsec.gov.in"],
        "news_sections": ["/notifications.php", "/pressrelease.php"]
    },
    {
        "id": "mohfw",
        "name": "Ministry of Health and Family Welfare (MoHFW)",
        "base_url": "https://mohfw.gov.in/",
        "canonical_domain": "mohfw.gov.in",
        "department": "Ministry of Health & Family Welfare",
        "description": "Oversees public health policies, medical education, Ayushman Bharat, disease surveillance, and national health programs.",
        "enabled": True,
        "allowed_domains": ["www.mohfw.gov.in", "mohfw.gov.in"],
        "news_sections": ["/media/press-release", "/notifications"]
    },
    {
        "id": "mha",
        "name": "Ministry of Home Affairs (MHA)",
        "base_url": "https://www.mha.gov.in/",
        "canonical_domain": "mha.gov.in",
        "department": "Ministry of Home Affairs",
        "description": "Responsible for internal security, border management, disaster management, civil defense, and central police forces.",
        "enabled": True,
        "allowed_domains": ["www.mha.gov.in", "mha.gov.in"],
        "news_sections": ["/en/commoncontent/press-releases", "/en/notifications"]
    },
    {
        "id": "mib",
        "name": "Ministry of Information and Broadcasting (MIB)",
        "base_url": "https://mib.gov.in/",
        "canonical_domain": "mib.gov.in",
        "department": "Ministry of Information & Broadcasting",
        "description": "Governs media, broadcasting rules, digital news guidelines, and national communications.",
        "enabled": True,
        "allowed_domains": ["www.mib.gov.in", "mib.gov.in"],
        "news_sections": ["/press-releases", "/notifications"]
    },
    {
        "id": "doe",
        "name": "Department of Expenditure",
        "base_url": "https://doe.gov.in/",
        "canonical_domain": "doe.gov.in",
        "department": "Ministry of Finance - Department of Expenditure",
        "description": "Nodal department for overseeing public financial management system and matters affecting government finances.",
        "enabled": True,
        "allowed_domains": ["www.doe.gov.in", "doe.gov.in"],
        "news_sections": ["/notifications", "/orders-circulars"]
    },
    {
        "id": "dea",
        "name": "Department of Economic Affairs (DEA)",
        "base_url": "https://dea.gov.in/",
        "canonical_domain": "dea.gov.in",
        "department": "Ministry of Finance - Department of Economic Affairs",
        "description": "Formulates and monitors economic policies and programs of the central government, Union Budget, and foreign investment.",
        "enabled": True,
        "allowed_domains": ["www.dea.gov.in", "dea.gov.in"],
        "news_sections": ["/press-releases", "/notifications"]
    },
    {
        "id": "pib",
        "name": "Press Information Bureau (PIB)",
        "base_url": "https://www.pib.gov.in/",
        "canonical_domain": "pib.gov.in",
        "department": "Press Information Bureau",
        "description": "Official nodal agency of the Government of India to disseminate information to print and electronic media on government policies.",
        "enabled": True,
        "allowed_domains": ["www.pib.gov.in", "pib.gov.in"],
        "news_sections": ["/PressReleasePage.aspx", "/allRel.aspx"]
    },
    {
        "id": "india_gov",
        "name": "National Portal of India",
        "base_url": "https://www.india.gov.in/",
        "canonical_domain": "india.gov.in",
        "department": "National Portal of India",
        "description": "Single-window access to information and services being provided by the various Indian Government entities.",
        "enabled": True,
        "allowed_domains": ["www.india.gov.in", "india.gov.in"],
        "news_sections": ["/news_lists", "/spotlight"]
    }
]

# Quick domain allowlist mapping
APPROVED_DOMAINS: set = set()
for site in OFFICIAL_WEBSITES:
    APPROVED_DOMAINS.add(site["canonical_domain"])
    for d in site.get("allowed_domains", []):
        APPROVED_DOMAINS.add(d)

def is_approved_domain(url_or_domain: str) -> bool:
    """Validate if domain or URL matches the official approved government domains."""
    from urllib.parse import urlparse
    if "://" in url_or_domain:
        domain = urlparse(url_or_domain).netloc.lower()
    else:
        domain = url_or_domain.lower()
    # Strip port if any
    if ":" in domain:
        domain = domain.split(":")[0]
    
    if domain in APPROVED_DOMAINS:
        return True
    
    # Also check if it's a subdomain of an approved domain (ends with .gov.in or .nic.in)
    for approved in APPROVED_DOMAINS:
        if domain == approved or domain.endswith("." + approved):
            return True
    return False
