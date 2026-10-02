"""
BharatAI — Security and Sanitization Module
Provides domain allowlisting, SSRF defenses, file upload validation,
prompt injection shielding for untrusted retrieved content, and secret redacting.
"""

import re
import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple, List, Optional
from config import APPROVED_DOMAINS, is_approved_domain

# Maximum file upload size: 25 MB
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_FILE_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}

# Disallowed internal / private network blocks for SSRF protection
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def validate_crawl_url(url: str) -> Tuple[bool, str]:
    """
    Validate that a URL is safe to crawl:
    1. Must use http or https scheme
    2. Must match approved Government of India domains or explicit subdomains
    3. Must not resolve to localhost, private/link-local IP, or cloud metadata IP (SSRF defense)
    """
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL"

    try:
        parsed = urlparse(url.strip())
    except Exception as e:
        return False, f"Malformed URL: {e}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported URL scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL lacks valid hostname"

    # Block obvious local names
    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "metadata.google.internal"):
        return False, "Access to localhost or metadata endpoints is forbidden (SSRF protection)"

    # Check approved government domain list
    if not is_approved_domain(hostname):
        return False, f"Domain '{hostname}' is not in approved Government of India registry"

    # Resolve IP and verify not private / link-local / loopback
    try:
        ip_addr = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip_addr)
        for net in BLOCKED_IP_NETWORKS:
            if ip_obj in net:
                return False, f"Resolved IP {ip_addr} is in a restricted private network (SSRF defense)"
    except Exception:
        # If DNS fails at validation time, crawler will handle standard connection error
        pass

    return True, "Valid"


def validate_file_upload(filename: str, file_size_bytes: int) -> Tuple[bool, str]:
    """Validate uploaded document against allowed extensions and size limits."""
    if file_size_bytes > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        return False, f"File exceeds maximum allowed size of {max_mb} MB"

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_FILE_EXTENSIONS:
        return False, f"File type '{ext}' not allowed. Allowed types: {', '.join(sorted(ALLOWED_FILE_EXTENSIONS))}"

    return True, "Valid"


def sanitize_retrieved_evidence(text: str) -> str:
    """
    Shield against prompt injection in retrieved web content.
    Prevents untrusted webpage text from instructing the model to disregard previous instructions.
    """
    if not text:
        return ""

    # Neutralize markdown code block injection and fake system tags
    sanitized = text.replace("```", "'''")
    patterns_to_neutralize = [
        (r"(?i)<system>.*?</system>", "[TAG REMOVED]"),
        (r"(?i)<assistant>.*?</assistant>", "[TAG REMOVED]"),
        (r"(?i)<user>.*?</user>", "[TAG REMOVED]"),
        (r"(?i)ignore (all )?previous instructions", "[PROMPT OVERRIDE ATTEMPT REDACTED]"),
        (r"(?i)disregard the above", "[PROMPT OVERRIDE ATTEMPT REDACTED]"),
        (r"(?i)system prompt:", "Document note:"),
        (r"(?i)you are now", "Content mentions:"),
    ]
    for pattern, replacement in patterns_to_neutralize:
        sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized


def redact_secrets(text: str) -> str:
    """Redact API keys or tokens from logs, traces, and exports."""
    if not text or not isinstance(text, str):
        return text

    # Matches gsk_... (Groq keys) or standard 32+ hex/alphanumeric tokens
    redacted = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[REDACTED_GROQ_KEY]", text)
    redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[a-zA-Z0-9_.\-]+", r"\1[REDACTED_TOKEN]", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[a-zA-Z0-9_\-]{16,}", r"\1[REDACTED_KEY]", redacted)
    return redacted
