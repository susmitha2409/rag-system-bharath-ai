"""
BharatAI — Centralized Groq LLM Client
Interacts with the Groq Chat Completions API with bounded retries, Retry-After handling,
detailed latency tracking, usage extraction, and robust error management.
Sole hosted LLM provider for the entire application.
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
try:
    import requests
except ImportError:
    requests = None
import urllib.request
import urllib.error

from config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    GROQ_MAX_RETRIES,
    GROQ_TEMPERATURE,
    GROQ_MAX_TOKENS,
)
from core.security import redact_secrets

logger = logging.getLogger("bharatai.llm_client")


class GroqLLMError(Exception):
    """Custom exception for Groq API interactions."""
    def __init__(self, message: str, status_code: Optional[int] = None, error_type: str = "GENERIC_ERROR"):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class GroqClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self.api_key = (api_key or GROQ_API_KEY or "").strip()
        self.api_url = (api_url or GROQ_API_URL).strip()
        self.model = (model or GROQ_MODEL).strip()
        self.timeout = timeout or GROQ_TIMEOUT_SECONDS
        self.max_retries = max_retries or GROQ_MAX_RETRIES

        self.session = requests.Session() if requests is not None else None

    def set_credentials(self, api_key: str, model: Optional[str] = None):
        """Update API credentials at runtime (e.g. from Streamlit UI Settings)."""
        self.api_key = api_key.strip()
        if model:
            self.model = model.strip()

    def is_configured(self) -> bool:
        """Check if an API key has been provided."""
        return bool(self.api_key and len(self.api_key) > 5)

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a chat completion request to the Groq API.
        Returns:
            Tuple[content: str, metadata: dict]
            metadata contains: latency_ms, prompt_tokens, completion_tokens, total_tokens, model
        """
        if not self.is_configured():
            raise GroqLLMError(
                "Groq API Key is not configured. Please configure your GROQ_API_KEY in the Settings page or .env file.",
                status_code=401,
                error_type="MISSING_API_KEY",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "BharatAI-AgenticRAG/1.0",
        }

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else GROQ_TEMPERATURE,
            "max_tokens": max_tokens or GROQ_MAX_TOKENS,
        }

        if response_format:
            payload["response_format"] = response_format

        attempt = 0
        backoff_delay = 1.0

        while attempt <= self.max_retries:
            attempt += 1
            start_perf = time.perf_counter()

            try:
                if requests is not None and self.session is not None:
                    response = self.session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    status_code = response.status_code
                    resp_text = response.text
                    resp_headers = response.headers
                else:
                    req_data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(self.api_url, data=req_data, headers=headers, method="POST")
                    try:
                        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                            status_code = resp.status
                            resp_text = resp.read().decode("utf-8")
                            resp_headers = resp.headers
                    except urllib.error.HTTPError as http_err:
                        status_code = http_err.code
                        resp_text = http_err.read().decode("utf-8")
                        resp_headers = http_err.headers

                latency_ms = (time.perf_counter() - start_perf) * 1000.0

                # Handle successful response
                if status_code == 200:
                    data = json.loads(resp_text)
                    choices = data.get("choices", [])
                    if not choices:
                        raise GroqLLMError("Groq API returned an empty choices array.", status_code=200, error_type="EMPTY_CHOICES")

                    content = choices[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})
                    metadata = {
                        "latency_ms": round(latency_ms, 2),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "model": data.get("model", self.model),
                        "finish_reason": choices[0].get("finish_reason", "stop"),
                    }
                    return content, metadata

                # Handle Rate Limiting (429) & Server Errors (500, 502, 503, 504)
                if status_code in (429, 500, 502, 503, 504):
                    retry_after_header = resp_headers.get("Retry-After")
                    wait_time = float(retry_after_header) if retry_after_header and str(retry_after_header).isdigit() else backoff_delay
                    logger.warning(
                        f"Groq API returned status {status_code} on attempt {attempt}. Retrying after {wait_time}s..."
                    )
                    if attempt > self.max_retries:
                        raise GroqLLMError(
                            f"Groq API rate limit or server error ({status_code}): {redact_secrets(resp_text)}",
                            status_code=status_code,
                            error_type="RATE_LIMIT_OR_SERVER_ERROR",
                        )
                    time.sleep(wait_time)
                    backoff_delay *= 2.0
                    continue

                # Handle Client Authentication / Bad Request Errors
                if status_code in (401, 403):
                    raise GroqLLMError(
                        "Groq API authentication failed. Please verify your GROQ_API_KEY.",
                        status_code=status_code,
                        error_type="AUTH_FAILED",
                    )

                if status_code == 400:
                    err_msg = redact_secrets(resp_text)
                    raise GroqLLMError(
                        f"Groq API Bad Request (400): {err_msg}",
                        status_code=400,
                        error_type="BAD_REQUEST",
                    )

                # Other HTTP status
                raise GroqLLMError(
                    f"Groq API returned unexpected status code {status_code}: {redact_secrets(resp_text)}",
                    status_code=status_code,
                    error_type="UNEXPECTED_STATUS",
                )

            except requests.exceptions.Timeout:
                logger.warning(f"Groq API timeout on attempt {attempt} after {self.timeout}s.")
                if attempt > self.max_retries:
                    raise GroqLLMError(
                        f"Groq API request timed out after {self.timeout}s after {self.max_retries} retries.",
                        status_code=408,
                        error_type="TIMEOUT",
                    )
                time.sleep(backoff_delay)
                backoff_delay *= 2.0

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Groq API connection error on attempt {attempt}: {e}")
                if attempt > self.max_retries:
                    raise GroqLLMError(
                        "Failed to connect to Groq API. Please verify internet connectivity.",
                        status_code=503,
                        error_type="CONNECTION_ERROR",
                    )
                time.sleep(backoff_delay)
                backoff_delay *= 2.0

            except GroqLLMError:
                raise

            except Exception as e:
                logger.error(f"Unexpected error communicating with Groq API: {e}")
                raise GroqLLMError(f"Unexpected error: {redact_secrets(str(e))}", error_type="UNKNOWN_ERROR")

        raise GroqLLMError("Exhausted retries calling Groq API.", error_type="RETRIES_EXHAUSTED")

    def generate_json_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Request JSON structured output from Groq.
        Includes automatic fallback parsing for markdown-wrapped JSON code blocks.
        """
        raw_text, meta = self.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            model=model,
            response_format={"type": "json_object"},
        )

        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            return parsed, meta
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Raw: {raw_text[:200]}")
            # Try to find JSON substring {...}
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    parsed = json.loads(cleaned[start_idx : end_idx + 1])
                    return parsed, meta
                except Exception:
                    pass
            # Return fallback dictionary with raw content
            return {"raw_output": raw_text, "parse_error": True}, meta

    def test_connection(self) -> Dict[str, Any]:
        """
        Verify Groq API connectivity and measure latency.
        Used on Settings page and startup diagnostics.
        """
        if not self.is_configured():
            return {
                "success": False,
                "status": "NOT_CONFIGURED",
                "message": "Groq API key is not configured.",
                "latency_ms": 0.0,
                "model": self.model,
            }

        start = time.perf_counter()
        try:
            messages = [
                {"role": "system", "content": "You are BharatAI assistant diagnostic tester. Respond with 'PONG'."},
                {"role": "user", "content": "PING"},
            ]
            content, meta = self.generate_chat_completion(messages=messages, max_tokens=10, temperature=0.0)
            latency_ms = (time.perf_counter() - start) * 1000.0

            return {
                "success": True,
                "status": "CONNECTED",
                "message": f"Successfully connected to Groq API using model '{meta.get('model', self.model)}'. Response: {content.strip()}",
                "latency_ms": round(latency_ms, 2),
                "model": meta.get("model", self.model),
                "tokens": meta.get("total_tokens", 0),
            }
        except GroqLLMError as e:
            return {
                "success": False,
                "status": "ERROR",
                "message": str(e),
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
                "model": self.model,
            }
        except Exception as e:
            return {
                "success": False,
                "status": "EXCEPTION",
                "message": redact_secrets(str(e)),
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
                "model": self.model,
            }


# Singleton GroqClient instance
_groq_client_instance: Optional[GroqClient] = None

def get_groq_client() -> GroqClient:
    global _groq_client_instance
    if _groq_client_instance is None:
        _groq_client_instance = GroqClient()
    return _groq_client_instance
