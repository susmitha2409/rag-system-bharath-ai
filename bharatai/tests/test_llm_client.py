from core.llm_client import GroqClient, GroqLLMError
from core.security import redact_secrets

def test_groq_client_unconfigured():
    client = GroqClient(api_key="")
    assert not client.is_configured()
    raised = False
    try:
        client.generate_chat_completion([{"role": "user", "content": "hi"}])
    except GroqLLMError as e:
        raised = True
        assert "not configured" in str(e).lower()
    assert raised, "Should have raised GroqLLMError when unconfigured"

def test_groq_client_set_credentials():
    client = GroqClient(api_key="")
    client.set_credentials("gsk_mock_test_key_1234567890", model="llama-3.3-70b-versatile")
    assert client.is_configured()
    assert client.model == "llama-3.3-70b-versatile"

def test_redact_secrets():
    raw = "My api key is gsk_abcdef12345678901234567890 and Bearer xyz1234567890"
    redacted = redact_secrets(raw)
    assert "gsk_abcdef12345678901234567890" not in redacted
    assert "[REDACTED_GROQ_KEY]" in redacted

if __name__ == "__main__":
    test_groq_client_unconfigured()
    test_groq_client_set_credentials()
    test_redact_secrets()
    print("test_llm_client: All assertions passed!")
