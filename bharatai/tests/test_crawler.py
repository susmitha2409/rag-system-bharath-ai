from core.security import validate_crawl_url
from config import is_approved_domain

def test_approved_domains():
    assert is_approved_domain("https://www.pmindia.gov.in/en/")
    assert is_approved_domain("https://meity.gov.in/notifications")
    assert is_approved_domain("https://pib.gov.in")
    assert is_approved_domain("https://cabsec.gov.in")
    assert not is_approved_domain("https://google.com")
    assert not is_approved_domain("https://malicious-site.org")

def test_ssrf_protection():
    # Localhost and internal network blocking
    valid, msg = validate_crawl_url("http://localhost:8000/admin")
    assert not valid
    assert "SSRF" in msg or "forbidden" in msg or "not in approved" in msg

    valid, msg = validate_crawl_url("http://127.0.0.1/secrets")
    assert not valid

    valid, msg = validate_crawl_url("http://169.254.169.254/latest/meta-data")
    assert not valid

if __name__ == "__main__":
    test_approved_domains()
    test_ssrf_protection()
    print("test_crawler: All assertions passed!")
