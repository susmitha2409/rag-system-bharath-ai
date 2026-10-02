from core.document_processor import DocumentProcessor
from core.language_processor import LanguageProcessor

def test_language_detection():
    lp = LanguageProcessor()
    assert lp.detect_language("Prime Minister Narendra Modi launched the national mission.") == "en"
    assert lp.detect_language("प्रधानमंत्री नरेंद्र मोदी ने नई योजना का शुभारंभ किया।") == "hi"
    assert lp.detect_language("பிரதமர் நரேந்திர மோடி புதிய திட்டத்தைத் தொடங்கினார்.") == "ta"

def test_document_chunking():
    dp = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "The Digital India program is a flagship program of the Government of India with a vision "
        "to transform India into a digitally empowered society and knowledge economy. "
        "It was launched on July 1, 2015, by Prime Minister Narendra Modi."
    )
    doc_rec = dp.process_plain_text(
        text=sample_text,
        url="https://meity.gov.in/digital-india",
        department="Ministry of Electronics & IT",
        domain="meity.gov.in",
        title="Digital India",
    )
    assert doc_rec["num_chunks"] >= 1
    assert doc_rec["chunks"][0]["source_domain"] == "meity.gov.in"
    assert doc_rec["chunks"][0]["department"] == "Ministry of Electronics & IT"

if __name__ == "__main__":
    test_language_detection()
    test_document_chunking()
    print("test_document_processor: All assertions passed!")
