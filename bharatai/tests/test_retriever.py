import tempfile
from core.vector_store import PersistentVectorStore

def test_vector_store_add_and_search():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = PersistentVectorStore(persist_dir=tmp_dir)
        chunks = [
            {
                "chunk_id": "c1",
                "english_text": "The Ministry of Health oversees the Ayushman Bharat PM-JAY health insurance scheme.",
                "source_url": "https://mohfw.gov.in/ayushman",
                "source_domain": "mohfw.gov.in",
                "department": "Ministry of Health & Family Welfare",
                "source_title": "Ayushman Bharat",
            },
            {
                "chunk_id": "c2",
                "english_text": "MeitY is responsible for the semiconductor production linked incentive (PLI) guidelines.",
                "source_url": "https://meity.gov.in/pli",
                "source_domain": "meity.gov.in",
                "department": "Ministry of Electronics & IT",
                "source_title": "Semiconductor PLI",
            },
        ]
        added = store.add_chunks(chunks)
        assert added == 2
        assert store.count() == 2

        # Query health
        res = store.similarity_search("health insurance Ayushman", top_k=2)
        assert len(res) >= 1
        assert "health" in res[0]["english_text"].lower()

if __name__ == "__main__":
    test_vector_store_add_and_search()
    print("test_retriever: All assertions passed!")
