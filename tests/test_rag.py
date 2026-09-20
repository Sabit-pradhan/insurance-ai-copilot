# tests/test_rag.py

import numpy as np

from src.rag import vector_store


# --------------------------------------------------
# Test 1:
# Relevant chunk should be returned
# --------------------------------------------------

def test_semantic_search_returns_relevant_chunk(
    monkeypatch
):

    # Fake stored document embeddings
    fake_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.9, 0.0]
        ],
        dtype=np.float32
    )

    # Fake metadata
    fake_metadata = [
        {
            "source": "claim_guide.txt",
            "page": None,
            "chunk": 1,
            "text": "Documents required for claim submission."
        },
        {
            "source": "claim_guide.txt",
            "page": None,
            "chunk": 2,
            "text": "This is demo information."
        }
    ]

    # Fake user query embedding
    fake_query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32
    )

    # Replace real index loading
    monkeypatch.setattr(
        vector_store,
        "load_vector_index",
        lambda: (
            fake_embeddings,
            fake_metadata
        )
    )

    # Replace real OpenRouter embedding call
    monkeypatch.setattr(
        vector_store,
        "embed_query",
        lambda question: fake_query
    )

    results = vector_store.semantic_search(
        "What documents are required for a claim?"
    )

    assert len(results) == 1

    assert (
        results[0]["source"]
        == "claim_guide.txt"
    )

    assert results[0]["chunk"] == 1

    assert results[0]["score"] >= 0.45


# --------------------------------------------------
# Test 2:
# Weak results should be filtered
# --------------------------------------------------

def test_semantic_search_filters_weak_chunks(
    monkeypatch
):

    fake_embeddings = np.array(
        [
            [0.2, 0.9, 0.0],
            [0.1, 0.8, 0.2]
        ],
        dtype=np.float32
    )

    fake_metadata = [
        {
            "source": "claim_guide.txt",
            "page": None,
            "chunk": 1,
            "text": "Unrelated content one."
        },
        {
            "source": "claim_guide.txt",
            "page": None,
            "chunk": 2,
            "text": "Unrelated content two."
        }
    ]

    fake_query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32
    )

    monkeypatch.setattr(
        vector_store,
        "load_vector_index",
        lambda: (
            fake_embeddings,
            fake_metadata
        )
    )

    monkeypatch.setattr(
        vector_store,
        "embed_query",
        lambda question: fake_query
    )

    results = vector_store.semantic_search(
        "Completely unrelated question"
    )

    # Scores are below RAG_MIN_SCORE
    assert results == []


# --------------------------------------------------
# Test 3:
# Empty question should fail safely
# --------------------------------------------------

def test_semantic_search_rejects_empty_question():

    try:

        vector_store.semantic_search("")

        assert False

    except ValueError as error:

        assert (
            "Question cannot be empty"
            in str(error)
        )