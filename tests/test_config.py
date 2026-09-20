# tests/test_config.py

from src.core.config import settings


def test_app_name():
    """
    Verify application configuration loads correctly.
    """

    assert settings.APP_NAME == "Insurance AI Copilot"


def test_rag_configuration():
    """
    Verify RAG configuration is valid.
    """

    assert settings.RAG_TOP_K > 0

    assert 0 <= settings.RAG_MIN_SCORE <= 1

    assert settings.RAG_CHUNK_SIZE > 0

    assert settings.RAG_CHUNK_OVERLAP >= 0


def test_rag_chunk_overlap():
    """
    Chunk overlap must always be smaller
    than the actual chunk size.
    """

    assert (
        settings.RAG_CHUNK_OVERLAP
        < settings.RAG_CHUNK_SIZE
    )