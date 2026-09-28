# src/rag/embedding_service.py

import os

import numpy as np
from langchain_ollama import OllamaEmbeddings

from src.core.config import settings
from src.core.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# OLLAMA CONFIG
# ==========================================================

OLLAMA_BASE_URL = getattr(
    settings,
    "OLLAMA_BASE_URL",
    os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ),
)


EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text",
)


# ==========================================================
# INITIALIZE EMBEDDING MODEL
# ==========================================================

embedding_model = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_embeddings(
    vectors: np.ndarray,
) -> np.ndarray:
    """
    Normalize vectors to unit length.

    After normalization:
        dot product = cosine similarity
    """

    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms[
        norms == 0
    ] = 1.0

    return (
        vectors
        / norms
    )


# ==========================================================
# EMBED DOCUMENTS
# ==========================================================

def embed_texts(
    texts: list[str],
) -> np.ndarray:
    """
    Generate local semantic embeddings
    using Ollama.
    """

    if not texts:

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    cleaned_texts = [
        text.strip()
        for text in texts
        if text
        and text.strip()
    ]

    if not cleaned_texts:

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    logger.info(
        f"Generating local embeddings "
        f"for {len(cleaned_texts)} text(s) "
        f"using {EMBEDDING_MODEL}"
    )

    try:

        vectors = (
            embedding_model
            .embed_documents(
                cleaned_texts
            )
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        vectors = normalize_embeddings(
            vectors
        )

        logger.info(
            f"Embedding generation successful. "
            f"Shape: {vectors.shape}"
        )

        return vectors

    except Exception:

        logger.exception(
            "Local embedding generation failed"
        )

        raise


# ==========================================================
# EMBED QUERY
# ==========================================================

def embed_query(
    question: str,
) -> np.ndarray:
    """
    Generate one normalized query embedding.
    """

    if (
        not question
        or not question.strip()
    ):

        raise ValueError(
            "Question cannot be empty."
        )

    try:

        vector = (
            embedding_model
            .embed_query(
                question.strip()
            )
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            vector
        )

        if norm > 0:

            vector = (
                vector
                / norm
            )

        return vector

    except Exception:

        logger.exception(
            "Query embedding generation failed"
        )

        raise


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    question = (
        "What documents are required "
        "for an insurance claim?"
    )

    vector = embed_query(
        question
    )

    print(
        "\nEmbedding Model:"
    )

    print(
        EMBEDDING_MODEL
    )

    print(
        "\nEmbedding Dimension:"
    )

    print(
        len(vector)
    )

    print(
        "\nFirst 5 values:"
    )

    print(
        vector[:5]
    )

    print(
        "\nVector Norm:"
    )

    print(
        np.linalg.norm(vector)
    )