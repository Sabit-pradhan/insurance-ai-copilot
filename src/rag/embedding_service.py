# src/rag/embedding_service.py

import numpy as np
from openai import OpenAI

from src.core.config import settings
from src.core.logger import get_logger


logger = get_logger(__name__)


if not settings.OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from environment variables."
    )


client = OpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    timeout=settings.LLM_TIMEOUT_SECONDS
)


def normalize_embeddings(
    vectors: np.ndarray
) -> np.ndarray:
    """
    Normalize vectors to unit length.
    """

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return vectors / norms


def embed_texts(
    texts: list[str]
) -> np.ndarray:
    """
    Convert multiple text strings into
    semantic embedding vectors.
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32
        )

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return np.empty(
            (0, 0),
            dtype=np.float32
        )

    logger.info(
        f"Generating embeddings for "
        f"{len(cleaned_texts)} text(s)"
    )

    try:

        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=cleaned_texts
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index
        )

        vectors = np.asarray(
            [
                item.embedding
                for item in ordered_data
            ],
            dtype=np.float32
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
            "Embedding generation failed"
        )

        raise


def embed_query(
    question: str
) -> np.ndarray:
    """
    Convert one user question into
    one semantic embedding vector.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    vectors = embed_texts(
        [question]
    )

    return vectors[0]


if __name__ == "__main__":

    test_text = (
        "What documents are required "
        "for an insurance claim?"
    )

    vector = embed_query(
        test_text
    )

    print("\nEmbedding Model:")
    print(settings.EMBEDDING_MODEL)

    print("\nEmbedding Dimension:")
    print(len(vector))

    print("\nFirst 5 Values:")
    print(vector[:5])

    print("\nVector Norm:")
    print(np.linalg.norm(vector))