# src/rag/vector_store.py

import json
from pathlib import Path

import numpy as np

from src.core.config import settings
from src.core.logger import get_logger

from src.rag.document_loader import (
    load_documents,
)

from src.rag.embedding_service import (
    embed_texts,
    embed_query,
    EMBEDDING_MODEL,
)


logger = get_logger(__name__)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


RAG_INDEX_DIR = getattr(
    settings,
    "RAG_INDEX_DIR",
    "artifacts/rag_index",
)


INDEX_DIR = (
    PROJECT_ROOT
    / RAG_INDEX_DIR
)


EMBEDDINGS_FILE = (
    INDEX_DIR
    / "embeddings.npy"
)


METADATA_FILE = (
    INDEX_DIR
    / "metadata.json"
)


INDEX_INFO_FILE = (
    INDEX_DIR
    / "index_info.json"
)


# ==========================================================
# BUILD INDEX
# ==========================================================

def build_vector_index() -> None:
    """
    Build persistent semantic vector index.
    """

    records = load_documents()

    if not records:

        raise ValueError(
            "No document chunks found."
        )

    texts = [
        item["text"]
        for item in records
    ]

    logger.info(
        "Building local semantic vector index..."
    )

    embeddings = embed_texts(
        texts
    )

    if len(embeddings) != len(records):

        raise ValueError(
            "Embedding count does not match "
            "metadata count."
        )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    index_info = {
        "embedding_model": EMBEDDING_MODEL,
        "total_chunks": len(records),
        "embedding_dimension": (
            int(
                embeddings.shape[1]
            )
            if embeddings.ndim == 2
            else None
        ),
    }

    with open(
        INDEX_INFO_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            index_info,
            file,
            indent=2,
        )

    logger.info(
        f"Vector index created. "
        f"Chunks: {len(records)}, "
        f"Shape: {embeddings.shape}"
    )


# ==========================================================
# LOAD INDEX
# ==========================================================

def load_vector_index():
    """
    Load persistent vector index.
    """

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            "Embedding index not found. "
            "Run build_vector_index() first."
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "RAG metadata file not found."
        )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    if len(embeddings) != len(metadata):

        raise ValueError(
            "RAG index is inconsistent. "
            "Rebuild the vector index."
        )

    return (
        embeddings,
        metadata,
    )


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================

def semantic_search(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """
    Retrieve semantically relevant chunks.
    """

    if (
        not question
        or not question.strip()
    ):

        raise ValueError(
            "Question cannot be empty."
        )


    if top_k is None:

        top_k = getattr(
            settings,
            "RAG_TOP_K",
            4,
        )


    if min_score is None:

        min_score = getattr(
            settings,
            "RAG_MIN_SCORE",
            0.45,
        )


    embeddings, metadata = (
        load_vector_index()
    )


    query_vector = embed_query(
        question
    )


    # ------------------------------------------------------
    # Prevent using old incompatible indexes
    # ------------------------------------------------------

    if (
        embeddings.ndim != 2
        or embeddings.shape[1]
        != query_vector.shape[0]
    ):

        raise ValueError(
            "Embedding dimension mismatch. "
            "The RAG index was probably built "
            "with a different embedding model. "
            "Rebuild the vector index."
        )


    # ------------------------------------------------------
    # Cosine similarity
    #
    # Vectors are normalized, so:
    # dot product = cosine similarity
    # ------------------------------------------------------

    scores = (
        embeddings
        @ query_vector
    )


    # Retrieve extra candidates first.
    candidate_count = min(
        max(
            top_k * 3,
            top_k,
        ),
        len(metadata),
    )


    ranked_indices = (
        np.argsort(
            scores
        )[::-1][
            :candidate_count
        ]
    )


    results = []

    seen_chunks = set()


    for index in ranked_indices:

        score = float(
            scores[index]
        )

        if score < min_score:
            continue

        record = metadata[
            int(index)
        ].copy()

        text_key = (
            record["text"]
            .strip()
            .lower()
        )

        # Remove duplicate retrieved context
        if text_key in seen_chunks:
            continue

        seen_chunks.add(
            text_key
        )

        record["score"] = round(
            score,
            4,
        )

        results.append(
            record
        )

        if len(results) >= top_k:
            break


    logger.info(
        f"Semantic search returned "
        f"{len(results)} chunk(s) "
        f"above threshold {min_score}"
    )


    return results


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    build_vector_index()