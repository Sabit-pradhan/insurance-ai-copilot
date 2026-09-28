# src/rag/reranker.py

import os
import re

from src.core.logger import get_logger


# ==========================================================
# LOGGER
# ==========================================================

logger = get_logger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

RAG_SEMANTIC_WEIGHT = float(
    os.getenv(
        "RAG_SEMANTIC_WEIGHT",
        "0.75",
    )
)

RAG_LEXICAL_WEIGHT = float(
    os.getenv(
        "RAG_LEXICAL_WEIGHT",
        "0.25",
    )
)


# ==========================================================
# SIMPLE STOP WORDS
# ==========================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize text for lexical comparison.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ==========================================================
# EXTRACT IMPORTANT TERMS
# ==========================================================

def extract_terms(
    text: str,
) -> set[str]:
    """
    Extract useful words and remove common
    stop words.
    """

    normalized = normalize_text(
        text
    )

    terms = normalized.split()

    return {
        term
        for term in terms
        if (
            term not in STOP_WORDS
            and len(term) > 2
        )
    }


# ==========================================================
# LEXICAL RELEVANCE
# ==========================================================

def calculate_lexical_score(
    question: str,
    chunk_text: str,
) -> float:
    """
    Calculate how many important query terms
    appear inside a retrieved chunk.

    Score range:
        0.0 -> no useful overlap
        1.0 -> all important query terms found
    """

    question_terms = extract_terms(
        question
    )

    chunk_terms = extract_terms(
        chunk_text
    )

    if not question_terms:
        return 0.0

    matched_terms = (
        question_terms
        & chunk_terms
    )

    return (
        len(matched_terms)
        / len(question_terms)
    )


# ==========================================================
# CALCULATE RERANK SCORE
# ==========================================================

def calculate_rerank_score(
    semantic_score: float,
    lexical_score: float,
) -> float:
    """
    Combine semantic similarity and
    lexical query relevance.
    """

    total_weight = (
        RAG_SEMANTIC_WEIGHT
        + RAG_LEXICAL_WEIGHT
    )

    if total_weight <= 0:
        return semantic_score

    semantic_weight = (
        RAG_SEMANTIC_WEIGHT
        / total_weight
    )

    lexical_weight = (
        RAG_LEXICAL_WEIGHT
        / total_weight
    )

    final_score = (
        semantic_score
        * semantic_weight
        +
        lexical_score
        * lexical_weight
    )

    return round(
        final_score,
        4,
    )


# ==========================================================
# RERANK CHUNKS
# ==========================================================

def rerank_chunks(
    question: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Rerank retrieved chunks using:

    1. Semantic embedding similarity
    2. Query keyword relevance

    Original semantic score is preserved.

    New fields:
        semantic_score
        lexical_score
        rerank_score
    """

    if not chunks:
        return []

    reranked = []

    for chunk in chunks:

        item = chunk.copy()

        semantic_score = float(
            item.get(
                "score",
                0.0,
            )
        )

        lexical_score = (
            calculate_lexical_score(
                question=question,
                chunk_text=item.get(
                    "text",
                    "",
                ),
            )
        )

        rerank_score = (
            calculate_rerank_score(
                semantic_score=semantic_score,
                lexical_score=lexical_score,
            )
        )

        item[
            "semantic_score"
        ] = round(
            semantic_score,
            4,
        )

        item[
            "lexical_score"
        ] = round(
            lexical_score,
            4,
        )

        item[
            "rerank_score"
        ] = rerank_score

        reranked.append(
            item
        )

    # Highest rerank score first
    reranked = sorted(
        reranked,
        key=lambda item: item.get(
            "rerank_score",
            0.0,
        ),
        reverse=True,
    )

    logger.info(
        "Reranked %s chunk(s)",
        len(reranked),
    )

    return reranked


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    question = (
        "What documents are required "
        "for an insurance claim?"
    )

    test_chunks = [
        {
            "chunk_id": "test_1",
            "source": "claim_guide.txt",
            "score": 0.83,
            "text": (
                "Documents required for claim submission "
                "include a completed claim form, insurance "
                "policy copy and identity proof."
            ),
        },
        {
            "chunk_id": "test_2",
            "source": "claim_guide.txt",
            "score": 0.75,
            "text": (
                "After verification the eligible settlement "
                "amount is processed according to policy terms."
            ),
        },
    ]

    results = rerank_chunks(
        question=question,
        chunks=test_chunks,
    )

    print(
        "\nQUESTION:"
    )

    print(
        question
    )

    print(
        "\nRERANKED CHUNKS:"
    )

    for item in results:

        print(
            "\n"
            + "-" * 60
        )

        print(
            "Chunk ID:",
            item.get(
                "chunk_id"
            ),
        )

        print(
            "Semantic Score:",
            item.get(
                "semantic_score"
            ),
        )

        print(
            "Lexical Score:",
            item.get(
                "lexical_score"
            ),
        )

        print(
            "Rerank Score:",
            item.get(
                "rerank_score"
            ),
        )