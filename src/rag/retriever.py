# src/rag/retriever.py

import os
import re

from src.rag.vector_store import semantic_search
from src.rag.reranker import rerank_chunks
from src.core.logger import get_logger


# ==========================================================
# LOGGER
# ==========================================================

logger = get_logger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

RAG_CANDIDATE_K = int(
    os.getenv(
        "RAG_CANDIDATE_K",
        "12",
    )
)

RAG_FINAL_K = int(
    os.getenv(
        "RAG_FINAL_K",
        "4",
    )
)

RAG_MIN_SCORE = float(
    os.getenv(
        "RAG_MIN_SCORE",
        "0.45",
    )
)

RAG_RELATIVE_SCORE_RATIO = float(
    os.getenv(
        "RAG_RELATIVE_SCORE_RATIO",
        "0.75",
    )
)

RAG_DEDUP_THRESHOLD = float(
    os.getenv(
        "RAG_DEDUP_THRESHOLD",
        "0.80",
    )
)

RAG_MAX_CHUNKS_PER_SOURCE = int(
    os.getenv(
        "RAG_MAX_CHUNKS_PER_SOURCE",
        "3",
    )
)


# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize text for duplicate comparison.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    return text.strip()


# ==========================================================
# TOKENIZE
# ==========================================================

def tokenize(
    text: str,
) -> set[str]:
    """
    Convert text into a unique token set.
    """

    normalized = normalize_text(
        text
    )

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


# ==========================================================
# TEXT SIMILARITY
# ==========================================================

def text_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Calculate Jaccard similarity between
    two text chunks.

    1.0 = very similar
    0.0 = completely different
    """

    tokens_a = tokenize(
        text_a
    )

    tokens_b = tokenize(
        text_b
    )

    if (
        not tokens_a
        or not tokens_b
    ):
        return 0.0

    intersection = (
        tokens_a
        & tokens_b
    )

    union = (
        tokens_a
        | tokens_b
    )

    if not union:
        return 0.0

    return (
        len(intersection)
        / len(union)
    )


# ==========================================================
# RELATIVE SCORE FILTERING
# ==========================================================

def filter_relative_scores(
    chunks: list[dict],
    ratio: float | None = None,
) -> list[dict]:
    """
    Remove weak chunks relative to the best
    retrieved semantic match.

    Example:

    Best score = 0.83
    Ratio      = 0.75

    Threshold:
        0.83 * 0.75 = 0.6225
    """

    if not chunks:
        return []

    if ratio is None:
        ratio = (
            RAG_RELATIVE_SCORE_RATIO
        )

    # Find strongest semantic score
    best_score = max(
        item.get(
            "score",
            0.0,
        )
        for item in chunks
    )

    # Dynamic threshold
    relative_threshold = (
        best_score
        * ratio
    )

    # Keep only sufficiently relevant chunks
    filtered = [
        item
        for item in chunks
        if item.get(
            "score",
            0.0,
        ) >= relative_threshold
    ]

    logger.info(
        "Relative score filtering: %s -> %s chunks "
        "(best=%.4f threshold=%.4f ratio=%.2f)",
        len(chunks),
        len(filtered),
        best_score,
        relative_threshold,
        ratio,
    )

    return filtered


# ==========================================================
# REMOVE NEAR-DUPLICATE CHUNKS
# ==========================================================

def remove_near_duplicates(
    chunks: list[dict],
    threshold: float | None = None,
) -> list[dict]:
    """
    Remove chunks containing nearly
    the same information.

    Useful because chunk overlap may retrieve
    repetitive content.
    """

    if threshold is None:
        threshold = (
            RAG_DEDUP_THRESHOLD
        )

    unique_chunks = []

    for candidate in chunks:

        candidate_text = candidate.get(
            "text",
            "",
        )

        duplicate = False

        for selected in unique_chunks:

            selected_text = selected.get(
                "text",
                "",
            )

            similarity = text_similarity(
                candidate_text,
                selected_text,
            )

            if similarity >= threshold:

                duplicate = True

                logger.debug(
                    "Near-duplicate chunk removed: "
                    "%s similarity=%.4f",
                    candidate.get(
                        "chunk_id"
                    ),
                    similarity,
                )

                break

        if not duplicate:

            unique_chunks.append(
                candidate
            )

    logger.info(
        "Duplicate filtering: %s -> %s chunks",
        len(chunks),
        len(unique_chunks),
    )

    return unique_chunks


# ==========================================================
# SOURCE DIVERSITY
# ==========================================================

def apply_source_diversity(
    chunks: list[dict],
    max_chunks_per_source: int | None = None,
) -> list[dict]:
    """
    Prevent one document from dominating
    the final RAG context.
    """

    if max_chunks_per_source is None:
        max_chunks_per_source = (
            RAG_MAX_CHUNKS_PER_SOURCE
        )

    selected = []
    source_counts = {}

    for chunk in chunks:

        source = chunk.get(
            "source",
            "unknown",
        )

        current_count = source_counts.get(
            source,
            0,
        )

        if (
            current_count
            >= max_chunks_per_source
        ):
            continue

        selected.append(
            chunk
        )

        source_counts[
            source
        ] = (
            current_count
            + 1
        )

    logger.info(
        "Source diversity filtering: %s -> %s chunks",
        len(chunks),
        len(selected),
    )

    return selected


# ==========================================================
# FINAL CONTEXT SELECTION
# ==========================================================

def select_final_chunks(
    chunks: list[dict],
    final_k: int | None = None,
) -> list[dict]:
    """
    Select strongest final chunks.

    Priority:
    1. rerank_score
    2. semantic score fallback
    """

    if final_k is None:
        final_k = (
            RAG_FINAL_K
        )

    sorted_chunks = sorted(
        chunks,
        key=lambda item: item.get(
            "rerank_score",
            item.get(
                "score",
                0.0,
            ),
        ),
        reverse=True,
    )

    return sorted_chunks[
        :final_k
    ]


# ==========================================================
# MAIN RETRIEVER
# ==========================================================

def retrieve_context(
    question: str,
    candidate_k: int | None = None,
    final_k: int | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """
    Enhanced RAG retrieval pipeline.

    Flow:

    Question
        ↓
    Semantic Candidate Retrieval
        ↓
    Absolute Similarity Threshold
        ↓
    Relative Score Filtering
        ↓
    Near-Duplicate Removal
        ↓
    Hybrid Reranking
        ↓
    Source Diversity
        ↓
    Final Context Selection
    """

    # ------------------------------------------------------
    # Validate question
    # ------------------------------------------------------

    if (
        not question
        or not question.strip()
    ):
        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    # ------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------

    if candidate_k is None:
        candidate_k = (
            RAG_CANDIDATE_K
        )

    if final_k is None:
        final_k = (
            RAG_FINAL_K
        )

    if min_score is None:
        min_score = (
            RAG_MIN_SCORE
        )

    logger.info(
        "Enhanced retrieval started. "
        "candidate_k=%s final_k=%s min_score=%s",
        candidate_k,
        final_k,
        min_score,
    )

    # ======================================================
    # STEP 1 — SEMANTIC CANDIDATE RETRIEVAL
    # ======================================================

    candidates = semantic_search(
        question=question,
        top_k=candidate_k,
        min_score=min_score,
    )

    if not candidates:

        logger.info(
            "No semantic retrieval candidates found."
        )

        return []

    logger.info(
        "Semantic retrieval produced %s candidate(s)",
        len(candidates),
    )

    # ======================================================
    # STEP 2 — RELATIVE SCORE FILTERING
    # ======================================================

    candidates = filter_relative_scores(
        candidates
    )

    if not candidates:

        logger.info(
            "No candidates remained after "
            "relative score filtering."
        )

        return []

    # ======================================================
    # STEP 3 — REMOVE NEAR DUPLICATES
    # ======================================================

    candidates = remove_near_duplicates(
        candidates
    )

    if not candidates:

        logger.info(
            "No candidates remained after "
            "duplicate filtering."
        )

        return []

    # ======================================================
    # STEP 4 — HYBRID RERANKING
    # ======================================================

    candidates = rerank_chunks(
        question=question,
        chunks=candidates,
    )

    logger.info(
        "Reranking completed for %s chunk(s)",
        len(candidates),
    )

    # ======================================================
    # STEP 5 — SOURCE DIVERSITY
    # ======================================================

    candidates = apply_source_diversity(
        candidates
    )

    # ======================================================
    # STEP 6 — FINAL CONTEXT SELECTION
    # ======================================================

    final_chunks = select_final_chunks(
        chunks=candidates,
        final_k=final_k,
    )

    logger.info(
        "Enhanced retrieval selected %s final chunk(s)",
        len(final_chunks),
    )

    return final_chunks


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    test_question = (
        "What documents are required "
        "for an insurance claim?"
    )

    results = retrieve_context(
        test_question
    )

    print(
        "\nQUESTION:"
    )

    print(
        test_question
    )

    print(
        "\nFINAL RETRIEVED CHUNKS:"
    )

    print(
        "Count:",
        len(results),
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
            "Source:",
            item.get(
                "source"
            ),
        )

        print(
            "Chunk:",
            item.get(
                "chunk"
            ),
        )

        print(
            "Semantic Score:",
            item.get(
                "semantic_score",
                item.get(
                    "score"
                ),
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

        print(
            "Text:"
        )

        print(
            item.get(
                "text",
                "",
            )[:500]
        )