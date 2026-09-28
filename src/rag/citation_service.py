# src/rag/citation_service.py

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

RAG_CLAIM_SUPPORT_THRESHOLD = float(
    os.getenv(
        "RAG_CLAIM_SUPPORT_THRESHOLD",
        "0.45",
    )
)


# ==========================================================
# STOP WORDS
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
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize text before grounding comparison.
    """

    if not text:
        return ""

    text = text.lower()

    # Remove source citations
    text = re.sub(
        r"\[SOURCE\s+\d+\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ==========================================================
# EXTRACT TERMS
# ==========================================================

def extract_terms(
    text: str,
) -> set[str]:
    """
    Extract important words for grounding comparison.
    """

    normalized = normalize_text(
        text
    )

    if not normalized:
        return set()

    return {
        word
        for word in normalized.split()
        if (
            word not in STOP_WORDS
            and len(word) > 2
        )
    }


# ==========================================================
# EXTRACT CITATIONS
# ==========================================================

def get_cited_source_numbers(
    answer: str,
) -> list[int]:
    """
    Extract cited source numbers.

    Example:

    [SOURCE 1]
    [SOURCE 2]

    Returns:

    [1, 2]
    """

    if not answer:
        return []

    numbers = re.findall(
        r"\[SOURCE\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    )

    # Remove duplicates while preserving order
    result = []

    for number in numbers:

        number = int(
            number
        )

        if number not in result:
            result.append(
                number
            )

    return result


# ==========================================================
# VALIDATE CITATION NUMBERS
# ==========================================================

def validate_citation_numbers(
    answer: str,
    source_count: int,
) -> dict:
    """
    Check whether cited source numbers
    actually exist.
    """

    citations = get_cited_source_numbers(
        answer
    )

    if not citations:

        return {
            "valid": False,
            "citations": [],
            "invalid_citations": [],
            "reason": (
                "No source citation was found."
            ),
        }

    invalid = [
        number
        for number in citations
        if (
            number < 1
            or number > source_count
        )
    ]

    return {
        "valid": len(invalid) == 0,
        "citations": citations,
        "invalid_citations": invalid,
        "reason": (
            None
            if not invalid
            else "Answer contains invalid source citations."
        ),
    }


# ==========================================================
# SPLIT ANSWER INTO CLAIMS
# ==========================================================

def split_claims(
    answer: str,
) -> list[str]:
    """
    Break an answer into individual claims.

    Supports:
    - sentences
    - numbered lists
    - bullet lists
    """

    if not answer:
        return []

    answer = re.sub(
        r"\[SOURCE\s+\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    parts = re.split(
        r"\n+|(?<=[.!?])\s+",
        answer,
    )

    claims = []

    for part in parts:

        part = part.strip()

        # Remove bullet / numbering prefix
        part = re.sub(
            r"^\s*(?:[-*•]|\d+[.)])\s*",
            "",
            part,
        )

        if len(
            extract_terms(
                part
            )
        ) >= 2:

            claims.append(
                part
            )

    return claims


# ==========================================================
# CLAIM SUPPORT SCORE
# ==========================================================

def calculate_claim_support(
    claim: str,
    source_text: str,
) -> float:
    """
    Calculate how much of a claim is
    supported lexically by source text.

    1.0 = all important claim terms appear
          in source.

    0.0 = no useful overlap.
    """

    claim_terms = extract_terms(
        claim
    )

    source_terms = extract_terms(
        source_text
    )

    if not claim_terms:
        return 1.0

    if not source_terms:
        return 0.0

    matched_terms = (
        claim_terms
        & source_terms
    )

    score = (
        len(matched_terms)
        / len(claim_terms)
    )

    return round(
        score,
        4,
    )


# ==========================================================
# GET CITED SOURCE TEXTS
# ==========================================================

def get_cited_source_texts(
    answer: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Map [SOURCE N] citations back to
    retrieved chunks.
    """

    citations = get_cited_source_numbers(
        answer
    )

    cited_sources = []

    for number in citations:

        index = (
            number - 1
        )

        if (
            0
            <= index
            < len(chunks)
        ):

            chunk = chunks[
                index
            ]

            cited_sources.append(
                {
                    "reference": number,
                    "source": chunk.get(
                        "source"
                    ),
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "text": chunk.get(
                        "text",
                        "",
                    ),
                }
            )

    return cited_sources


# ==========================================================
# VALIDATE CLAIM SUPPORT
# ==========================================================

def validate_claim_support(
    answer: str,
    chunks: list[dict],
    threshold: float | None = None,
) -> dict:
    """
    Check whether answer claims are supported
    by the cited retrieved context.

    This is a lightweight local grounding check.
    """

    if threshold is None:

        threshold = (
            RAG_CLAIM_SUPPORT_THRESHOLD
        )

    claims = split_claims(
        answer
    )

    cited_sources = get_cited_source_texts(
        answer=answer,
        chunks=chunks,
    )

    if not cited_sources:

        return {
            "valid": False,
            "claim_results": [],
            "unsupported_claims": claims,
            "reason": (
                "No valid cited source text "
                "was available."
            ),
        }

    source_texts = [
        item.get(
            "text",
            "",
        )
        for item in cited_sources
    ]

    claim_results = []
    unsupported_claims = []

    for claim in claims:

        # Compare claim against every cited source.
        scores = [
            calculate_claim_support(
                claim=claim,
                source_text=source_text,
            )
            for source_text in source_texts
        ]

        best_score = (
            max(scores)
            if scores
            else 0.0
        )

        supported = (
            best_score
            >= threshold
        )

        claim_result = {
            "claim": claim,
            "support_score": best_score,
            "supported": supported,
        }

        claim_results.append(
            claim_result
        )

        if not supported:

            unsupported_claims.append(
                claim
            )

    valid = (
        len(unsupported_claims)
        == 0
    )

    logger.info(
        "Claim grounding validation: "
        "%s/%s claims supported",
        (
            len(claim_results)
            - len(unsupported_claims)
        ),
        len(claim_results),
    )

    return {
        "valid": valid,
        "threshold": threshold,
        "claim_results": claim_results,
        "unsupported_claims": unsupported_claims,
        "reason": (
            None
            if valid
            else (
                "One or more generated claims "
                "were not sufficiently supported "
                "by the cited document context."
            )
        ),
    }


# ==========================================================
# COMPLETE GROUNDING VALIDATION
# ==========================================================

def validate_grounded_answer(
    answer: str,
    chunks: list[dict],
) -> dict:
    """
    Complete citation + claim grounding validation.

    Checks:

    1. Citation exists
    2. Citation number is valid
    3. Claims are supported by cited context
    """

    citation_check = (
        validate_citation_numbers(
            answer=answer,
            source_count=len(
                chunks
            ),
        )
    )

    if not citation_check[
        "valid"
    ]:

        return {
            "grounded": False,
            "citation_check": citation_check,
            "claim_check": None,
        }

    claim_check = (
        validate_claim_support(
            answer=answer,
            chunks=chunks,
        )
    )

    return {
        "grounded": (
            citation_check[
                "valid"
            ]
            and claim_check[
                "valid"
            ]
        ),
        "citation_check": citation_check,
        "claim_check": claim_check,
    }


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    test_chunks = [
        {
            "source": "claim_guide.txt",
            "chunk_id": "claim_guide_c1",
            "text": """
The following documents may be required
for claim processing:

1. Completed claim form
2. Copy of the insurance policy
3. Identity proof of the policyholder
4. Incident report or supporting evidence
5. Bills, invoices, or repair estimates where applicable
6. Bank account details for settlement
""",
        }
    ]

    good_answer = """
The following documents may be required for claim processing:
1. Completed claim form
2. Copy of the insurance policy
3. Identity proof of the policyholder
4. Incident report or supporting evidence
5. Bills, invoices, or repair estimates where applicable
6. Bank account details for settlement [SOURCE 1]
"""

    bad_answer = """
The insurer guarantees every claim will be paid
within 24 hours [SOURCE 1].
"""

    print(
        "\nGOOD ANSWER TEST"
    )

    print(
        validate_grounded_answer(
            answer=good_answer,
            chunks=test_chunks,
        )
    )

    print(
        "\nBAD ANSWER TEST"
    )

    print(
        validate_grounded_answer(
            answer=bad_answer,
            chunks=test_chunks,
        )
    )