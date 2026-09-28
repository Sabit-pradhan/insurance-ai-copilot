# src/rag/rag_service.py

import re

from src.utils.llm import llm

from src.rag.retriever import (
    retrieve_context,
)

from src.rag.citation_service import (
    validate_grounded_answer,
)

from src.core.config import settings
from src.core.logger import get_logger


# ==========================================================
# LOGGER
# ==========================================================

logger = get_logger(__name__)


# ==========================================================
# CONSTANTS
# ==========================================================

NO_CONTEXT_ANSWER = (
    "I could not find sufficient supporting "
    "information in the available insurance documents."
)


# ==========================================================
# CLEAN MODEL OUTPUT
# ==========================================================

def clean_llm_response(
    text: str,
) -> str:
    """
    Remove accidental thinking tags from model output.
    """

    if not text:
        return ""

    # Remove complete thinking blocks
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove stray think tags
    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ==========================================================
# REMOVE COMMON REASONING PREFIXES
# ==========================================================

def remove_reasoning_prefixes(
    text: str,
) -> str:
    """
    Extra protection against reasoning-style
    content accidentally appearing to users.
    """

    if not text:
        return ""

    patterns = [
        r"^\s*We must be cautious.*?(?:\n\n|$)",
        r"^\s*We need to.*?(?:\n\n|$)",
        r"^\s*I need to.*?(?:\n\n|$)",
        r"^\s*The user is asking.*?(?:\n\n|$)",
        r"^\s*The user asked.*?(?:\n\n|$)",
        r"^\s*Let's analyze.*?(?:\n\n|$)",
        r"^\s*We should.*?(?:\n\n|$)",
        r"^\s*Need to.*?(?:\n\n|$)",
        r"^\s*\*?Final check\*?.*?(?:\n\n|$)",
    ]

    cleaned = text

    for pattern in patterns:

        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    return cleaned


# ==========================================================
# EXTRACT FINAL ANSWER
# ==========================================================

def extract_final_answer(
    text: str,
) -> str:
    """
    Extract only the user-facing final answer.

    Supported formats:

    <FINAL_ANSWER>
    ...
    </FINAL_ANSWER>

    FINAL ANSWER: ...

    FINAL: ...

    Or a citation-grounded response containing
    [SOURCE N].
    """

    if not text:
        return ""

    cleaned_text = clean_llm_response(
        text
    )

    if not cleaned_text:
        return ""

    # ------------------------------------------------------
    # FORMAT 1
    # <FINAL_ANSWER>...</FINAL_ANSWER>
    # ------------------------------------------------------

    match = re.search(
        r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .strip()
        )

    # ------------------------------------------------------
    # FORMAT 2
    # FINAL ANSWER: ...
    # ------------------------------------------------------

    match = re.search(
        r"FINAL\s+ANSWER\s*:\s*(.*)",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .strip()
        )

    # ------------------------------------------------------
    # FORMAT 3
    # FINAL: ...
    # ------------------------------------------------------

    match = re.search(
        r"(?:^|\n)\s*FINAL\s*:\s*(.*)",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .strip()
        )

    # ------------------------------------------------------
    # Remove possible reasoning prefixes
    # ------------------------------------------------------

    cleaned_text = remove_reasoning_prefixes(
        cleaned_text
    )

    # ------------------------------------------------------
    # Citation-grounded fallback
    # ------------------------------------------------------

    if re.search(
        r"\[SOURCE\s+\d+\]",
        cleaned_text,
        flags=re.IGNORECASE,
    ):

        return cleaned_text.strip()

    logger.warning(
        "Could not reliably extract "
        "a grounded final answer."
    )

    return ""


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(
    chunks: list[dict],
) -> str:
    """
    Convert retrieved chunks into numbered
    source-aware context for the LLM.
    """

    context_parts = []

    for number, item in enumerate(
        chunks,
        start=1,
    ):

        source = item.get(
            "source",
            "Unknown document",
        )

        page = item.get(
            "page"
        )

        chunk_number = item.get(
            "chunk"
        )

        semantic_score = item.get(
            "semantic_score",
            item.get(
                "score"
            ),
        )

        rerank_score = item.get(
            "rerank_score"
        )

        text = item.get(
            "text",
            "",
        )

        # --------------------------------------------------
        # LOCATION
        # --------------------------------------------------

        if page is not None:

            location = (
                f"Page {page}, "
                f"Chunk {chunk_number}"
            )

        else:

            location = (
                f"Chunk {chunk_number}"
            )

        # --------------------------------------------------
        # OPTIONAL RERANK INFORMATION
        # --------------------------------------------------

        rerank_line = ""

        if rerank_score is not None:

            rerank_line = (
                f"\nRerank Score: "
                f"{rerank_score}"
            )

        # --------------------------------------------------
        # SOURCE BLOCK
        # --------------------------------------------------

        context_part = f"""
[SOURCE {number}]

Document: {source}
Location: {location}
Semantic Score: {semantic_score}{rerank_line}

Content:
{text}
""".strip()

        context_parts.append(
            context_part
        )

    separator = (
        "\n\n"
        "----------------------------------------"
        "\n\n"
    )

    return separator.join(
        context_parts
    )


# ==========================================================
# BUILD STRUCTURED SOURCES
# ==========================================================

def build_sources(
    chunks: list[dict],
) -> list[dict]:
    """
    Build structured source metadata
    for API / Streamlit responses.
    """

    sources = []

    for number, item in enumerate(
        chunks,
        start=1,
    ):

        source_info = {
            "reference": number,

            "source": item.get(
                "source"
            ),

            "chunk": item.get(
                "chunk"
            ),

            "score": item.get(
                "score"
            ),
        }

        # --------------------------------------------------
        # SEMANTIC SCORE
        # --------------------------------------------------

        if item.get(
            "semantic_score"
        ) is not None:

            source_info[
                "semantic_score"
            ] = item[
                "semantic_score"
            ]

        # --------------------------------------------------
        # LEXICAL SCORE
        # --------------------------------------------------

        if item.get(
            "lexical_score"
        ) is not None:

            source_info[
                "lexical_score"
            ] = item[
                "lexical_score"
            ]

        # --------------------------------------------------
        # RERANK SCORE
        # --------------------------------------------------

        if item.get(
            "rerank_score"
        ) is not None:

            source_info[
                "rerank_score"
            ] = item[
                "rerank_score"
            ]

        # --------------------------------------------------
        # PAGE
        # --------------------------------------------------

        if item.get(
            "page"
        ) is not None:

            source_info[
                "page"
            ] = item[
                "page"
            ]

        # --------------------------------------------------
        # CHUNK ID
        # --------------------------------------------------

        if item.get(
            "chunk_id"
        ):

            source_info[
                "chunk_id"
            ] = item[
                "chunk_id"
            ]

        # --------------------------------------------------
        # DOCUMENT NAME
        # --------------------------------------------------

        if item.get(
            "document_name"
        ):

            source_info[
                "document_name"
            ] = item[
                "document_name"
            ]

        # --------------------------------------------------
        # DOCUMENT TYPE
        # --------------------------------------------------

        if item.get(
            "document_type"
        ):

            source_info[
                "document_type"
            ] = item[
                "document_type"
            ]

        sources.append(
            source_info
        )

    return sources


# ==========================================================
# MAIN RAG SERVICE
# ==========================================================

def answer_rag_question(
    question: str,
) -> dict:
    """
    Production-oriented document-grounded
    RAG service.

    Flow:

    User Question
        ↓
    Semantic Candidate Retrieval
        ↓
    Absolute Score Filter
        ↓
    Relative Score Filter
        ↓
    Duplicate Removal
        ↓
    Hybrid Reranking
        ↓
    Source Diversity
        ↓
    Final Context
        ↓
    Local LLM
        ↓
    Final Answer Extraction
        ↓
    Citation Validation
        ↓
    Claim Support Validation
        ↓
    Grounded Response
    """

    # ======================================================
    # STEP 0 — VALIDATE QUESTION
    # ======================================================

    if (
        not question
        or not question.strip()
    ):

        return {
            "status": "error",
            "question": question,
            "answer": (
                "Question cannot be empty."
            ),
            "sources": [],
            "retrieved_chunks": 0,
            "grounded": False,
            "grounding_validation": None,
        }

    question = question.strip()

    logger.info(
        "RAG request received: %s",
        question,
    )

    try:

        # ==================================================
        # STEP 1 — RETRIEVAL CONFIGURATION
        # ==================================================

        min_score = getattr(
            settings,
            "RAG_MIN_SCORE",
            0.45,
        )

        candidate_k = getattr(
            settings,
            "RAG_CANDIDATE_K",
            12,
        )

        final_k = getattr(
            settings,
            "RAG_FINAL_K",
            4,
        )

        # ==================================================
        # STEP 2 — ENHANCED RETRIEVAL
        # ==================================================

        chunks = retrieve_context(
            question=question,
            candidate_k=candidate_k,
            final_k=final_k,
            min_score=min_score,
        )

        # ==================================================
        # STEP 3 — NO RELEVANT CONTEXT
        # ==================================================

        if not chunks:

            logger.info(
                "No relevant RAG context found."
            )

            return {
                "status": "success",
                "question": question,
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "retrieved_chunks": 0,
                "grounded": False,
                "grounding_validation": None,
            }

        logger.info(
            "Enhanced retrieval returned %s "
            "final chunk(s)",
            len(chunks),
        )

        # ==================================================
        # STEP 4 — BUILD FINAL CONTEXT
        # ==================================================

        context = build_context(
            chunks
        )

        # ==================================================
        # STEP 5 — GROUNDED PROMPT
        # ==================================================

        prompt = f"""
You are the Document Intelligence component of an
Insurance AI Decision Intelligence Platform.

Answer the USER QUESTION using ONLY information
contained in DOCUMENT CONTEXT.

STRICT GROUNDING RULES:

1. Use ONLY DOCUMENT CONTEXT.

2. Do NOT use outside knowledge.

3. Do NOT invent or assume:
   - policy conditions
   - coverage
   - claim requirements
   - exclusions
   - limits
   - eligibility rules
   - timelines
   - benefits
   - settlement rules
   - regulatory requirements

4. Every factual statement must be supported by
   DOCUMENT CONTEXT.

5. Cite supporting information using exactly:

   [SOURCE 1]
   [SOURCE 2]

6. Cite ONLY source numbers that exist in
   DOCUMENT CONTEXT.

7. Put citations directly after the information
   supported by that source.

8. If one source fully supports the answer,
   do not add unnecessary citations.

9. If DOCUMENT CONTEXT does not contain enough
   information to answer the question, answer exactly:

   {NO_CONTEXT_ANSWER}

10. Never fill missing information using
    general insurance knowledge.

11. Demo, sample, or testing documents must not
    be presented as official policy wording.

12. If the retrieved document explicitly states
    that it is a demo or testing document,
    mention that limitation when relevant.

13. If retrieved sources conflict, clearly state
    that the supplied documents contain differing
    information.

14. Keep the response concise, clear,
    and business-friendly.

15. Do NOT output:
    - reasoning
    - analysis
    - planning
    - self-reflection
    - hidden thoughts

16. Preferred response format:

<FINAL_ANSWER>
Your final grounded answer with [SOURCE N] citations.
</FINAL_ANSWER>

17. If XML tags are not used, return:

FINAL ANSWER: Your final grounded answer with
[SOURCE N] citations.


DOCUMENT CONTEXT
================

{context}


USER QUESTION
=============

{question}
"""

        # ==================================================
        # STEP 6 — LLM GENERATION
        # ==================================================

        logger.info(
            "Generating grounded answer using "
            "%s final chunk(s)",
            len(chunks),
        )

        response = llm.invoke(
            prompt
        )

        if response:

            raw_answer = (
                response.content
                or ""
            )

        else:

            raw_answer = ""

        if not isinstance(
            raw_answer,
            str,
        ):

            raw_answer = str(
                raw_answer
            )

        # ==================================================
        # STEP 7 — EXTRACT FINAL ANSWER
        # ==================================================

        answer = extract_final_answer(
            raw_answer
        )

        # ==================================================
        # STEP 8 — INVALID MODEL OUTPUT
        # ==================================================

        if not answer:

            logger.warning(
                "No valid grounded final answer "
                "could be extracted."
            )

            return {
                "status": "success",
                "question": question,
                "answer": NO_CONTEXT_ANSWER,
                "sources": build_sources(
                    chunks
                ),
                "retrieved_chunks": len(
                    chunks
                ),
                "grounded": False,
                "grounding_validation": None,
            }

        # ==================================================
        # STEP 9 — MODEL SAYS CONTEXT INSUFFICIENT
        # ==================================================

        if (
            NO_CONTEXT_ANSWER.lower()
            in answer.lower()
        ):

            logger.info(
                "Retrieved context was insufficient "
                "for a grounded answer."
            )

            return {
                "status": "success",
                "question": question,
                "answer": NO_CONTEXT_ANSWER,
                "sources": build_sources(
                    chunks
                ),
                "retrieved_chunks": len(
                    chunks
                ),
                "grounded": False,
                "grounding_validation": None,
            }

        # ==================================================
        # STEP 10 — COMPLETE GROUNDING VALIDATION
        # ==================================================

        grounding_result = (
            validate_grounded_answer(
                answer=answer,
                chunks=chunks,
            )
        )

        # ==================================================
        # STEP 11 — REJECT UNSUPPORTED ANSWER
        # ==================================================

        if not grounding_result.get(
            "grounded",
            False,
        ):

            logger.warning(
                "Generated answer failed "
                "grounding validation."
            )

            return {
                "status": "success",
                "question": question,
                "answer": NO_CONTEXT_ANSWER,
                "sources": build_sources(
                    chunks
                ),
                "retrieved_chunks": len(
                    chunks
                ),
                "grounded": False,
                "grounding_validation": (
                    grounding_result
                ),
            }

        # ==================================================
        # STEP 12 — BUILD SOURCE METADATA
        # ==================================================

        sources = build_sources(
            chunks
        )

        logger.info(
            "RAG answer generated successfully "
            "using %s final chunk(s)",
            len(chunks),
        )

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(
                chunks
            ),
            "grounded": True,
            "grounding_validation": (
                grounding_result
            ),
        }

    # ======================================================
    # ERROR HANDLING
    # ======================================================

    except Exception as error:

        logger.exception(
            "RAG service failed"
        )

        return {
            "status": "error",
            "question": question,
            "answer": (
                "The document intelligence service "
                "encountered an error."
            ),
            "error": str(
                error
            ),
            "sources": [],
            "retrieved_chunks": 0,
            "grounded": False,
            "grounding_validation": None,
        }


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    test_question = (
        "What documents are required "
        "for an insurance claim?"
    )

    result = answer_rag_question(
        test_question
    )

    print(
        "\nQUESTION:"
    )

    print(
        result.get(
            "question"
        )
    )

    print(
        "\nANSWER:"
    )

    print(
        result.get(
            "answer"
        )
    )

    print(
        "\nSOURCES:"
    )

    for source in result.get(
        "sources",
        [],
    ):

        print(
            source
        )

    print(
        "\nRETRIEVED CHUNKS:"
    )

    print(
        result.get(
            "retrieved_chunks"
        )
    )

    print(
        "\nGROUNDED:"
    )

    print(
        result.get(
            "grounded"
        )
    )

    print(
        "\nGROUNDING VALIDATION:"
    )

    print(
        result.get(
            "grounding_validation"
        )
    )