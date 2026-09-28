# src/agents/rag_agent.py

from src.rag.rag_service import (
    answer_rag_question,
)

from src.core.logger import (
    get_logger,
)


logger = get_logger(__name__)


# ==========================================================
# RAG AGENT
# ==========================================================

def ask_rag_agent(
    question: str,
) -> dict:
    """
    RAG Agent entry point.

    Retrieval and grounded generation are handled
    by src.rag.rag_service.
    """

    if (
        not question
        or not question.strip()
    ):

        return {
            "status": "error",
            "agent": "rag",
            "answer": (
                "Question cannot be empty."
            ),
            "sources": [],
        }


    logger.info(
        f"RAG Agent received: "
        f"{question}"
    )


    try:

        result = answer_rag_question(
            question
        )


        result[
            "agent"
        ] = "rag"


        return result


    except Exception as error:

        logger.exception(
            "RAG Agent failed"
        )


        return {
            "status": "error",
            "agent": "rag",
            "question": question,
            "answer": (
                "The document intelligence service "
                "encountered an error."
            ),
            "error": str(error),
            "sources": [],
        }