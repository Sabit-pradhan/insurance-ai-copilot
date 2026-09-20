# src/agents/rag_agent.py

from src.utils.llm import llm
from src.rag.vector_store import semantic_search
from src.core.config import settings
from src.core.logger import get_logger


# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = get_logger(__name__)


# --------------------------------------------------
# RAG Agent
# --------------------------------------------------

def ask_rag_agent(
    question: str
) -> dict:
    """
    Answer insurance document questions using
    semantic retrieval + LLM grounded generation.
    """

    if not question or not question.strip():

        return {
            "status": "error",
            "agent": "rag",
            "answer": "Question cannot be empty.",
            "sources": []
        }

    logger.info(
        f"RAG question received: {question}"
    )

    try:

        # --------------------------------------------------
        # Step 1: Semantic Search
        # --------------------------------------------------

        chunks = semantic_search(
            question=question,
            top_k=settings.RAG_TOP_K
        )

        if not chunks:

            return {
                "status": "success",
                "agent": "rag",
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the available insurance documents."
                ),
                "sources": []
            }


        # --------------------------------------------------
        # Step 2: Build LLM Context
        # --------------------------------------------------

        context_parts = []

        for item in chunks:

            source = item["source"]
            page = item.get("page")
            chunk = item["chunk"]
            text = item["text"]

            location = (
                f"Page {page}"
                if page is not None
                else f"Chunk {chunk}"
            )

            context_parts.append(
                f"""
SOURCE: {source}
LOCATION: {location}

{text}
"""
            )


        context = "\n\n---\n\n".join(
            context_parts
        )


        # --------------------------------------------------
        # Step 3: Grounded Prompt
        # --------------------------------------------------

        prompt = f"""
You are an Insurance Document Assistant.

Answer the user's question ONLY using the provided
document context.

Rules:

1. Do not use outside knowledge.
2. Do not invent policy conditions.
3. If the answer is not available in the context,
   clearly say that the information is not available
   in the provided documents.
4. Keep the answer clear and concise.
5. Do not treat demo documents as official policy documents.
6. Mention the source document when useful.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


        # --------------------------------------------------
        # Step 4: LLM Answer
        # --------------------------------------------------

        response = llm.invoke(
            prompt
        )

        answer = response.content.strip()


        # --------------------------------------------------
        # Step 5: Build Citations
        # --------------------------------------------------

        sources = []

        for item in chunks:

            source_info = {
                "source": item["source"],
                "chunk": item["chunk"],
                "score": round(
                    item["score"],
                    4
                )
            }

            if item.get("page") is not None:

                source_info["page"] = (
                    item["page"]
                )

            sources.append(
                source_info
            )


        logger.info(
            f"RAG answer generated using "
            f"{len(chunks)} chunk(s)"
        )


        # --------------------------------------------------
        # Final Response
        # --------------------------------------------------

        return {
            "status": "success",
            "agent": "rag",
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(chunks)
        }


    except Exception as error:

        logger.exception(
            "RAG Agent failed"
        )

        return {
            "status": "error",
            "agent": "rag",
            "question": question,
            "answer": (
                "The document retrieval service "
                "encountered an error."
            ),
            "error": str(error),
            "sources": []
        }