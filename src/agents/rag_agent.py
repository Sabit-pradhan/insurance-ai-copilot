# src/agents/rag_agent.py

from src.utils.llm import llm
from src.rag.rag_service import retrieve_documents


# --------------------------------------------------
# RAG Agent
# --------------------------------------------------

def ask_rag_agent(question: str):
    """
    Answer insurance document questions
    using retrieved document context only.
    """

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_documents(
        question=question,
        top_k=4
    )


    # --------------------------------------------------
    # No relevant document found
    # --------------------------------------------------

    if not retrieved_chunks:

        return {
            "question": question,
            "answer": (
                "I could not find relevant information "
                "in the available insurance documents."
            ),
            "sources": []
        }


    # --------------------------------------------------
    # Build context for LLM
    # --------------------------------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['source']}
CHUNK: {chunk['chunk_number']}

CONTENT:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)


    # --------------------------------------------------
    # Grounded RAG Prompt
    # --------------------------------------------------

    prompt = f"""
You are an Insurance AI Copilot.

Answer the user's question using ONLY the
document context provided below.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

RULES:

1. Use only information from DOCUMENT CONTEXT.
2. Do not invent insurance rules or policy details.
3. If the context does not contain enough information,
   say:
   "The available documents do not contain enough
   information to answer this question."
4. Keep the answer concise and business-friendly.
5. Mention the relevant source document name.
6. Do not use outside knowledge.
7. Return only the final answer.
"""

    # Step 2: Generate answer using NVIDIA/OpenRouter
    response = llm.invoke(prompt)

    answer = response.content.strip()


    # --------------------------------------------------
    # Collect unique source names
    # --------------------------------------------------

    sources = list(
        dict.fromkeys(
            chunk["source"]
            for chunk in retrieved_chunks
        )
    )


    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    }


# --------------------------------------------------
# Quick Test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What documents are required "
        "for an insurance claim?"
    )

    result = ask_rag_agent(question)

    print("\nRAG Agent Result:")
    print(result)