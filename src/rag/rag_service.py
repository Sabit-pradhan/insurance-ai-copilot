# src/rag/rag_service.py

from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# insurance-ai-copilot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# insurance-ai-copilot/data/documents/
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


# --------------------------------------------------
# Read one document
# --------------------------------------------------

def read_document(file_path: Path):
    """
    Read text from:
    PDF
    TXT
    Markdown
    """

    extension = file_path.suffix.lower()

    # ------------------------------
    # PDF
    # ------------------------------

    if extension == ".pdf":

        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text:
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        return "\n".join(pages)


    # ------------------------------
    # TXT / Markdown
    # ------------------------------

    if extension in [".txt", ".md"]:

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )


    return ""


# --------------------------------------------------
# Split text into chunks
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
):
    """
    Split large text into smaller overlapping chunks.

    Example:
    chunk 1 = characters 0-1200
    chunk 2 = characters 1000-2200

    200 characters overlap helps preserve context.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Load all document chunks
# --------------------------------------------------

def load_all_chunks():
    """
    Read all supported documents and convert
    them into chunks.
    """

    all_chunks = []

    # Folder does not exist
    if not DOCUMENTS_DIR.exists():
        return all_chunks


    # Read each document
    for file_path in DOCUMENTS_DIR.iterdir():

        # Only supported formats
        if file_path.suffix.lower() not in [
            ".pdf",
            ".txt",
            ".md"
        ]:
            continue


        # Extract document text
        text = read_document(file_path)

        if not text.strip():
            continue


        # Split document
        chunks = chunk_text(text)


        # Store metadata with each chunk
        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "source": file_path.name,
                    "chunk_number": chunk_number,
                    "text": chunk
                }
            )

    return all_chunks


# --------------------------------------------------
# Retrieve relevant document chunks
# --------------------------------------------------

def retrieve_documents(
    question: str,
    top_k: int = 4
):
    """
    Search document chunks using:

    TF-IDF
        +
    Cosine Similarity
    """

    chunks = load_all_chunks()


    # No documents available
    if not chunks:
        return []


    # Extract chunk text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # --------------------------------------------------
    # Create TF-IDF vectors
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(
        texts
    )


    # Convert question into same vector space
    question_vector = vectorizer.transform(
        [question]
    )


    # --------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------

    similarity_scores = cosine_similarity(
        question_vector,
        document_vectors
    )[0]


    # Highest score first
    ranked_indexes = similarity_scores.argsort()[::-1]


    # --------------------------------------------------
    # Prepare final results
    # --------------------------------------------------

    results = []

    for index in ranked_indexes[:top_k]:

        score = float(
            similarity_scores[index]
        )

        # Ignore irrelevant chunks
        if score <= 0:
            continue


        chunk = chunks[index].copy()

        chunk["score"] = round(
            score,
            4
        )

        results.append(chunk)


    return results


# --------------------------------------------------
# Quick Test
# --------------------------------------------------

if __name__ == "__main__":

    print("\nDocuments folder:")
    print(DOCUMENTS_DIR)

    chunks = load_all_chunks()

    print("\nTotal chunks:")
    print(len(chunks))

    question = (
        "What documents are required "
        "for an insurance claim?"
    )

    results = retrieve_documents(
        question=question
    )

    print("\nRetrieved Results:")

    for result in results:

        print("\n" + "-" * 60)

        print(
            "Source:",
            result["source"]
        )

        print(
            "Chunk:",
            result["chunk_number"]
        )

        print(
            "Score:",
            result["score"]
        )

        print(
            result["text"][:500]
        )