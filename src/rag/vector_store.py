# src/rag/vector_store.py

import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.embedding_service import embed_texts, embed_query


logger = get_logger(__name__)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "documents"
)

INDEX_DIR = (
    PROJECT_ROOT
    / settings.RAG_INDEX_DIR
)

EMBEDDINGS_FILE = (
    INDEX_DIR
    / "embeddings.npy"
)

METADATA_FILE = (
    INDEX_DIR
    / "metadata.json"
)


# --------------------------------------------------
# Text Chunking
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> list[str]:

    text = text.strip()

    if not text:
        return []

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
# Load PDF
# --------------------------------------------------

def load_pdf(
    file_path: Path
) -> list[dict]:

    reader = PdfReader(
        str(file_path)
    )

    records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        chunks = chunk_text(
            text=text,
            chunk_size=settings.RAG_CHUNK_SIZE,
            overlap=settings.RAG_CHUNK_OVERLAP
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            records.append(
                {
                    "source": file_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk
                }
            )

    return records


# --------------------------------------------------
# Load TXT / MD
# --------------------------------------------------

def load_text_file(
    file_path: Path
) -> list[dict]:

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    chunks = chunk_text(
        text=text,
        chunk_size=settings.RAG_CHUNK_SIZE,
        overlap=settings.RAG_CHUNK_OVERLAP
    )

    records = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        records.append(
            {
                "source": file_path.name,
                "page": None,
                "chunk": chunk_number,
                "text": chunk
            }
        )

    return records


# --------------------------------------------------
# Load All Documents
# --------------------------------------------------

def load_documents() -> list[dict]:

    if not DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENTS_DIR}"
        )

    records = []

    for file_path in DOCUMENTS_DIR.iterdir():

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":

            records.extend(
                load_pdf(file_path)
            )

        elif suffix in {
            ".txt",
            ".md"
        }:

            records.extend(
                load_text_file(file_path)
            )

    logger.info(
        f"Loaded {len(records)} document chunks"
    )

    return records


# --------------------------------------------------
# Build Persistent Vector Index
# --------------------------------------------------

def build_vector_index():

    records = load_documents()

    if not records:

        raise ValueError(
            "No document chunks found."
        )

    texts = [
        record["text"]
        for record in records
    ]

    logger.info(
        "Building semantic vector index..."
    )

    embeddings = embed_texts(
        texts
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )

    logger.info(
        f"Vector index saved to: {INDEX_DIR}"
    )

    print("\nSemantic Index Created")
    print(f"Chunks: {len(records)}")
    print(
        f"Embedding shape: {embeddings.shape}"
    )


# --------------------------------------------------
# Load Existing Vector Index
# --------------------------------------------------

def load_vector_index():

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            "Embedding index not found. "
            "Run build_vector_index() first."
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "Metadata file not found."
        )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return embeddings, metadata


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------

def semantic_search(
    question: str,
    top_k: int | None = None
) -> list[dict]:

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if top_k is None:
        top_k = settings.RAG_TOP_K

    embeddings, metadata = (
        load_vector_index()
    )

    query_vector = embed_query(
        question
    )

    # Vectors are normalized,
    # so dot product = cosine similarity
    scores = embeddings @ query_vector

    best_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in best_indices:

        score = float(
            scores[index]
        )

        # Ignore weak chunks
        if score < settings.RAG_MIN_SCORE:
            continue

        record = metadata[
            int(index)
        ].copy()

        record["score"] = score

        results.append(
            record
        )

    logger.info(
        f"Semantic search returned "
        f"{len(results)} chunk(s) "
        f"above threshold "
        f"{settings.RAG_MIN_SCORE}"
    )

    return results


# --------------------------------------------------
# Run Index Builder
# --------------------------------------------------

if __name__ == "__main__":

    build_vector_index()