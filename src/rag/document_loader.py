# src/rag/document_loader.py

import re
from pathlib import Path

from pypdf import PdfReader

from src.core.config import settings
from src.core.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "documents"
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
}


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(
    text: str,
) -> str:
    """
    Basic text cleanup before chunking.
    """

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Keep paragraph breaks,
    # remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ==========================================================
# CHUNK TEXT
# ==========================================================

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Split document text into overlapping chunks.
    """

    if chunk_size is None:
        chunk_size = getattr(
            settings,
            "RAG_CHUNK_SIZE",
            1200,
        )

    if overlap is None:
        overlap = getattr(
            settings,
            "RAG_CHUNK_OVERLAP",
            200,
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    text = clean_text(
        text
    )

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(
                chunk
            )

        start += (
            chunk_size
            - overlap
        )

    return chunks


# ==========================================================
# LOAD PDF
# ==========================================================

def load_pdf(
    file_path: Path,
) -> list[dict]:
    """
    Extract PDF page-by-page so page metadata
    is preserved for citations.
    """

    reader = PdfReader(
        str(file_path)
    )

    records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        page_text = (
            page.extract_text()
            or ""
        )

        page_text = clean_text(
            page_text
        )

        if not page_text:
            continue

        chunks = chunk_text(
            page_text
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk_id = (
                f"{file_path.stem}"
                f"_p{page_number}"
                f"_c{chunk_number}"
            )

            records.append(
                {
                    "chunk_id": chunk_id,
                    "source": file_path.name,
                    "document_name": file_path.stem,
                    "document_type": "pdf",
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                }
            )

    return records


# ==========================================================
# LOAD TEXT / MARKDOWN
# ==========================================================

def load_text_file(
    file_path: Path,
) -> list[dict]:
    """
    Load TXT or Markdown document.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    text = clean_text(
        text
    )

    chunks = chunk_text(
        text
    )

    records = []

    for chunk_number, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_id = (
            f"{file_path.stem}"
            f"_c{chunk_number}"
        )

        records.append(
            {
                "chunk_id": chunk_id,
                "source": file_path.name,
                "document_name": file_path.stem,
                "document_type": (
                    file_path.suffix
                    .lower()
                    .replace(".", "")
                ),
                "page": None,
                "chunk": chunk_number,
                "text": chunk,
            }
        )

    return records


# ==========================================================
# LOAD ONE DOCUMENT
# ==========================================================

def load_document(
    file_path: Path,
) -> list[dict]:

    suffix = (
        file_path
        .suffix
        .lower()
    )

    if suffix == ".pdf":

        return load_pdf(
            file_path
        )

    if suffix in {
        ".txt",
        ".md",
    }:

        return load_text_file(
            file_path
        )

    return []


# ==========================================================
# LOAD ALL DOCUMENTS
# ==========================================================

def load_documents() -> list[dict]:
    """
    Load every supported insurance document.
    """

    if not DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Documents directory not found: "
            f"{DOCUMENTS_DIR}"
        )

    records = []

    files = sorted(
        DOCUMENTS_DIR.iterdir()
    )

    for file_path in files:

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        logger.info(
            f"Loading document: "
            f"{file_path.name}"
        )

        file_records = load_document(
            file_path
        )

        records.extend(
            file_records
        )

    # ------------------------------------------------------
    # Remove exact duplicate chunks
    # ------------------------------------------------------

    unique_records = []

    seen_texts = set()

    for record in records:

        normalized = (
            record["text"]
            .strip()
            .lower()
        )

        if normalized in seen_texts:
            continue

        seen_texts.add(
            normalized
        )

        unique_records.append(
            record
        )

    logger.info(
        f"Loaded {len(unique_records)} "
        f"unique document chunks"
    )

    return unique_records


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    documents = load_documents()

    print(
        "\nTotal chunks:",
        len(documents),
    )

    for item in documents[:5]:

        print(
            "\n",
            "-" * 60,
        )

        print(
            "Chunk ID:",
            item["chunk_id"],
        )

        print(
            "Source:",
            item["source"],
        )

        print(
            "Page:",
            item["page"],
        )

        print(
            "Chunk:",
            item["chunk"],
        )

        print(
            item["text"][:300]
        )