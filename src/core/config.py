# src/core/config.py

import os

from dotenv import load_dotenv


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


class Settings:
    """
    Central application configuration.

    All important project settings are managed
    from one place.
    """

    # --------------------------------------------------
    # Application
    # --------------------------------------------------

    APP_NAME = "Insurance AI Copilot"
    APP_VERSION = "1.0.0"

    ENVIRONMENT = os.getenv(
        "ENVIRONMENT",
        "development"
    )


    # --------------------------------------------------
    # PostgreSQL
    # --------------------------------------------------

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    DB_HOST = os.getenv(
        "DB_HOST",
        "localhost"
    )

    DB_PORT = int(
        os.getenv(
            "DB_PORT",
            "5432"
        )
    )

    DB_NAME = os.getenv("DB_NAME")


    # --------------------------------------------------
    # OpenRouter / LLM
    # --------------------------------------------------

    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY"
    )

    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL",
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    )

    OPENROUTER_BASE_URL = (
        "https://openrouter.ai/api/v1"
    )


    # --------------------------------------------------
    # SQL Agent Configuration
    # --------------------------------------------------

    # Maximum rows returned by SQL Agent
    SQL_MAX_ROWS = 100

    # Stop long-running SQL after 10 seconds
    SQL_STATEMENT_TIMEOUT_MS = 10_000


    # --------------------------------------------------
    # RAG Configuration
    # --------------------------------------------------

    # Maximum number of chunks retrieved
    RAG_TOP_K = 4

    # Minimum semantic similarity score.
    # Chunks below this value will be ignored.
    RAG_MIN_SCORE = 0.45

    # Embedding model used for semantic search
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "liquid/lfm-2.5-embedding-350m:free"
    )

    # Size of each document chunk
    RAG_CHUNK_SIZE = 1000

    # Overlap between consecutive chunks
    RAG_CHUNK_OVERLAP = 150

    # Location where semantic index will be stored
    RAG_INDEX_DIR = (
        "artifacts/rag_index"
    )


    # --------------------------------------------------
    # LLM Configuration
    # --------------------------------------------------

    # Deterministic responses
    LLM_TEMPERATURE = 0

    # Maximum wait time for LLM request
    LLM_TIMEOUT_SECONDS = 60


# --------------------------------------------------
# Single global settings object
# --------------------------------------------------

settings = Settings()