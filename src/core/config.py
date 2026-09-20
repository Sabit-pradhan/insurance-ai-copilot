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

    Instead of reading environment variables
    separately in every file, all configuration
    comes from this class.
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
    DB_HOST = os.getenv("DB_HOST", "localhost")

    DB_PORT = int(
        os.getenv("DB_PORT", "5432")
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

    # Stop SQL queries after 10 seconds
    SQL_STATEMENT_TIMEOUT_MS = 10_000


    # --------------------------------------------------
    # RAG Configuration
    # --------------------------------------------------

    # Number of document chunks retrieved
    RAG_TOP_K = 4


    # --------------------------------------------------
    # LLM Configuration
    # --------------------------------------------------

    # Deterministic responses
    LLM_TEMPERATURE = 0

    # LLM request timeout
    LLM_TIMEOUT_SECONDS = 60


# --------------------------------------------------
# Single global settings object
# --------------------------------------------------

settings = Settings()