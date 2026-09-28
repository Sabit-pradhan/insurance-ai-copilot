# src/utils/llm.py

from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.core.logger import get_logger


# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = get_logger(__name__)


# --------------------------------------------------
# Validate configuration
# --------------------------------------------------

if not settings.OPENROUTER_API_KEY:

    raise ValueError(
        "OPENROUTER_API_KEY is missing from environment variables."
    )


# --------------------------------------------------
# Shared Production LLM Client
# --------------------------------------------------

logger.info(
    f"Initializing LLM model: {settings.OPENROUTER_MODEL}"
)


llm = ChatOpenAI(

    # NVIDIA Nemotron through OpenRouter
    model=settings.OPENROUTER_MODEL,

    api_key=settings.OPENROUTER_API_KEY,

    base_url=settings.OPENROUTER_BASE_URL,

    # Deterministic output
    temperature=settings.LLM_TEMPERATURE,

    # Do not wait forever if provider has a problem
    timeout=settings.LLM_TIMEOUT_SECONDS,

    # Automatically retry temporary API failures
    max_retries=2
)


logger.info(
    "LLM client initialized successfully"
)