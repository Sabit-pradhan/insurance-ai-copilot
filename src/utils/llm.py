# src/utils/llm.py

import os
import re
import logging

from dotenv import load_dotenv
from ollama import Client
from langchain_core.messages import AIMessage


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()


# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# OLLAMA CONFIG
# ==========================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b",
)


# ==========================================================
# RESPONSE CLEANING
# ==========================================================

def clean_model_output(
    text: str,
) -> str:
    """
    Remove Qwen thinking output and stray thinking tags.
    """

    if not text:
        return ""

    # Remove full thinking blocks
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove stray opening/closing tags
    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# ==========================================================
# LOCAL OLLAMA CLIENT
# ==========================================================

client = Client(
    host=OLLAMA_BASE_URL
)


# ==========================================================
# LLM WRAPPER
# ==========================================================

class LocalOllamaLLM:
    """
    Small compatibility wrapper.

    Existing code can continue using:

        llm.invoke(prompt).content

    Internally we use Ollama directly with:

        think=False
    """

    def __init__(
        self,
        model: str,
    ):

        self.model = model

        logger.info(
            "Initializing native Ollama model: %s",
            self.model,
        )

    def invoke(
        self,
        prompt,
        **kwargs,
    ) -> AIMessage:
        """
        Send prompt to local Ollama.

        Returns a LangChain AIMessage so existing
        agents can continue using response.content.
        """

        # --------------------------------------------------
        # Convert prompt to string
        # --------------------------------------------------

        if isinstance(
            prompt,
            str,
        ):

            prompt_text = prompt

        else:

            prompt_text = str(
                prompt
            )

        # --------------------------------------------------
        # Native Ollama call
        # --------------------------------------------------

        response = client.chat(
            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],

            # Disable Qwen reasoning mode
            think=False,

            # Keep model loaded between requests
            keep_alive="10m",

            options={
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 512,
            },
        )

        # --------------------------------------------------
        # Extract content
        # --------------------------------------------------

        raw_content = (
            response.message.content
            or ""
        )

        # --------------------------------------------------
        # Final global safety cleanup
        # --------------------------------------------------

        clean_content = clean_model_output(
            raw_content
        )

        return AIMessage(
            content=clean_content
        )


# ==========================================================
# SHARED LLM OBJECT
# ==========================================================

llm = LocalOllamaLLM(
    model=OLLAMA_MODEL
)


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    response = llm.invoke(
        "Answer in one sentence: What is insurance?"
    )

    print(
        "\nMODEL:"
    )

    print(
        OLLAMA_MODEL
    )

    print(
        "\nRESPONSE:"
    )

    print(
        response.content
    )