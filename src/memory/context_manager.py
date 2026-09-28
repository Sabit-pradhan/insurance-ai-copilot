# src/memory/context_manager.py

import re

from src.core.logger import get_logger
from src.memory.session_memory import session_memory
from src.utils.llm import llm


# ==========================================================
# LOGGER
# ==========================================================

logger = get_logger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_HISTORY_LIMIT = 6


# ==========================================================
# TEXT HELPERS
# ==========================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for lightweight intent checks.
    """

    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ==========================================================
# FOLLOW-UP DETECTION
# ==========================================================

def is_follow_up_question(
    question: str,
) -> bool:
    """
    Detect whether the latest user question
    probably depends on previous context.
    """

    if not question:
        return False

    normalized = normalize_text(
        question
    )

    words = set(
        re.findall(
            r"\b[a-z]+\b",
            normalized,
        )
    )

    reference_words = {
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "there",
        "then",
    }

    if words & reference_words:
        return True

    phrase_patterns = [
        "after that",
        "before that",
        "what about",
        "how about",
        "and what",
        "and how",
        "what happens next",
        "what happens after",
        "what happens before",
        "then what",
    ]

    if any(
        phrase in normalized
        for phrase in phrase_patterns
    ):
        return True

    tokens = normalized.split()

    if (
        len(tokens) <= 5
        and normalized.startswith(
            (
                "why ",
                "how ",
                "when ",
                "where ",
                "what next",
                "then what",
            )
        )
    ):
        return True

    return False


# ==========================================================
# HISTORY
# ==========================================================

def get_conversation_context(
    session_id: str,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> str:
    """
    Return recent conversation history.
    """

    return session_memory.format_history(
        session_id=session_id,
        limit=limit,
    )


# ==========================================================
# TOPIC DETECTION
# ==========================================================

def infer_recent_topic(
    session_id: str,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> str | None:
    """
    Infer the most recent insurance topic
    from previous user messages.
    """

    messages = (
        session_memory.get_recent_messages(
            session_id=session_id,
            limit=limit,
        )
    )

    for message in reversed(
        messages
    ):

        if (
            message.get("role")
            != "user"
        ):
            continue

        text = normalize_text(
            message.get(
                "content",
                "",
            )
        )

        # ----------------------------------------------
        # FRAUD
        # ----------------------------------------------

        if (
            "fraud" in text
            or "suspicious claim" in text
        ):
            return "fraud risk review"

        # ----------------------------------------------
        # POLICY SERVICING
        # ----------------------------------------------

        if (
            "policy servicing" in text
            or "policyholder request" in text
            or "servicing" in text
        ):
            return "policy servicing"

        # ----------------------------------------------
        # UNDERWRITING
        # ----------------------------------------------

        if (
            "underwriting" in text
            or "underwriter" in text
            or "applicant" in text
        ):
            return "insurance underwriting"

        # ----------------------------------------------
        # RENEWAL
        # ----------------------------------------------

        if (
            "renewal" in text
            or "renew" in text
            or "non-renewal" in text
            or "churn" in text
        ):
            return "policy renewal"

        # ----------------------------------------------
        # CLAIM
        # ----------------------------------------------

        if "claim" in text:
            return "insurance claim"

    return None


# ==========================================================
# CLEAN LLM OUTPUT
# ==========================================================

def clean_rewritten_question(
    text: str,
) -> str:
    """
    Remove unwanted wrappers from
    LLM rewrite output.
    """

    if not text:
        return ""

    text = text.strip()

    # Remove thinking block
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    prefixes = [
        r"^\s*standalone question\s*:\s*",
        r"^\s*rewritten question\s*:\s*",
        r"^\s*question\s*:\s*",
        r"^\s*final\s*:\s*",
    ]

    for pattern in prefixes:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    text = text.strip(
        ' "\''
    )

    return text.strip()


# ==========================================================
# TOPIC CHECK
# ==========================================================

def topic_present(
    question: str,
    topic: str,
) -> bool:
    """
    Check whether rewritten question
    still contains the topic.
    """

    q = normalize_text(
        question
    )

    topic_keywords = {

        "insurance claim": [
            "claim",
        ],

        "fraud risk review": [
            "fraud",
            "suspicious",
            "investigation",
        ],

        "policy servicing": [
            "servicing",
            "policyholder",
            "policy service",
        ],

        "insurance underwriting": [
            "underwriting",
            "underwriter",
            "applicant",
        ],

        "policy renewal": [
            "renewal",
            "renew",
            "non-renewal",
            "churn",
        ],
    }

    keywords = topic_keywords.get(
        topic,
        [],
    )

    return any(
        keyword in q
        for keyword in keywords
    )


# ==========================================================
# PRESERVE TOPIC
# ==========================================================

def ensure_topic_preserved(
    question: str,
    topic: str | None,
) -> str:
    """
    Make sure the rewritten follow-up
    keeps the original insurance subject.
    """

    if (
        not question
        or not topic
    ):
        return question

    if topic_present(
        question,
        topic,
    ):
        return question

    # ======================================================
    # CLAIM-SPECIFIC REWRITE
    # ======================================================

    if topic == "insurance claim":

        updated = re.sub(
            r"\brequired\s+documents\b",
            (
                "required insurance "
                "claim documents"
            ),
            question,
            flags=re.IGNORECASE,
        )

        if updated != question:
            return updated

        updated = re.sub(
            r"\bthe\s+documents\b",
            (
                "the insurance "
                "claim documents"
            ),
            question,
            flags=re.IGNORECASE,
        )

        if updated != question:
            return updated

        updated = re.sub(
            r"\bdocuments\b",
            (
                "insurance claim documents"
            ),
            question,
            count=1,
            flags=re.IGNORECASE,
        )

        if updated != question:
            return updated

    # ======================================================
    # GENERIC SAFE FALLBACK
    # ======================================================

    base = (
        question
        .rstrip()
        .rstrip("?")
        .strip()
    )

    return (
        f"{base} in the context of {topic}?"
    )


# ==========================================================
# FOLLOW-UP RESOLUTION
# ==========================================================

def resolve_question(
    session_id: str,
    question: str,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
) -> dict:
    """
    Convert a context-dependent question
    into a standalone question.
    """

    if (
        not question
        or not question.strip()
    ):
        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    # ======================================================
    # GET PREVIOUS HISTORY
    # ======================================================

    history = (
        get_conversation_context(
            session_id=session_id,
            limit=history_limit,
        )
    )

    # ======================================================
    # FIRST TURN
    # ======================================================

    if not history:

        logger.info(
            (
                "No conversation history "
                "found for session %s. "
                "Using original question."
            ),
            session_id,
        )

        return {
            "original_question":
                question,

            "resolved_question":
                question,

            "was_rewritten":
                False,

            "follow_up_detected":
                False,

            "context_topic":
                None,
        }

    # ======================================================
    # FOLLOW-UP DETECTION
    # ======================================================

    follow_up = (
        is_follow_up_question(
            question
        )
    )

    # ======================================================
    # STANDALONE QUESTION
    # ======================================================

    if not follow_up:

        logger.info(
            (
                "Question appears "
                "standalone. "
                "Skipping context rewrite."
            )
        )

        return {
            "original_question":
                question,

            "resolved_question":
                question,

            "was_rewritten":
                False,

            "follow_up_detected":
                False,

            "context_topic":
                None,
        }

    # ======================================================
    # GET RECENT TOPIC
    # ======================================================

    context_topic = (
        infer_recent_topic(
            session_id=session_id,
            limit=history_limit,
        )
    )

    logger.info(
        (
            "Follow-up question detected "
            "for session %s"
        ),
        session_id,
    )

    # ======================================================
    # REWRITE PROMPT
    # ======================================================

    prompt = f"""
You rewrite conversational follow-up questions into
clear standalone questions for an Insurance AI Copilot.

Use the conversation history only to resolve references.

Examples of references:

- it
- them
- this
- that
- these
- those
- they
- the documents
- what happens next
- what about
- how about

RULES:

1. Do NOT answer the question.

2. Rewrite only the latest user question.

3. Preserve the user's original intent.

4. Do NOT invent facts.

5. Keep the rewritten question concise.

6. Return ONLY the rewritten standalone question.

7. Do not output reasoning or explanation.

8. Preserve the insurance topic from the conversation.

9. Replace vague references such as "them" or "it"
   with the actual subject from conversation history.

10. Never convert a specific insurance topic into
    a generic question.


EXAMPLE

Conversation:
USER: What documents are required for an insurance claim?

Latest question:
What happens after I submit them?

Correct standalone question:
What happens after I submit the required insurance claim documents?


CONVERSATION HISTORY
====================

{history}


LATEST USER QUESTION
====================

{question}
"""

    # ======================================================
    # CALL LOCAL LLM
    # ======================================================

    try:

        response = llm.invoke(
            prompt
        )

        raw_text = (
            response.content
            if response
            else ""
        )

        if not isinstance(
            raw_text,
            str,
        ):
            raw_text = str(
                raw_text
            )

        resolved_question = (
            clean_rewritten_question(
                raw_text
            )
        )

        # ==================================================
        # EMPTY OUTPUT FALLBACK
        # ==================================================

        if not resolved_question:

            logger.warning(
                (
                    "Question rewrite "
                    "returned empty output. "
                    "Using original question."
                )
            )

            resolved_question = (
                question
            )

        # ==================================================
        # FORCE TOPIC PRESERVATION
        # ==================================================

        resolved_question = (
            ensure_topic_preserved(
                question=(
                    resolved_question
                ),
                topic=context_topic,
            )
        )

        was_rewritten = (
            normalize_text(
                resolved_question
            )
            !=
            normalize_text(
                question
            )
        )

        logger.info(
            (
                "Question resolution "
                "completed. "
                "rewritten=%s topic=%s"
            ),
            was_rewritten,
            context_topic,
        )

        return {
            "original_question":
                question,

            "resolved_question":
                resolved_question,

            "was_rewritten":
                was_rewritten,

            "follow_up_detected":
                True,

            "context_topic":
                context_topic,
        }

    # ======================================================
    # SAFE FALLBACK
    # ======================================================

    except Exception as error:

        logger.exception(
            (
                "Follow-up question "
                "resolution failed."
            )
        )

        fallback_question = (
            ensure_topic_preserved(
                question=question,
                topic=context_topic,
            )
        )

        return {
            "original_question":
                question,

            "resolved_question":
                fallback_question,

            "was_rewritten":
                (
                    normalize_text(
                        fallback_question
                    )
                    !=
                    normalize_text(
                        question
                    )
                ),

            "follow_up_detected":
                True,

            "context_topic":
                context_topic,

            "error":
                str(error),
        }


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    test_session = (
        "memory_test_session"
    )

    # Clear old test data
    session_memory.clear_session(
        test_session
    )

    # Previous user question
    session_memory.add_user_message(
        session_id=test_session,
        content=(
            "What documents are required "
            "for an insurance claim?"
        ),
    )

    # Previous assistant answer
    session_memory.add_assistant_message(
        session_id=test_session,
        content=(
            "The claim guide explains "
            "the required claim documents."
        ),
    )

    # Follow-up
    result = resolve_question(
        session_id=test_session,
        question=(
            "What happens after "
            "I submit them?"
        ),
    )

    print(
        "\nORIGINAL QUESTION:"
    )

    print(
        result[
            "original_question"
        ]
    )

    print(
        "\nFOLLOW-UP DETECTED:"
    )

    print(
        result[
            "follow_up_detected"
        ]
    )

    print(
        "\nCONTEXT TOPIC:"
    )

    print(
        result.get(
            "context_topic"
        )
    )

    print(
        "\nWAS REWRITTEN:"
    )

    print(
        result[
            "was_rewritten"
        ]
    )

    print(
        "\nRESOLVED QUESTION:"
    )

    print(
        result[
            "resolved_question"
        ]
    )