# src/memory/session_memory.py

from src.memory.conversation_store import (
    ensure_session as db_ensure_session,
    add_message as db_add_message,
    add_user_message as db_add_user_message,
    add_assistant_message as db_add_assistant_message,
    get_history as db_get_history,
    format_history as db_format_history,
    message_count as db_message_count,
    session_exists as db_session_exists,
    clear_session as db_clear_session,
    clear_all as db_clear_all,
    session_count as db_session_count,
)


# ==========================================================
# SESSION MEMORY
# ==========================================================

class SessionMemory:
    """
    Persistent conversational memory.

    Public interface remains similar to our
    previous in-memory implementation, but
    PostgreSQL is now the source of truth.
    """

    def __init__(
        self,
        max_messages=20,
    ):

        self.max_messages = (
            max_messages
        )


    # ======================================================
    # ENSURE SESSION
    # ======================================================

    def ensure_session(
        self,
        session_id,
    ):

        db_ensure_session(
            session_id
        )


    # ======================================================
    # ADD GENERIC MESSAGE
    # ======================================================

    def add_message(
        self,
        session_id,
        role,
        content,
        route=None,
    ):

        db_add_message(
            session_id=session_id,
            role=role,
            content=content,
            route=route,
        )


    # ======================================================
    # ADD USER MESSAGE
    # ======================================================

    def add_user_message(
        self,
        session_id,
        content,
    ):

        db_add_user_message(
            session_id=session_id,
            content=content,
        )


    # ======================================================
    # ADD ASSISTANT MESSAGE
    # ======================================================

    def add_assistant_message(
        self,
        session_id,
        content,
        route=None,
    ):

        db_add_assistant_message(
            session_id=session_id,
            content=content,
            route=route,
        )


    # ======================================================
    # GET HISTORY
    # ======================================================

    def get_history(
        self,
        session_id,
        limit=None,
    ):

        if limit is None:

            limit = (
                self.max_messages
            )

        return db_get_history(
            session_id=session_id,
            limit=limit,
        )


    # ======================================================
    # RECENT HISTORY
    # ======================================================

    def get_recent(
        self,
        session_id,
        limit=10,
    ):

        return db_get_history(
            session_id=session_id,
            limit=limit,
        )


    def get_recent_messages(
        self,
        session_id,
        limit=10,
    ):

        return self.get_recent(
            session_id=session_id,
            limit=limit,
        )


    # ======================================================
    # FORMAT HISTORY
    # ======================================================

    def format_history(
        self,
        session_id,
        limit=10,
    ):

        return db_format_history(
            session_id=session_id,
            limit=limit,
        )


    # Compatibility helper
    def format_for_prompt(
        self,
        session_id,
        limit=10,
    ):

        return self.format_history(
            session_id=session_id,
            limit=limit,
        )


    # ======================================================
    # MESSAGE COUNT
    # ======================================================

    def message_count(
        self,
        session_id,
    ):

        return db_message_count(
            session_id
        )


    # ======================================================
    # SESSION EXISTS
    # ======================================================

    def session_exists(
        self,
        session_id,
    ):

        return db_session_exists(
            session_id
        )


    # ======================================================
    # CLEAR SESSION
    # ======================================================

    def clear_session(
        self,
        session_id,
    ):

        return db_clear_session(
            session_id
        )


    # ======================================================
    # CLEAR ALL
    # ======================================================

    def clear_all(
        self,
    ):

        return db_clear_all()


    # ======================================================
    # SESSION COUNT
    # ======================================================

    def session_count(
        self,
    ):

        return db_session_count()


# ==========================================================
# GLOBAL MEMORY OBJECT
# ==========================================================

session_memory = SessionMemory(
    max_messages=20
)