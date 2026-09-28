# src/memory/conversation_store.py

import os
import logging

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ==========================================================
# LOAD ENV
# ==========================================================

load_dotenv(override=True)


# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# MEMORY DATABASE CONFIG
# ==========================================================

MEMORY_DB_USER = os.getenv(
    "MEMORY_DB_USER",
    "insurance_app",
)

MEMORY_DB_PASSWORD = os.getenv(
    "MEMORY_DB_PASSWORD"
)

MEMORY_DB_HOST = os.getenv(
    "MEMORY_DB_HOST",
    "localhost",
)

MEMORY_DB_PORT = int(
    os.getenv(
        "MEMORY_DB_PORT",
        "5432",
    )
)

MEMORY_DB_NAME = os.getenv(
    "MEMORY_DB_NAME",
    "insurance_ai_copilot",
)


# ==========================================================
# DATABASE URL
# ==========================================================

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=MEMORY_DB_USER,
    password=MEMORY_DB_PASSWORD,
    host=MEMORY_DB_HOST,
    port=MEMORY_DB_PORT,
    database=MEMORY_DB_NAME,
)


# ==========================================================
# ENGINE
# ==========================================================

memory_engine = create_engine(
    database_url,
    pool_pre_ping=True,
)


# ==========================================================
# TEST CONNECTION
# ==========================================================

def test_memory_connection():

    try:

        with memory_engine.connect() as connection:

            result = connection.execute(
                text("SELECT 1")
            )

            value = result.scalar()

        return value == 1

    except Exception as error:

        logger.exception(
            "Memory database connection failed"
        )

        print(error)

        return False


# ==========================================================
# ENSURE SESSION EXISTS
# ==========================================================

def ensure_session(session_id):

    query = text(
        """
        INSERT INTO app.copilot_sessions
        (
            session_id,
            created_at,
            updated_at
        )
        VALUES
        (
            :session_id,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        )

        ON CONFLICT (session_id)

        DO UPDATE SET

            updated_at =
                CURRENT_TIMESTAMP;
        """
    )

    with memory_engine.begin() as connection:

        connection.execute(
            query,
            {
                "session_id": session_id
            },
        )


# ==========================================================
# ADD MESSAGE
# ==========================================================

def add_message(
    session_id,
    role,
    content,
    route=None,
):

    ensure_session(
        session_id
    )

    query = text(
        """
        INSERT INTO app.copilot_messages
        (
            session_id,
            role,
            content,
            route
        )
        VALUES
        (
            :session_id,
            :role,
            :content,
            :route
        );
        """
    )

    with memory_engine.begin() as connection:

        connection.execute(
            query,
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "route": route,
            },
        )


# ==========================================================
# ADD USER MESSAGE
# ==========================================================

def add_user_message(
    session_id,
    content,
):

    add_message(
        session_id=session_id,
        role="user",
        content=content,
    )


# ==========================================================
# ADD ASSISTANT MESSAGE
# ==========================================================

def add_assistant_message(
    session_id,
    content,
    route=None,
):

    add_message(
        session_id=session_id,
        role="assistant",
        content=content,
        route=route,
    )


# ==========================================================
# GET HISTORY
# ==========================================================

def get_history(
    session_id,
    limit=20,
):

    query = text(
        """
        SELECT
            message_id,
            role,
            content,
            route,
            created_at

        FROM
        (
            SELECT
                message_id,
                role,
                content,
                route,
                created_at

            FROM app.copilot_messages

            WHERE session_id =
                :session_id

            ORDER BY
                message_id DESC

            LIMIT :limit
        ) recent

        ORDER BY
            message_id ASC;
        """
    )

    with memory_engine.connect() as connection:

        rows = connection.execute(
            query,
            {
                "session_id": session_id,
                "limit": limit,
            },
        ).mappings().all()

    return [
        dict(row)
        for row in rows
    ]


# ==========================================================
# FORMAT HISTORY
# ==========================================================

def format_history(
    session_id,
    limit=10,
):

    history = get_history(
        session_id,
        limit,
    )

    lines = []

    for message in history:

        role = message[
            "role"
        ].upper()

        content = message[
            "content"
        ]

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(
        lines
    )


# ==========================================================
# MESSAGE COUNT
# ==========================================================

def message_count(
    session_id,
):

    query = text(
        """
        SELECT
            COUNT(*)

        FROM app.copilot_messages

        WHERE session_id =
            :session_id;
        """
    )

    with memory_engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "session_id": session_id
            },
        )

        count = result.scalar()

    return int(
        count or 0
    )


# ==========================================================
# SET LAST ROUTE
# ==========================================================

def set_session_route(
    session_id,
    route,
):

    ensure_session(
        session_id
    )

    query = text(
        """
        UPDATE app.copilot_sessions

        SET
            last_route =
                :route,

            updated_at =
                CURRENT_TIMESTAMP

        WHERE session_id =
            :session_id;
        """
    )

    with memory_engine.begin() as connection:

        connection.execute(
            query,
            {
                "session_id": session_id,
                "route": route,
            },
        )


# ==========================================================
# GET LAST ROUTE
# ==========================================================

def get_session_route(
    session_id,
):

    query = text(
        """
        SELECT
            last_route

        FROM app.copilot_sessions

        WHERE session_id =
            :session_id;
        """
    )

    with memory_engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "session_id": session_id
            },
        )

        route = result.scalar()

    return route


# ==========================================================
# SET CONTEXT TOPIC
# ==========================================================

def set_context_topic(
    session_id,
    context_topic,
):

    ensure_session(
        session_id
    )

    query = text(
        """
        UPDATE app.copilot_sessions

        SET
            context_topic =
                :context_topic,

            updated_at =
                CURRENT_TIMESTAMP

        WHERE session_id =
            :session_id;
        """
    )

    with memory_engine.begin() as connection:

        connection.execute(
            query,
            {
                "session_id": session_id,
                "context_topic": context_topic,
            },
        )


# ==========================================================
# GET CONTEXT TOPIC
# ==========================================================

def get_context_topic(
    session_id,
):

    query = text(
        """
        SELECT
            context_topic

        FROM app.copilot_sessions

        WHERE session_id =
            :session_id;
        """
    )

    with memory_engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "session_id": session_id
            },
        )

        topic = result.scalar()

    return topic


# ==========================================================
# CLEAR SESSION
# ==========================================================

def clear_session(
    session_id,
):

    query = text(
        """
        DELETE FROM app.copilot_sessions

        WHERE session_id =
            :session_id;
        """
    )

    with memory_engine.begin() as connection:

        result = connection.execute(
            query,
            {
                "session_id": session_id
            },
        )

    return result.rowcount > 0

# ==========================================================
# SESSION EXISTS
# ==========================================================

def session_exists(
    session_id,
):

    query = text(
        """
        SELECT EXISTS
        (
            SELECT 1
            FROM app.copilot_sessions
            WHERE session_id = :session_id
        );
        """
    )

    with memory_engine.connect() as connection:

        result = connection.execute(
            query,
            {
                "session_id": session_id
            },
        )

        exists = result.scalar()

    return bool(exists)


# ==========================================================
# CLEAR ALL SESSIONS
# ==========================================================

def clear_all():

    query = text(
        """
        DELETE FROM app.copilot_sessions;
        """
    )

    with memory_engine.begin() as connection:

        result = connection.execute(
            query
        )

    return int(
        result.rowcount or 0
    )


# ==========================================================
# SESSION COUNT
# ==========================================================

def session_count():

    query = text(
        """
        SELECT COUNT(*)
        FROM app.copilot_sessions;
        """
    )

    with memory_engine.connect() as connection:

        result = connection.execute(
            query
        )

        count = result.scalar()

    return int(
        count or 0
    )


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "Memory DB connected:",
        test_memory_connection(),
    )