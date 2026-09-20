# src/data/query_service.py

import re

from sqlalchemy import text

from src.data.db import engine
from src.core.config import settings
from src.core.logger import get_logger


# --------------------------------------------------
# Logger
# --------------------------------------------------

logger = get_logger(__name__)


# --------------------------------------------------
# Blocked SQL Commands
# --------------------------------------------------

BLOCKED_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "MERGE",
    "COPY",
    "CALL",
    "DO",
    "VACUUM"
]


# --------------------------------------------------
# Dangerous PostgreSQL Functions
# --------------------------------------------------

BLOCKED_FUNCTIONS = [
    "PG_SLEEP",
    "PG_READ_FILE",
    "PG_READ_BINARY_FILE",
    "PG_LS_DIR",
    "LO_IMPORT",
    "LO_EXPORT",
    "DBLINK"
]


# --------------------------------------------------
# SQL Validation
# --------------------------------------------------

def validate_select_query(query: str) -> str:
    """
    Validate SQL before sending it to PostgreSQL.

    Only SELECT / WITH queries are allowed.
    """

    if not query:
        raise ValueError(
            "SQL query cannot be empty."
        )


    # Remove leading/trailing whitespace
    cleaned_query = query.strip()


    # --------------------------------------------------
    # Remove final semicolon
    # --------------------------------------------------

    cleaned_query = (
        cleaned_query
        .rstrip(";")
        .strip()
    )


    # --------------------------------------------------
    # Block SQL comments
    # --------------------------------------------------

    if (
        "--" in cleaned_query
        or "/*" in cleaned_query
        or "*/" in cleaned_query
    ):
        raise ValueError(
            "SQL comments are not allowed."
        )


    # --------------------------------------------------
    # Block multiple SQL statements
    # --------------------------------------------------

    if ";" in cleaned_query:
        raise ValueError(
            "Multiple SQL statements are not allowed."
        )


    upper_query = cleaned_query.upper()


    # --------------------------------------------------
    # Only SELECT or WITH
    # --------------------------------------------------

    if not (
        upper_query.startswith("SELECT")
        or upper_query.startswith("WITH")
    ):
        raise ValueError(
            "Only SELECT or WITH queries are allowed."
        )


    # --------------------------------------------------
    # Block dangerous commands
    # --------------------------------------------------

    for keyword in BLOCKED_KEYWORDS:

        pattern = rf"\b{keyword}\b"

        if re.search(
            pattern,
            upper_query
        ):
            raise ValueError(
                f"Unsafe SQL detected: {keyword}"
            )


    # --------------------------------------------------
    # Block dangerous PostgreSQL functions
    # --------------------------------------------------

    for function_name in BLOCKED_FUNCTIONS:

        pattern = (
            rf"\b{function_name}\s*\("
        )

        if re.search(
            pattern,
            upper_query
        ):
            raise ValueError(
                f"Unsafe SQL function detected: "
                f"{function_name}"
            )


    return cleaned_query


# --------------------------------------------------
# Apply Maximum Row Limit
# --------------------------------------------------

def apply_row_limit(query: str) -> str:
    """
    Wrap query so API never returns more than
    SQL_MAX_ROWS records.
    """

    limited_query = f"""
    SELECT *
    FROM (
        {query}
    ) AS safe_result
    LIMIT :max_rows
    """

    return limited_query


# --------------------------------------------------
# Run Query Safely
# --------------------------------------------------

def run_query(query: str) -> list:
    """
    Execute read-only SQL safely.

    Security layers:

    1. Validate SQL
    2. Read-only transaction
    3. Statement timeout
    4. Maximum row limit
    """

    logger.info(
        "SQL validation started"
    )


    # --------------------------------------------------
    # Step 1: Validate SQL
    # --------------------------------------------------

    validated_query = validate_select_query(
        query
    )

    logger.info(
        "SQL validation successful"
    )


    # --------------------------------------------------
    # Step 2: Apply maximum rows
    # --------------------------------------------------

    limited_query = apply_row_limit(
        validated_query
    )


    try:

        with engine.begin() as connection:


            # --------------------------------------------------
            # Step 3: Force transaction to read-only
            # --------------------------------------------------

            connection.execute(
                text(
                    "SET TRANSACTION READ ONLY"
                )
            )


            # --------------------------------------------------
            # Step 4: Query timeout
            # --------------------------------------------------

            timeout_ms = int(
                settings.SQL_STATEMENT_TIMEOUT_MS
            )

            connection.execute(
                text(
                    f"SET LOCAL statement_timeout = "
                    f"'{timeout_ms}ms'"
                )
            )


            logger.info(
                "Executing validated read-only SQL"
            )


            # --------------------------------------------------
            # Step 5: Execute SQL
            # --------------------------------------------------

            result = connection.execute(
                text(limited_query),
                {
                    "max_rows":
                        settings.SQL_MAX_ROWS
                }
            )


            # --------------------------------------------------
            # Step 6: Convert rows into dictionaries
            # --------------------------------------------------

            rows = [
                dict(row._mapping)
                for row in result
            ]


        logger.info(
            f"SQL execution successful. "
            f"Rows returned: {len(rows)}"
        )

        return rows


    except Exception:

        logger.exception(
            "SQL execution failed"
        )

        raise