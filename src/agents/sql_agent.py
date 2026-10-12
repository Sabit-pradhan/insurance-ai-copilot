# src/agents/sql_agent.py

import logging
import re
from decimal import Decimal
from typing import Any

from src.utils.llm import llm
from src.data.schema_service import get_database_schema
from src.data.query_service import run_query

logger = logging.getLogger(__name__)


# ==========================================================
# SQL CLEANING
# ==========================================================

def clean_sql(sql: str) -> str:
    """Extract a SQL query from the LLM response."""

    if not sql:
        raise ValueError("The LLM returned an empty SQL response.")

    sql = str(sql).strip()

    sql = re.sub(
        r"<think>.*?</think>",
        "",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        sql = match.group(1).strip()

    sql = re.sub(
        r"^\s*SQL\s*:\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.strip().rstrip(";").strip()

    start = re.search(r"\b(SELECT|WITH)\b", sql, re.IGNORECASE)

    if not start:
        raise ValueError(
            "The LLM did not return a SELECT or WITH query."
        )

    return sql[start.start():].strip().rstrip(";").strip()


# ==========================================================
# SQL VALIDATION
# ==========================================================

def validate_sql(sql: str) -> None:
    """Permit read-only queries and reject prohibited operations."""

    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")

    query = sql.strip()

    if ";" in query:
        raise ValueError("Multiple SQL statements are not allowed.")

    if "--" in query or "/*" in query or "*/" in query:
        raise ValueError("SQL comments are not allowed.")

    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError(
            "Only SELECT or WITH ... SELECT queries are allowed."
        )

    dangerous_pattern = (
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|"
        r"GRANT|REVOKE|COPY|CALL|MERGE|DO|EXECUTE|"
        r"VACUUM|ANALYZE|REFRESH|REINDEX)\b"
    )

    if re.search(dangerous_pattern, query, re.IGNORECASE):
        raise ValueError("The SQL contains a prohibited operation.")

    if re.search(r"\bINTO\b", query, re.IGNORECASE):
        raise ValueError("SELECT INTO is not allowed.")

    unsafe_functions = (
        r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|"
        r"pg_write_file|lo_import|lo_export)\s*\("
    )

    if re.search(unsafe_functions, query, re.IGNORECASE):
        raise ValueError("Unsafe database functions are not allowed.")


# ==========================================================
# SCHEMA-AWARE HELPERS
# ==========================================================

def _schema_has_customers_table(schema: Any) -> bool:
    """Check whether the schema mentions the CUSTOMERS table."""

    if not schema:
        return False

    return bool(
        re.search(
            r"\bCUSTOMERS\b",
            str(schema),
            re.IGNORECASE,
        )
    )


def _is_customer_count_question(question: str) -> bool:
    """Recognize common requests for total customer counts."""

    normalized = re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip(" \t\r\n?!.,")

    patterns = (
        r"total customers?",
        r"count customers?",
        r"number of customers?",
        r"total number of customers?",
        r"how many customers (?:are there|do we have|are in (?:the )?database)?",
    )

    return any(
        re.fullmatch(pattern, normalized)
        for pattern in patterns
    )


def _is_state_group_question(question: str) -> bool:
    """Recognize questions asking for customer counts by state."""

    normalized = re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip(" \t\r\n?!.,")

    patterns = (
        r"\bcustomers?\b.*\b(each|per|by)\s+state\b",
        r"\bstate[- ]wise customers?\b",
        r"\bcustomers?\s+in\s+each\s+state\b",
        r"\bhow many customers?\b.*\bstate\b",
        r"\bnumber of customers?\b.*\bstate\b",
        r"\bcount of customers?\b.*\bstate\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


# ==========================================================
# SQL GENERATION
# ==========================================================

def generate_sql(question: str) -> str:
    """Generate read-only PostgreSQL using the database schema."""

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    schema = get_database_schema()

    if not schema:
        raise ValueError("Could not retrieve the database schema.")

    normalized_question = re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip(" \t\r\n?!.,")

    schema_text = str(schema).lower()

    # ------------------------------------------------------
    # 1. Total number of policies
    # ------------------------------------------------------

    is_policy_count = re.search(
        r"\b(how many|total number of|number of|total)\s+policies\b",
        normalized_question,
    )

    if "policies" in schema_text and is_policy_count:
        sql = """
            SELECT COUNT(*) AS policy_count
            FROM policies
        """

        validate_sql(sql)
        logger.info("Generated policy-count SQL.")
        return sql.strip()

    # ------------------------------------------------------
    # 2. Overdue premium payments
    # ------------------------------------------------------

    is_overdue_question = (
        bool(
            re.search(
                r"\b(overdue|past due)\b",
                normalized_question,
            )
        )
        and bool(
            re.search(
                r"\b(payment|payments|premium|premiums)\b",
                normalized_question,
            )
        )
    )

    if (
        "premiums" in schema_text
        and "payment_status" in schema_text
        and is_overdue_question
    ):
        sql = """
            SELECT COUNT(*) AS overdue_payments
            FROM premiums
            WHERE LOWER(payment_status) = 'overdue'
        """

        validate_sql(sql)
        logger.info("Generated overdue-payment SQL.")
        return sql.strip()

    # ------------------------------------------------------
    # 3. Customers in each state
    # ------------------------------------------------------

    if (
        "customers" in schema_text
        and "customer_id" in schema_text
        and "state" in schema_text
        and _is_state_group_question(question)
    ):
        sql = """
            SELECT
                state,
                COUNT(DISTINCT customer_id) AS customer_count
            FROM customers
            GROUP BY state
            ORDER BY customer_count DESC, state
        """

        validate_sql(sql)
        logger.info("Generated state-wise customer SQL.")
        return sql.strip()

    # ------------------------------------------------------
    # 4. Total number of customers
    # ------------------------------------------------------

    if _is_customer_count_question(question):
        if _schema_has_customers_table(schema):
            if "customer_id" in schema_text:
                sql = """
                    SELECT COUNT(DISTINCT customer_id) AS total_customers
                    FROM customers
                """
            else:
                sql = """
                    SELECT COUNT(*) AS total_customers
                    FROM customers
                """

            validate_sql(sql)
            logger.info("Generated deterministic customer-count SQL.")
            return sql.strip()

        logger.warning(
            "CUSTOMERS table was not clearly identified in the schema. "
            "Using schema-aware LLM."
        )

    # ------------------------------------------------------
    # 5. Dynamic SQL generation through the LLM
    # ------------------------------------------------------

    prompt = f"""
You are the SQL generation component of an Insurance AI Copilot.

Convert the user's question into ONE PostgreSQL read-only query
using the database schema provided below.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question.strip()}

RULES:
1. Use only tables and columns present in the schema.
2. Generate valid PostgreSQL syntax.
3. Generate exactly one SELECT or WITH ... SELECT query.
4. Never generate data-modification statements.
5. Use JOINs only when required and supported by the schema.
6. Use GROUP BY and aggregate functions when appropriate.
7. Support COUNT, SUM, AVG, MIN, MAX, CASE, DISTINCT and HAVING.
8. Support ORDER BY, ranking, top-N and bottom-N queries.
9. Support PostgreSQL window functions when appropriate.
10. Support CTEs, date filtering, date grouping and string functions.
11. Use appropriate PostgreSQL date expressions.
12. Do not invent tables, columns, relationships or business facts.
13. Do not assume a join key unless supported by the schema.
14. Avoid SELECT * unless explicitly requested.
15. Use clear column aliases.
16. Return only the SQL query, without Markdown or explanation.

SQL:
"""

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)

    sql = clean_sql(str(content))
    validate_sql(sql)

    logger.info("Generated SQL: %s", sql)
    return sql


# ==========================================================
# RESULT FORMATTING
# ==========================================================

def format_value(value: Any) -> str:
    """Format database values for readable output."""

    if value is None:
        return "NULL"

    if isinstance(value, Decimal):
        return f"{value:,.2f}"

    if isinstance(value, float):
        return f"{value:,.2f}"

    if isinstance(value, int):
        return f"{value:,}"

    return str(value)


def generate_result_summary(results: list[dict]) -> str:
    """Format SQL results as a readable Markdown table."""

    if not isinstance(results, list):
        return str(results)

    if not results:
        return (
            "The query executed successfully, "
            "but no matching records were found."
        )

    if not all(isinstance(row, dict) for row in results):
        return str(results)

    columns = list(results[0].keys())

    if not columns:
        return "The query returned no displayable columns."

    headers = [
        str(column).replace("_", " ").title()
        for column in columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in results[:50]:
        values = []

        for column in columns:
            value = format_value(row.get(column))
            value = value.replace("|", r"\|").replace("\n", " ")
            values.append(value)

        lines.append("| " + " | ".join(values) + " |")

    answer = "\n".join(lines)

    if len(results) > 50:
        answer += (
            f"\n\nShowing the first 50 of "
            f"{len(results)} returned rows."
        )

    return answer


# ==========================================================
# NATURAL-LANGUAGE RESULT EXPLANATION
# ==========================================================

def generate_natural_answer(
    question: str,
    sql: str,
    results: list[dict],
) -> str:
    """Explain SQL results without inventing values or context."""

    if not isinstance(results, list):
        return str(results)

    if not results:
        return (
            "The query ran successfully, "
            "but no matching records were found."
        )

    result_text = generate_result_summary(results)

    normalized_question = re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip(" \t\r\n?!.,")

    normalized_sql = sql.lower()

    # ------------------------------------------------------
    # 1. Deterministic response for customers by state
    # ------------------------------------------------------

    if (
        _is_state_group_question(normalized_question)
        and "group by state" in normalized_sql
        and all(isinstance(row, dict) for row in results)
        and "state" in results[0]
        and "customer_count" in results[0]
    ):
        lines = [
            "The following table shows the number of unique "
            "customers registered in each state across the "
            "insurance database.",
            "",
            "| State | Customer Count |",
            "|---|---:|",
        ]

        total_customers = 0

        for row in results:
            state = row.get("state")
            count = row.get("customer_count")

            if isinstance(count, (int, float, Decimal)):
                total_customers += int(count)

            state_label = (
                str(state)
                if state is not None
                else "Unknown"
            )

            lines.append(
                f"| {state_label} | {format_value(count)} |"
            )

        lines.extend([
            "",
            (
                f"**Total across the returned state groups: "
                f"{format_value(total_customers)} customers.**"
            ),
            "",
            "These are database-wide customer counts grouped "
            "by state, not counts for a single insurance policy.",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------
    # 2. Deterministic response for single-value aggregates
    # ------------------------------------------------------

    if (
        len(results) == 1
        and isinstance(results[0], dict)
        and len(results[0]) == 1
    ):
        column, value = next(iter(results[0].items()))

        if isinstance(value, (int, float, Decimal)):
            if column == "overdue_payments":
                return f"Overdue payments: {format_value(value)}"

            if column == "total_customers":
                return f"Total customers: {format_value(value)}"

            if column == "policy_count":
                return f"Total policies: {format_value(value)}"

            label = str(column).replace("_", " ")
            return f"{label.capitalize()}: {format_value(value)}"

    # ------------------------------------------------------
    # 3. LLM explanation for other SQL results
    # ------------------------------------------------------

    prompt = f"""
You are the answer-generation component of an Insurance AI Copilot.

Answer the user's question using ONLY the actual SQL results below.

USER QUESTION:
{question}

SQL EXECUTED:
{sql}

ACTUAL SQL RESULTS:
{result_text}

RULES:
1. Answer the question directly and clearly.
2. Use only values present in the SQL results.
3. Do not invent numbers, records, causes or business conclusions.
4. Include the table when it helps answer the question.
5. Explain SQL output in simple business language.
6. Do not describe database-wide results as belonging to a
   single insurance policy unless the SQL explicitly filters
   for that policy.
7. Do not claim that a prediction was generated unless the
   results actually contain it.
8. Do not reveal hidden reasoning.
9. Use readable Markdown.
10. Clearly distinguish overall database statistics from
    policy-specific statistics.

ANSWER:
"""

    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        answer = str(content).strip()

        if answer:
            return answer

    except Exception:
        logger.exception(
            "Natural-language answer generation failed."
        )

    return result_text


# ==========================================================
# MAIN SQL AGENT
# ==========================================================

def ask_sql_agent(question: str) -> dict[str, Any]:
    """
    Main SQL Agent entry point.

    Flow:
    1. Receive the user's question.
    2. Retrieve the database schema.
    3. Generate SQL.
    4. Validate SQL.
    5. Execute the query.
    6. Explain the actual results.
    """

    logger.info(
        "SQL Agent request received: %s",
        question,
    )

    try:
        sql = generate_sql(question)

        # Validate immediately before execution.
        validate_sql(sql)

        results = run_query(sql)

        if results is None:
            results = []
        elif not isinstance(results, list):
            results = list(results)

        answer = generate_natural_answer(
            question=question,
            sql=sql,
            results=results,
        )

        return {
            "question": question,
            "sql": sql,
            "results": results,
            "answer": answer,
            "source": "dynamic_sql_agent",
            "success": True,
        }

    except Exception as exc:
        logger.exception(
            "SQL Agent failed while processing question: %s",
            question,
        )

        return {
            "question": question,
            "sql": None,
            "results": [],
            "answer": (
                "I could not complete that database query. "
                "Please check the question, database schema, "
                "and application logs."
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "source": "dynamic_sql_agent",
            "success": False,
        }