# src/agents/sql_agent.py

import re
import logging
from decimal import Decimal
from typing import Any

from src.utils.llm import llm
from src.data.schema_service import get_database_schema
from src.data.query_service import run_query


logger = logging.getLogger(__name__)


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize_question(question: str) -> str:

    text = question.lower().strip()

    text = re.sub(
        r"[^\w\s%-]",
        " ",
        text,
    )

    replacements = {
        "customers": "customer",
        "policies": "policy",
        "claims": "claim",
        "agents": "agent",
        "premiums": "premium",
        "renewals": "renewal",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_any(
    text: str,
    phrases: list[str],
) -> bool:

    return any(
        phrase in text
        for phrase in phrases
    )


# ==========================================================
# SQL SAFETY
# ==========================================================

DANGEROUS_SQL = [
    "insert ",
    "update ",
    "delete ",
    "drop ",
    "alter ",
    "truncate ",
    "create ",
    "grant ",
    "revoke ",
    "copy ",
    "call ",
    "merge ",
]


def clean_sql(sql: str) -> str:
    """
    Extract one PostgreSQL SELECT / WITH query from Ollama output.

    Handles common model output such as:
    - <think>...</think> blocks
    - ```sql ... ``` fences
    - "SQL: SELECT ..." prefixes
    - short explanatory text before the query
    """

    if not sql:
        raise ValueError("Ollama returned an empty SQL response.")

    text = str(sql).strip()

    # Qwen models can occasionally return reasoning blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Prefer SQL inside a fenced code block when present.
    fenced = re.search(
        r"```(?:sql)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        text = fenced.group(1).strip()
    else:
        text = text.replace("```sql", "")
        text = text.replace("```SQL", "")
        text = text.replace("```", "")
        text = text.strip()

    # Remove labels such as "SQL:" or "Query:".
    text = re.sub(
        r"^(?:sql|query)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Find the first line that actually begins a SELECT / WITH query.
    lines = text.splitlines()
    start_index = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        if re.match(
            r"^(select|with)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            start_index = index
            break

    if start_index is not None:
        text = "\n".join(lines[start_index:]).strip()
    else:
        # Fallback for a query returned on the same line as some text.
        match = re.search(
            r"\b(select|with)\b[\s\S]*",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            raise ValueError(
                "Ollama did not return a SELECT/WITH SQL query."
            )

        text = match.group(0).strip()

    # Keep only the first SQL statement.
    semicolon_position = text.find(";")

    if semicolon_position != -1:
        text = text[:semicolon_position + 1]

    return text.strip()


def validate_sql(sql: str) -> None:
    """
    Only read-only SELECT / WITH queries are allowed.
    """

    normalized = sql.lower().strip()

    if not (
        normalized.startswith("select")
        or normalized.startswith("with")
    ):
        raise ValueError(
            "Only SELECT queries are allowed."
        )

    for keyword in DANGEROUS_SQL:

        if keyword in normalized:
            raise ValueError(
                f"Unsafe SQL detected: {keyword.strip()}"
            )


# ==========================================================
# COMMON SQL QUESTIONS
# ==========================================================

def get_local_sql(
    question: str,
) -> tuple[str, str] | None:
    """
    Common questions are converted directly to SQL.

    Returns:
        (sql, query_type)

    No Ollama call required.
    """

    q = normalize_question(
        question
    )


    # ======================================================
    # TOTAL CUSTOMERS
    # ======================================================

    if contains_any(
        q,
        [
            "how many customer do we have",
            "how many customer",
            "total customer",
            "number of customer",
            "customer count",
            "kitne customer",
        ],
    ) and "state" not in q:

        return (
            """
            SELECT COUNT(*) AS total_customers
            FROM customers;
            """,
            "total_customers",
        )


    # ======================================================
    # TOTAL POLICIES
    # ======================================================

    if (
        contains_any(
            q,
            [
                "how many policy do we have",
                "total policy",
                "policy count",
                "kitni policy",
            ],
        )
        and not contains_any(
            q,
            [
                "agent",
                "top",
                "sold",
                "sales",
                "channel",
                "state",
                "customer",
                "premium",
                "claim",
                "type",
                "by ",
            ],
        )
    ):

        return (
            """
            SELECT COUNT(*) AS total_policies
            FROM policies;
            """,
            "total_policies",
        )


    # ======================================================
    # TOTAL CLAIMS
    # ======================================================

    if contains_any(
        q,
        [
            "how many claim",
            "total claim",
            "number of claim",
            "claim count",
            "kitne claim",
        ],
    ) and "type" not in q and "amount" not in q:

        return (
            """
            SELECT COUNT(*) AS total_claims
            FROM claims;
            """,
            "total_claims",
        )


    # ======================================================
    # TOTAL AGENTS
    # ======================================================

    if contains_any(
        q,
        [
            "how many agent",
            "total agent",
            "number of agent",
            "agent count",
        ],
    ):

        return (
            """
            SELECT COUNT(*) AS total_agents
            FROM agents;
            """,
            "total_agents",
        )


    # ======================================================
    # PREMIUM TRANSACTION COUNT
    # ======================================================

    if contains_any(
        q,
        [
            "how many premium transaction",
            "premium transaction count",
            "total premium transaction",
        ],
    ):

        return (
            """
            SELECT COUNT(*) AS total_premium_transactions
            FROM premiums;
            """,
            "total_premium_transactions",
        )


    # ======================================================
    # RENEWAL RECORD COUNT
    # ======================================================

    if contains_any(
        q,
        [
            "how many renewal record",
            "total renewal record",
            "renewal record count",
        ],
    ):

        return (
            """
            SELECT COUNT(*) AS total_renewal_records
            FROM renewals;
            """,
            "total_renewal_records",
        )


    # ======================================================
    # UNDERWRITING RECORD COUNT
    # ======================================================

    if contains_any(
        q,
        [
            "how many underwriting record",
            "total underwriting record",
            "underwriting record count",
        ],
    ):

        return (
            """
            SELECT COUNT(*) AS total_underwriting_records
            FROM underwriting;
            """,
            "total_underwriting_records",
        )


    # ======================================================
    # AVERAGE PREMIUM
    # ======================================================

    if (
        contains_any(
            q,
            [
                "average premium",
                "avg premium",
                "mean premium",
            ],
        )
        and "policy type" not in q
        and "state" not in q
    ):

        return (
            """
            SELECT ROUND(
                AVG(annual_premium)::numeric,
                2
            ) AS average_annual_premium
            FROM policies;
            """,
            "average_premium",
        )


    # ======================================================
    # TOTAL PREMIUM
    # ======================================================

    if contains_any(
        q,
        [
            "total annual premium",
            "total premium",
            "sum of premium",
        ],
    ) and "policy type" not in q:

        return (
            """
            SELECT
                SUM(annual_premium)
                AS total_annual_premium
            FROM policies;
            """,
            "total_premium",
        )


    # ======================================================
    # STATE WITH MOST CUSTOMERS
    # ======================================================

    if (
        "state" in q
        and "customer" in q
        and contains_any(
            q,
            [
                "most",
                "highest",
                "maximum",
                "max",
                "sabse jyada",
                "sabse zyada",
            ],
        )
    ):

        return (
            """
            SELECT
                state,
                COUNT(*) AS total_customers
            FROM customers
            GROUP BY state
            ORDER BY total_customers DESC
            LIMIT 1;
            """,
            "top_customer_state",
        )


    # ======================================================
    # CUSTOMER DISTRIBUTION BY STATE
    # ======================================================

    if (
        "customer" in q
        and "state" in q
        and contains_any(
            q,
            [
                "distribution",
                "breakdown",
                "by state",
            ],
        )
    ):

        return (
            """
            SELECT
                state,
                COUNT(*) AS total_customers
            FROM customers
            GROUP BY state
            ORDER BY total_customers DESC;
            """,
            "customers_by_state",
        )


    # ======================================================
    # AVAILABLE POLICY TYPES
    # ======================================================

    if (
        "policy type" in q
        and contains_any(
            q,
            [
                "what",
                "which",
                "show",
                "list",
                "available",
                "exist",
                "have",
            ],
        )
        and "premium" not in q
        and "highest" not in q
    ):

        return (
            """
            SELECT DISTINCT policy_type
            FROM policies
            WHERE policy_type IS NOT NULL
            ORDER BY policy_type;
            """,
            "policy_types",
        )


    # ======================================================
    # POLICY DISTRIBUTION BY TYPE
    # ======================================================

    if (
        "policy" in q
        and "type" in q
        and contains_any(
            q,
            [
                "distribution",
                "breakdown",
                "count by",
            ],
        )
    ):

        return (
            """
            SELECT
                policy_type,
                COUNT(*) AS total_policies
            FROM policies
            GROUP BY policy_type
            ORDER BY total_policies DESC;
            """,
            "policies_by_type",
        )


    # ======================================================
    # POLICY TYPE WITH HIGHEST AVERAGE PREMIUM
    # ======================================================

    if (
        "policy" in q
        and "premium" in q
        and contains_any(
            q,
            [
                "average",
                "avg",
                "mean",
            ],
        )
        and contains_any(
            q,
            [
                "highest",
                "most",
                "maximum",
                "max",
                "sabse jyada",
                "sabse zyada",
            ],
        )
    ):

        return (
            """
            SELECT
                policy_type,
                ROUND(
                    AVG(annual_premium)::numeric,
                    2
                ) AS average_annual_premium
            FROM policies
            GROUP BY policy_type
            ORDER BY average_annual_premium DESC
            LIMIT 1;
            """,
            "highest_average_premium_policy_type",
        )


    # ======================================================
    # POLICY TYPE WITH HIGHEST TOTAL PREMIUM
    # ======================================================

    if (
        "policy" in q
        and "premium" in q
        and contains_any(
            q,
            [
                "highest",
                "most",
                "maximum",
                "generates the highest",
                "sabse jyada",
                "sabse zyada",
            ],
        )
        and not contains_any(
            q,
            [
                "average",
                "avg",
                "mean",
            ],
        )
    ):

        return (
            """
            SELECT
                policy_type,
                SUM(annual_premium)
                AS total_annual_premium
            FROM policies
            GROUP BY policy_type
            ORDER BY total_annual_premium DESC
            LIMIT 1;
            """,
            "highest_premium_policy_type",
        )


    # ======================================================
    # AVERAGE PREMIUM BY POLICY TYPE
    # ======================================================

    if (
        "policy" in q
        and "premium" in q
        and contains_any(
            q,
            [
                "average",
                "avg",
                "mean",
            ],
        )
        and contains_any(
            q,
            [
                "by policy type",
                "policy type",
                "breakdown",
                "compare",
                "show",
            ],
        )
    ):

        return (
            """
            SELECT
                policy_type,
                ROUND(
                    AVG(annual_premium)::numeric,
                    2
                ) AS average_annual_premium
            FROM policies
            GROUP BY policy_type
            ORDER BY average_annual_premium DESC;
            """,
            "average_premium_by_policy_type",
        )


    # ======================================================
    # POLICY TYPE PREMIUM BREAKDOWN
    # ======================================================

    if (
        "policy type" in q
        and "premium" in q
        and contains_any(
            q,
            [
                "show",
                "breakdown",
                "by policy type",
                "compare",
            ],
        )
        and not contains_any(
            q,
            [
                "average",
                "avg",
                "mean",
            ],
        )
    ):

        return (
            """
            SELECT
                policy_type,
                SUM(annual_premium)
                AS total_annual_premium
            FROM policies
            GROUP BY policy_type
            ORDER BY total_annual_premium DESC;
            """,
            "premium_by_policy_type",
        )


    # ======================================================
    # TOTAL CLAIM AMOUNT
    # ======================================================

    if contains_any(
        q,
        [
            "total claim amount",
            "sum of claim amount",
        ],
    ):

        return (
            """
            SELECT
                SUM(claim_amount)
                AS total_claim_amount
            FROM claims;
            """,
            "total_claim_amount",
        )


    # ======================================================
    # AVERAGE CLAIM AMOUNT
    # ======================================================

    if contains_any(
        q,
        [
            "average claim amount",
            "avg claim amount",
            "mean claim amount",
        ],
    ):

        return (
            """
            SELECT
                ROUND(
                    AVG(claim_amount)::numeric,
                    2
                ) AS average_claim_amount
            FROM claims;
            """,
            "average_claim_amount",
        )


    # ======================================================
    # CLAIM DISTRIBUTION BY TYPE
    # ======================================================

    if (
        "claim" in q
        and "type" in q
        and contains_any(
            q,
            [
                "distribution",
                "breakdown",
                "count",
                "most common",
                "occur",
            ],
        )
    ):

        return (
            """
            SELECT
                claim_type,
                COUNT(*) AS total_claims
            FROM claims
            GROUP BY claim_type
            ORDER BY total_claims DESC;
            """,
            "claims_by_type",
        )


    # ======================================================
    # CLAIM TYPE WITH HIGHEST TOTAL CLAIM AMOUNT
    # ======================================================

    if (
        "claim type" in q
        and "claim amount" in q
        and contains_any(
            q,
            [
                "highest",
                "maximum",
                "most",
                "sabse jyada",
                "sabse zyada",
            ],
        )
    ):

        return (
            """
            SELECT
                claim_type,
                SUM(claim_amount)
                AS total_claim_amount
            FROM claims
            GROUP BY claim_type
            ORDER BY total_claim_amount DESC
            LIMIT 1;
            """,
            "highest_claim_type",
        )


    # ======================================================
    # CLAIM SEVERITY DISTRIBUTION
    # ======================================================

    if (
        "claim" in q
        and "severity" in q
        and contains_any(
            q,
            [
                "distribution",
                "breakdown",
                "show",
                "count",
            ],
        )
    ):

        return (
            """
            SELECT
                claim_severity,
                COUNT(*) AS total_claims
            FROM claims
            GROUP BY claim_severity
            ORDER BY total_claims DESC;
            """,
            "claims_by_severity",
        )


    # ======================================================
    # SPECIFIC POLICY TYPE COUNT
    # ======================================================

    known_policy_types = [
        "health",
        "motor",
        "term life",
        "whole life",
        "travel",
        "home",
        "commercial",
        "personal accident",
    ]


    for policy_type in known_policy_types:

        if (
            policy_type in q
            and "policy" in q
            and contains_any(
                q,
                [
                    "how many",
                    "count",
                    "total",
                    "kitni",
                ],
            )
        ):

            return (
                f"""
                SELECT
                    COUNT(*) AS total_policies
                FROM policies
                WHERE LOWER(policy_type)
                    = '{policy_type}';
                """,
                "specific_policy_count",
            )


    # No local match
    return None


# ==========================================================
# RESULT HELPERS
# ==========================================================

def format_number(
    value: Any,
) -> str:

    if value is None:
        return "0"

    if isinstance(
        value,
        Decimal,
    ):
        value = float(value)

    if isinstance(
        value,
        float,
    ):

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    if isinstance(
        value,
        int,
    ):
        return f"{value:,}"

    return str(value)


def first_row(
    results: list[dict],
) -> dict:

    if not results:
        return {}

    return results[0]


# ==========================================================
# LOCAL ANSWER GENERATION
# ==========================================================

def generate_local_answer(
    query_type: str,
    results: list[dict],
) -> str:

    if not results:
        return (
            "The query ran successfully, "
            "but no matching records were found."
        )


    row = first_row(
        results
    )


    if query_type == "total_customers":

        return (
            f"We currently have "
            f"**{format_number(row.get('total_customers'))} customers**."
        )


    if query_type == "total_policies":

        return (
            f"We currently have "
            f"**{format_number(row.get('total_policies'))} policies**."
        )


    if query_type == "total_claims":

        return (
            f"We currently have "
            f"**{format_number(row.get('total_claims'))} claims**."
        )


    if query_type == "total_agents":

        return (
            f"We currently have "
            f"**{format_number(row.get('total_agents'))} agents**."
        )


    if query_type == "total_premium_transactions":

        return (
            f"The database contains "
            f"**{format_number(row.get('total_premium_transactions'))} "
            f"premium transactions**."
        )


    if query_type == "total_renewal_records":

        return (
            f"The database contains "
            f"**{format_number(row.get('total_renewal_records'))} "
            f"renewal records**."
        )


    if query_type == "total_underwriting_records":

        return (
            f"The database contains "
            f"**{format_number(row.get('total_underwriting_records'))} "
            f"underwriting records**."
        )


    if query_type == "average_premium":

        return (
            "The average annual premium is "
            f"**₹{format_number(row.get('average_annual_premium'))}**."
        )


    if query_type == "total_premium":

        return (
            "The total annual premium is "
            f"**₹{format_number(row.get('total_annual_premium'))}**."
        )


    if query_type == "top_customer_state":

        return (
            f"**{row.get('state')}** has the most customers, "
            f"with **{format_number(row.get('total_customers'))} customers**."
        )


    if query_type == "customers_by_state":

        return (
            "### Customer Distribution by State\n\n"
            + generate_result_summary(results)
        )


    if query_type == "policy_types":

        policy_types = [
            str(item.get("policy_type"))
            for item in results
            if item.get("policy_type")
        ]

        return (
            "The policy types available in the database are: "
            + ", ".join(policy_types)
            + "."
        )


    if query_type == "policies_by_type":

        return (
            "### Policy Distribution by Type\n\n"
            + generate_result_summary(results)
        )


    if query_type == "highest_average_premium_policy_type":

        return (
            f"**{row.get('policy_type')}** has the highest "
            f"average annual premium at "
            f"**₹{format_number(row.get('average_annual_premium'))}**."
        )


    if query_type == "highest_premium_policy_type":

        return (
            f"**{row.get('policy_type')}** generates the highest "
            f"total annual premium at "
            f"**₹{format_number(row.get('total_annual_premium'))}**."
        )


    if query_type == "average_premium_by_policy_type":

        return (
            "### Average Annual Premium by Policy Type\n\n"
            + generate_result_summary(results)
        )


    if query_type == "premium_by_policy_type":

        return (
            "### Total Annual Premium by Policy Type\n\n"
            + generate_result_summary(results)
        )


    if query_type == "total_claim_amount":

        return (
            "The total claim amount is "
            f"**₹{format_number(row.get('total_claim_amount'))}**."
        )


    if query_type == "average_claim_amount":

        return (
            "The average claim amount is "
            f"**₹{format_number(row.get('average_claim_amount'))}**."
        )


    if query_type == "claims_by_type":

        return (
            "### Claims by Type\n\n"
            + generate_result_summary(results)
        )


    if query_type == "highest_claim_type":

        return (
            f"**{row.get('claim_type')}** has the highest total "
            f"claim amount at "
            f"**₹{format_number(row.get('total_claim_amount'))}**."
        )


    if query_type == "claims_by_severity":

        return (
            "### Claims by Severity\n\n"
            + generate_result_summary(results)
        )


    if query_type == "specific_policy_count":

        return (
            f"The database contains "
            f"**{format_number(row.get('total_policies'))} matching policies**."
        )


    return (
        f"The query returned "
        f"**{len(results)} row(s)**."
    )


# ==========================================================
# OLLAMA SQL GENERATION
# ==========================================================

def generate_sql(
    question: str,
) -> str:
    """
    Used only when the question is not covered
    by our fast local SQL rules.
    """

    logger.info(
        "Starting natural-language to SQL generation"
    )

    schema = get_database_schema()


    prompt = f"""
/no_think

You are the SQL engine for an Insurance AI Copilot.

Convert the user's question into ONE safe PostgreSQL SELECT query.

DATABASE SCHEMA
---------------
{schema}


USER QUESTION
-------------
{question}


RULES
-----

1. Use only tables and columns present in the schema.

2. PostgreSQL syntax only.

3. Generate only a SELECT query or a WITH ... SELECT query.

4. Never generate:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   CREATE
   TRUNCATE

5. Do not invent columns.

6. Use JOINs only when required.

7. Use meaningful aliases.

8. For ranking questions, use ORDER BY and LIMIT.

9. For aggregate questions, use appropriate GROUP BY.

10. Return SQL only.

11. Do not use markdown.

12. Do not include ```sql.

13. Do not explain the query.

SQL:
"""


    response = llm.invoke(
        prompt
    )


    sql = clean_sql(
        response.content
    )


    validate_sql(
        sql
    )


    logger.info(
        "Generated SQL: %s",
        sql,
    )


    return sql


# ==========================================================
# FALLBACK RESULT SUMMARY
# ==========================================================

def generate_result_summary(
    results: list[dict],
) -> str:
    """
    Create a readable answer from SQL results locally.

    No second Ollama call is required.
    """

    if not results:

        return (
            "The query executed successfully, "
            "but no matching records were found."
        )

    # ======================================================
    # SINGLE ROW
    # ======================================================

    if len(results) == 1:

        row = results[0]

        parts = []

        for key, value in row.items():

            readable_key = (
                key
                .replace("_", " ")
                .title()
            )

            if (
                "premium" in key.lower()
                or "amount" in key.lower()
            ):

                formatted_value = (
                    f"₹{format_number(value)}"
                )

            else:

                formatted_value = (
                    format_number(value)
                )

            parts.append(
                f"**{readable_key}:** "
                f"{formatted_value}"
            )

        return "\n\n".join(parts)

    # ======================================================
    # MULTIPLE ROWS -> MARKDOWN TABLE
    # ======================================================

    columns = list(
        results[0].keys()
    )

    headers = [
        column
        .replace("_", " ")
        .title()
        for column in columns
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            ["---"] * len(columns)
        )
        + " |",
    ]

    for row in results[:20]:

        values = []

        for column in columns:

            value = row.get(
                column
            )

            if (
                "premium" in column.lower()
                or "amount" in column.lower()
            ):

                formatted = (
                    f"₹{format_number(value)}"
                )

            else:

                formatted = (
                    format_number(value)
                )

            values.append(
                formatted
            )

        lines.append(
            "| "
            + " | ".join(values)
            + " |"
        )

    answer = "\n".join(
        lines
    )

    if len(results) > 20:

        answer += (
            f"\n\nShowing the first **20** "
            f"of **{len(results)}** results."
        )

    return answer


# ==========================================================
# MAIN SQL AGENT
# ==========================================================

def ask_sql_agent(
    question: str,
) -> dict[str, Any]:

    logger.info(
        "SQL Agent request received"
    )


    # ======================================================
    # STEP 1 — TRY LOCAL SQL
    # ======================================================

    local_match = get_local_sql(
        question
    )


    if local_match:

        sql, query_type = local_match

        sql = clean_sql(
            sql
        )

        validate_sql(
            sql
        )


        logger.info(
            "Using local SQL rule: %s",
            query_type,
        )


        results = run_query(
            sql
        )


        answer = generate_local_answer(
            query_type,
            results,
        )


        return {

            "question":
                question,

            "sql":
                sql,

            "results":
                results,

            "answer":
                answer,

            "source":
                "local_sql_rule",
        }


    # ======================================================
    # STEP 2 — OLLAMA FALLBACK
    # ======================================================

    logger.info(
        "No local SQL rule matched. Using Ollama."
    )


    sql = generate_sql(
        question
    )


    results = run_query(
        sql
    )


    answer = generate_result_summary(
        results
    )


    return {

        "question":
            question,

        "sql":
            sql,

        "results":
            results,

        "answer":
            answer,

        "source":
            "ollama_sql_generation",
    }