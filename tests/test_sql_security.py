# tests/test_sql_security.py

import pytest

from src.data.query_service import (
    validate_select_query,
    apply_row_limit
)


# --------------------------------------------------
# Test 1:
# Normal SELECT should be allowed
# --------------------------------------------------

def test_valid_select_query():

    query = """
    SELECT customer_id, age
    FROM customers
    """

    result = validate_select_query(query)

    assert result.startswith("SELECT")


# --------------------------------------------------
# Test 2:
# WITH / CTE query should be allowed
# --------------------------------------------------

def test_valid_with_query():

    query = """
    WITH customer_summary AS (
        SELECT state, COUNT(*) AS total
        FROM customers
        GROUP BY state
    )
    SELECT *
    FROM customer_summary
    """

    result = validate_select_query(query)

    assert result.startswith("WITH")


# --------------------------------------------------
# Test 3:
# Dangerous SQL commands must be blocked
# --------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM customers",
        "UPDATE customers SET age = 30",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN test TEXT",
        "TRUNCATE TABLE customers",
        "CREATE TABLE test(id INT)",
        "INSERT INTO customers(customer_id) VALUES ('1')"
    ]
)
def test_dangerous_commands_are_blocked(query):

    with pytest.raises(ValueError):

        validate_select_query(query)


# --------------------------------------------------
# Test 4:
# Dangerous command hidden inside SELECT must fail
# --------------------------------------------------

def test_hidden_delete_is_blocked():

    query = """
    SELECT *
    FROM customers;
    DELETE FROM customers
    """

    with pytest.raises(ValueError):

        validate_select_query(query)


# --------------------------------------------------
# Test 5:
# SQL comments must be blocked
# --------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM customers -- comment",
        "SELECT * FROM customers /* comment */"
    ]
)
def test_sql_comments_are_blocked(query):

    with pytest.raises(ValueError):

        validate_select_query(query)


# --------------------------------------------------
# Test 6:
# Dangerous PostgreSQL functions must be blocked
# --------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "SELECT pg_sleep(10)",
        "SELECT pg_read_file('/tmp/file')",
        "SELECT pg_ls_dir('/')"
    ]
)
def test_dangerous_functions_are_blocked(query):

    with pytest.raises(ValueError):

        validate_select_query(query)


# --------------------------------------------------
# Test 7:
# Empty SQL must fail
# --------------------------------------------------

def test_empty_query_is_blocked():

    with pytest.raises(ValueError):

        validate_select_query("")


# --------------------------------------------------
# Test 8:
# Final semicolon should be safely removed
# --------------------------------------------------

def test_final_semicolon_is_removed():

    query = "SELECT * FROM customers;"

    result = validate_select_query(query)

    assert result == "SELECT * FROM customers"


# --------------------------------------------------
# Test 9:
# Row-limit wrapper should be applied
# --------------------------------------------------

def test_row_limit_is_applied():

    query = "SELECT * FROM customers"

    limited_query = apply_row_limit(query)

    assert "LIMIT :max_rows" in limited_query

    assert query in limited_query