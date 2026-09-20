# tests/test_api.py

from fastapi.testclient import TestClient

import api.main as api_main


# --------------------------------------------------
# Test Client
# --------------------------------------------------

client = TestClient(api_main.app)


# --------------------------------------------------
# Test 1:
# Root endpoint
# --------------------------------------------------

def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Insurance AI Copilot API is running"
    }


# --------------------------------------------------
# Test 2:
# Health endpoint
# --------------------------------------------------

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


# --------------------------------------------------
# Test 3:
# Copilot endpoint using mocked agent
# --------------------------------------------------

def test_copilot_endpoint(monkeypatch):

    # Fake Copilot response.
    # This prevents real LLM / DB / RAG API calls.
    fake_result = {
        "route": "rag",
        "result": {
            "status": "success",
            "agent": "rag",
            "answer": "Test answer",
            "sources": []
        }
    }

    monkeypatch.setattr(
        api_main,
        "ask_copilot",
        lambda question, input_data=None: fake_result
    )

    response = client.post(
        "/ask/copilot",
        json={
            "question": "What documents are required?",
            "input_data": None
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["route"] == "rag"

    assert (
        data["result"]["status"]
        == "success"
    )

    assert (
        data["result"]["answer"]
        == "Test answer"
    )


# --------------------------------------------------
# Test 4:
# Invalid Copilot request
# --------------------------------------------------

def test_copilot_missing_question():

    response = client.post(
        "/ask/copilot",
        json={
            "input_data": None
        }
    )

    # Pydantic validation error
    assert response.status_code == 422


# --------------------------------------------------
# Test 5:
# Underwriting validation
# --------------------------------------------------

def test_underwriting_invalid_request():

    response = client.post(
        "/predict/underwriting",
        json={
            "AGE": 35
        }
    )

    # Required fields are missing
    assert response.status_code == 422