# api/main.py

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

# Import Agents
from src.agents.sql_agent import ask_sql_agent
from src.agents.ml_agent import ask_ml_agent
from src.agents.router import ask_copilot

# Import prediction functions
from src.models.underwriting_predictor import predict_underwriting
from src.models.fraud_predictor import predict_fraud
from src.models.renewal_predictor import predict_renewal


# ---------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------

app = FastAPI(
    title="Insurance AI Copilot API",
    version="1.0.0",
    description="ML + SQL + LangGraph Agent APIs for Insurance AI Copilot"
)


# ---------------------------------------------------
# Health Endpoints
# ---------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Insurance AI Copilot API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ---------------------------------------------------
# Underwriting Request Schema
# ---------------------------------------------------

class UnderwritingRequest(BaseModel):

    AGE: int
    HEALTH_SCORE: float
    BMI: float
    CREDIT_SCORE: int

    LIFESTYLE: str
    MEDICAL_HISTORY_FLAG: str
    SMOKER_FLAG: str
    OCCUPATION_RISK: str


# ---------------------------------------------------
# Fraud Request Schema
# ---------------------------------------------------

class FraudRequest(BaseModel):

    # Numeric features
    CLAIM_AMOUNT: float
    REPORTING_DELAY_DAYS: float
    INCIDENT_MONTH: int

    AGE: int
    ANNUAL_INCOME: float
    CREDIT_SCORE: int

    SUM_INSURED: float
    ANNUAL_PREMIUM: float
    RISK_SCORE: float

    # Categorical features
    CLAIM_TYPE: str
    SOURCE: str
    CLAIM_SEVERITY: str

    GENDER: str
    MARITAL_STATUS: str
    OCCUPATION: str
    STATE: str

    CUSTOMER_RISK_SEGMENT: str
    POLICY_TYPE: str
    PAYMENT_MODE: str
    RISK_BAND: str


# ---------------------------------------------------
# Renewal Request Schema
# ---------------------------------------------------

class RenewalRequest(BaseModel):

    # Numerical features
    AGE: int
    ANNUAL_INCOME: float
    CREDIT_SCORE: int

    SUM_INSURED: float
    ANNUAL_PREMIUM: float
    RISK_SCORE: float

    PREMIUM_INCREASE_PCT: float
    CUSTOMER_TENURE_YEARS: float

    TOTAL_PAYMENTS: float
    AVG_PAYMENT_DELAY: float
    MAX_PAYMENT_DELAY: float

    TOTAL_CLAIMS: float
    TOTAL_CLAIM_AMOUNT: float
    AVG_CLAIM_AMOUNT: float
    MAX_CLAIM_AMOUNT: float

    # Categorical features
    GENDER: str
    MARITAL_STATUS: str
    OCCUPATION: str
    STATE: str

    CUSTOMER_RISK_SEGMENT: str
    POLICY_TYPE: str
    SALES_CHANNEL: str
    PAYMENT_MODE: str
    RISK_BAND: str


# ---------------------------------------------------
# SQL Agent Request Schema
# ---------------------------------------------------

class SQLAgentRequest(BaseModel):

    # Natural-language database question
    question: str


# ---------------------------------------------------
# ML Agent Request Schema
# ---------------------------------------------------

class MLAgentRequest(BaseModel):

    # Natural-language ML request
    question: str

    # Input features required by selected ML model
    input_data: dict[str, Any]


# ---------------------------------------------------
# Unified Copilot Request Schema
# ---------------------------------------------------

class CopilotRequest(BaseModel):

    # Natural-language user question
    question: str

    # Optional ML input features
    input_data: dict[str, Any] | None = None


# ---------------------------------------------------
# Underwriting Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict/underwriting")
def underwriting_prediction(request: UnderwritingRequest):

    applicant_data = request.model_dump()

    result = predict_underwriting(applicant_data)

    return result


# ---------------------------------------------------
# Fraud Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict/fraud")
def fraud_prediction(request: FraudRequest):

    claim_data = request.model_dump()

    result = predict_fraud(claim_data)

    return result


# ---------------------------------------------------
# Renewal Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict/renewal")
def renewal_prediction(request: RenewalRequest):

    customer_data = request.model_dump()

    result = predict_renewal(customer_data)

    return result


# ---------------------------------------------------
# SQL Agent Endpoint
# ---------------------------------------------------

@app.post("/ask/sql")
def sql_agent_endpoint(request: SQLAgentRequest):

    result = ask_sql_agent(
        request.question
    )

    return result


# ---------------------------------------------------
# ML Agent Endpoint
# ---------------------------------------------------

@app.post("/ask/ml")
def ml_agent_endpoint(request: MLAgentRequest):

    result = ask_ml_agent(
        question=request.question,
        input_data=request.input_data
    )

    return result


# ---------------------------------------------------
# Unified LangGraph Copilot Endpoint
# ---------------------------------------------------

@app.post("/ask/copilot")
def copilot_endpoint(request: CopilotRequest):

    # LangGraph automatically decides
    # whether SQL Agent or ML Agent is required
    result = ask_copilot(
        question=request.question,
        input_data=request.input_data
    )

    return result