# src/agents/ml_agent.py

# Shared NVIDIA Nemotron model through OpenRouter
from src.utils.llm import llm

# Import trained ML prediction functions
from src.models.renewal_predictor import predict_renewal
from src.models.fraud_predictor import predict_fraud
from src.models.underwriting_predictor import predict_underwriting


# --------------------------------------------------
# Required features for each ML model
# --------------------------------------------------

RENEWAL_FEATURES = [
    "AGE",
    "ANNUAL_INCOME",
    "CREDIT_SCORE",
    "SUM_INSURED",
    "ANNUAL_PREMIUM",
    "RISK_SCORE",
    "PREMIUM_INCREASE_PCT",
    "CUSTOMER_TENURE_YEARS",
    "TOTAL_PAYMENTS",
    "AVG_PAYMENT_DELAY",
    "MAX_PAYMENT_DELAY",
    "TOTAL_CLAIMS",
    "TOTAL_CLAIM_AMOUNT",
    "AVG_CLAIM_AMOUNT",
    "MAX_CLAIM_AMOUNT",
    "GENDER",
    "MARITAL_STATUS",
    "OCCUPATION",
    "STATE",
    "CUSTOMER_RISK_SEGMENT",
    "POLICY_TYPE",
    "SALES_CHANNEL",
    "PAYMENT_MODE",
    "RISK_BAND",
]


FRAUD_FEATURES = [
    "CLAIM_AMOUNT",
    "REPORTING_DELAY_DAYS",
    "INCIDENT_MONTH",
    "AGE",
    "ANNUAL_INCOME",
    "CREDIT_SCORE",
    "SUM_INSURED",
    "ANNUAL_PREMIUM",
    "RISK_SCORE",
    "CLAIM_TYPE",
    "SOURCE",
    "CLAIM_SEVERITY",
    "GENDER",
    "MARITAL_STATUS",
    "OCCUPATION",
    "STATE",
    "CUSTOMER_RISK_SEGMENT",
    "POLICY_TYPE",
    "PAYMENT_MODE",
    "RISK_BAND",
]


UNDERWRITING_FEATURES = [
    "AGE",
    "HEALTH_SCORE",
    "BMI",
    "CREDIT_SCORE",
    "LIFESTYLE",
    "MEDICAL_HISTORY_FLAG",
    "SMOKER_FLAG",
    "OCCUPATION_RISK",
]


# --------------------------------------------------
# Detect which ML model should be used
# --------------------------------------------------

def detect_ml_task(question: str) -> str:
    """
    Detect whether the user wants:
    renewal, fraud, underwriting, or unknown.
    """

    prompt = f"""
You are routing an insurance machine-learning request.

Choose exactly ONE category:

renewal
fraud
underwriting
unknown

Definitions:

renewal:
Questions about policy renewal, customer churn,
non-renewal risk, or renewal probability.

fraud:
Questions about suspicious claims,
fraud risk, or claim fraud probability.

underwriting:
Questions about insurance application assessment,
underwriting recommendation, applicant risk,
approval, decline, or premium loading.

unknown:
Any request that does not belong to
renewal, fraud, or underwriting.

Return ONLY one word.

USER QUESTION:
{question}
"""

    # Ask NVIDIA Nemotron through OpenRouter
    response = llm.invoke(prompt)

    task = response.content.strip().lower()

    # Deterministic cleanup
    if "renewal" in task:
        return "renewal"

    if "fraud" in task:
        return "fraud"

    if "underwriting" in task:
        return "underwriting"

    return "unknown"


# --------------------------------------------------
# Validate model features
# --------------------------------------------------

def validate_features(
    input_data: dict,
    required_features: list
) -> list:
    """
    Find features required by the model
    but missing from input_data.
    """

    missing_features = [
        feature
        for feature in required_features
        if feature not in input_data
    ]

    return missing_features


# --------------------------------------------------
# Complete ML Agent
# --------------------------------------------------

def ask_ml_agent(
    question: str,
    input_data: dict
) -> dict:
    """
    Complete flow:

    User question
        -> detect ML task
        -> validate required features
        -> run correct ML model
        -> return prediction
    """

    # Step 1: Detect which model is required
    task = detect_ml_task(question)

    # ------------------------------------------------
    # Renewal Model
    # ------------------------------------------------

    if task == "renewal":

        missing_features = validate_features(
            input_data,
            RENEWAL_FEATURES
        )

        if missing_features:
            return {
                "task": "renewal",
                "error": "Missing required features",
                "missing_features": missing_features,
            }

        prediction = predict_renewal(
            input_data
        )

    # ------------------------------------------------
    # Fraud Model
    # ------------------------------------------------

    elif task == "fraud":

        missing_features = validate_features(
            input_data,
            FRAUD_FEATURES
        )

        if missing_features:
            return {
                "task": "fraud",
                "error": "Missing required features",
                "missing_features": missing_features,
            }

        prediction = predict_fraud(
            input_data
        )

    # ------------------------------------------------
    # Underwriting Model
    # ------------------------------------------------

    elif task == "underwriting":

        missing_features = validate_features(
            input_data,
            UNDERWRITING_FEATURES
        )

        if missing_features:
            return {
                "task": "underwriting",
                "error": "Missing required features",
                "missing_features": missing_features,
            }

        prediction = predict_underwriting(
            input_data
        )

    # ------------------------------------------------
    # Unknown Request
    # ------------------------------------------------

    else:

        return {
            "task": "unknown",
            "error": (
                "Could not determine which ML model "
                "should handle this request."
            ),
        }

    # ------------------------------------------------
    # Final Response
    # ------------------------------------------------

    return {
        "task": task,
        "question": question,
        "prediction": prediction,
    }


# --------------------------------------------------
# Quick Test
# --------------------------------------------------

if __name__ == "__main__":

    # Uppercase names are used because these variables
    # exist at module scope during this quick test.
    TEST_QUESTION = (
        "Evaluate this applicant for underwriting"
    )

    SAMPLE_DATA = {
        "AGE": 37,
        "HEALTH_SCORE": 63,
        "BMI": 26.0,
        "CREDIT_SCORE": 789,
        "LIFESTYLE": "Good",
        "MEDICAL_HISTORY_FLAG": "No",
        "SMOKER_FLAG": "No",
        "OCCUPATION_RISK": "Medium",
    }

    TEST_RESULT = ask_ml_agent(
        question=TEST_QUESTION,
        input_data=SAMPLE_DATA
    )

    print("\nML Agent Result:")
    print(TEST_RESULT)