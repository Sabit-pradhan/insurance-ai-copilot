# src/models/fraud_predictor.py

from pathlib import Path
import joblib
import pandas as pd


# Project and model paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


# Load trained fraud model
fraud_model = joblib.load(
    MODEL_DIR / "fraud_xgboost_model.joblib"
)

# Load selected fraud threshold
fraud_threshold = joblib.load(
    MODEL_DIR / "fraud_threshold.joblib"
)


def predict_fraud(claim_data: dict):
    """
    Predict fraud probability for one insurance claim.
    """

    # Convert dictionary into one-row DataFrame
    input_df = pd.DataFrame([claim_data])

    # Probability of fraud class = 1
    fraud_probability = fraud_model.predict_proba(
        input_df
    )[0, 1]

    # Apply business threshold
    fraud_prediction = int(
        fraud_probability >= fraud_threshold
    )

    return {
        "fraud_probability": round(
            float(fraud_probability), 4
        ),
        "prediction": (
            "Fraud Risk"
            if fraud_prediction == 1
            else "Normal Claim"
        ),
        "threshold": round(
            float(fraud_threshold), 4
        )
    }