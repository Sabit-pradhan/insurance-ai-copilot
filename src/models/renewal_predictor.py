# src/models/renewal_predictor.py

from pathlib import Path

import joblib
import pandas as pd


# Find project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder containing saved models
MODEL_DIR = PROJECT_ROOT / "models"


# Load model once when this file starts
renewal_model = joblib.load(
    MODEL_DIR / "renewal_xgboost_model.joblib"
)

# Load business threshold
renewal_threshold = joblib.load(
    MODEL_DIR / "renewal_churn_threshold.joblib"
)


def predict_renewal(customer_data: dict):
    """
    Predict renewal/churn probability for one policy/customer.

    customer_data:
        Dictionary containing all features expected by the model.
    """

    # Convert one customer dictionary into a DataFrame
    input_df = pd.DataFrame([customer_data])

    # Model predicts probability of churn
    churn_probability = renewal_model.predict_proba(
        input_df
    )[0, 1]

    # Apply our selected business threshold
    churn_prediction = int(
        churn_probability >= renewal_threshold
    )

    # Convert churn probability into renewal probability
    renewal_probability = 1 - churn_probability

    # Return chatbot/API-friendly result
    return {
        "renewal_probability": round(
            float(renewal_probability), 4
        ),
        "churn_probability": round(
            float(churn_probability), 4
        ),
        "prediction": (
            "High Risk of Non-Renewal"
            if churn_prediction == 1
            else "Likely to Renew"
        ),
        "threshold": round(
            float(renewal_threshold), 4
        )
    }