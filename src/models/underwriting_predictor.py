# src/models/underwriting_predictor.py

from pathlib import Path
import joblib
import pandas as pd

# Define model path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

# Load model and encoder
underwriting_model = joblib.load(
    MODEL_DIR / "underwriting_xgboost_model.joblib"
)

label_encoder = joblib.load(
    MODEL_DIR / "underwriting_label_encoder.joblib"
)


def predict_underwriting(applicant_data: dict):

    # Convert input into one-row DataFrame
    input_df = pd.DataFrame([applicant_data])

    # Predict encoded class
    encoded_prediction = underwriting_model.predict(
        input_df
    )[0]

    # Convert encoded class back to label
    decision = label_encoder.inverse_transform(
        [int(encoded_prediction)]
    )[0]

    # Predict probabilities
    probabilities = underwriting_model.predict_proba(
        input_df
    )[0]

    class_probabilities = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(
            label_encoder.classes_,
            probabilities
        )
    }

    return {
        "decision": decision,
        "probabilities": class_probabilities
    }