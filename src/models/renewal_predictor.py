from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


MODEL_PATH = MODEL_DIR / "renewal_v3_xgboost_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "renewal_v3_churn_threshold.joblib"
FEATURES_PATH = MODEL_DIR / "renewal_v3_features.joblib"


class RenewalPredictor:
    def __init__(self):
        print("Loading Renewal V3 model...")

        self.model = joblib.load(MODEL_PATH)
        self.threshold = float(
            joblib.load(THRESHOLD_PATH)
        )

        feature_config = joblib.load(
            FEATURES_PATH
        )

        self.numeric_features = feature_config[
            "numeric_features"
        ]

        self.categorical_features = feature_config[
            "categorical_features"
        ]

        self.feature_columns = feature_config[
            "feature_columns"
        ]

        print(
            f"Renewal V3 model loaded successfully."
        )

        print(
            f"Churn threshold: {self.threshold:.4f}"
        )

    def predict(self, features: dict) -> dict:

        # Create DataFrame with exactly the
        # same feature order used during training.
        df = pd.DataFrame(
            [features],
            columns=self.feature_columns,
        )

        probability = float(
            self.model.predict_proba(df)[0][1]
        )

        churn = (
            probability >= self.threshold
        )

        if churn:
            prediction = "Churn"
        else:
            prediction = "Renewed"

        return {
            "churn_probability": round(
                probability,
                4,
            ),
            "renewal_probability": round(
                1 - probability,
                4,
            ),
            "prediction": prediction,
            "churn": bool(churn),
            "threshold": round(
                self.threshold,
                4,
            ),
            "model_version": "V3",
        }


_predictor = None


def get_predictor():
    global _predictor

    if _predictor is None:
        _predictor = RenewalPredictor()

    return _predictor


def predict_renewal(features: dict) -> dict:
    predictor = get_predictor()
    return predictor.predict(features)