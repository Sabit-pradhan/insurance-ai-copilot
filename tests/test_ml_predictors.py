# tests/test_ml_predictors.py

import numpy as np

from src.models import renewal_predictor
from src.models import fraud_predictor
from src.models import underwriting_predictor


# ==================================================
# Fake Models
# ==================================================

class FakeBinaryModel:
    """
    Fake binary classification model.

    probability = probability of class 1
    """

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, X):

        return np.array(
            [
                [
                    1 - self.probability,
                    self.probability
                ]
            ]
        )


class FakeUnderwritingModel:

    def predict(self, X):

        # Encoded class 1
        return np.array([1])

    def predict_proba(self, X):

        return np.array(
            [
                [
                    0.20,
                    0.70,
                    0.10
                ]
            ]
        )


class FakeLabelEncoder:

    classes_ = np.array(
        [
            "Approved",
            "Approved with Loading",
            "Declined"
        ]
    )

    def inverse_transform(self, values):

        mapping = {
            0: "Approved",
            1: "Approved with Loading",
            2: "Declined"
        }

        return np.array(
            [
                mapping[value]
                for value in values
            ]
        )


# ==================================================
# Renewal Tests
# ==================================================

def test_renewal_high_churn_risk(monkeypatch):

    monkeypatch.setattr(
        renewal_predictor,
        "renewal_model",
        FakeBinaryModel(0.80)
    )

    monkeypatch.setattr(
        renewal_predictor,
        "renewal_threshold",
        0.50
    )

    result = renewal_predictor.predict_renewal(
        {
            "AGE": 40
        }
    )

    assert result["churn_probability"] == 0.8

    assert result["renewal_probability"] == 0.2

    assert (
        result["prediction"]
        == "High Risk of Non-Renewal"
    )


def test_renewal_likely_to_renew(monkeypatch):

    monkeypatch.setattr(
        renewal_predictor,
        "renewal_model",
        FakeBinaryModel(0.20)
    )

    monkeypatch.setattr(
        renewal_predictor,
        "renewal_threshold",
        0.50
    )

    result = renewal_predictor.predict_renewal(
        {
            "AGE": 35
        }
    )

    assert result["churn_probability"] == 0.2

    assert result["renewal_probability"] == 0.8

    assert (
        result["prediction"]
        == "Likely to Renew"
    )


# ==================================================
# Fraud Tests
# ==================================================

def test_fraud_high_risk(monkeypatch):

    monkeypatch.setattr(
        fraud_predictor,
        "fraud_model",
        FakeBinaryModel(0.85)
    )

    monkeypatch.setattr(
        fraud_predictor,
        "fraud_threshold",
        0.60
    )

    result = fraud_predictor.predict_fraud(
        {
            "CLAIM_AMOUNT": 100000
        }
    )

    assert result["fraud_probability"] == 0.85

    assert (
        result["prediction"]
        == "Fraud Risk"
    )


def test_fraud_normal_claim(monkeypatch):

    monkeypatch.setattr(
        fraud_predictor,
        "fraud_model",
        FakeBinaryModel(0.20)
    )

    monkeypatch.setattr(
        fraud_predictor,
        "fraud_threshold",
        0.60
    )

    result = fraud_predictor.predict_fraud(
        {
            "CLAIM_AMOUNT": 10000
        }
    )

    assert result["fraud_probability"] == 0.2

    assert (
        result["prediction"]
        == "Normal Claim"
    )


# ==================================================
# Underwriting Tests
# ==================================================

def test_underwriting_decision(monkeypatch):

    monkeypatch.setattr(
        underwriting_predictor,
        "underwriting_model",
        FakeUnderwritingModel()
    )

    monkeypatch.setattr(
        underwriting_predictor,
        "label_encoder",
        FakeLabelEncoder()
    )

    result = (
        underwriting_predictor
        .predict_underwriting(
            {
                "AGE": 35,
                "HEALTH_SCORE": 80
            }
        )
    )

    assert (
        result["decision"]
        == "Approved with Loading"
    )


def test_underwriting_probabilities(monkeypatch):

    monkeypatch.setattr(
        underwriting_predictor,
        "underwriting_model",
        FakeUnderwritingModel()
    )

    monkeypatch.setattr(
        underwriting_predictor,
        "label_encoder",
        FakeLabelEncoder()
    )

    result = (
        underwriting_predictor
        .predict_underwriting(
            {
                "AGE": 35
            }
        )
    )

    probabilities = result[
        "probabilities"
    ]

    assert probabilities["Approved"] == 0.20

    assert (
        probabilities[
            "Approved with Loading"
        ]
        == 0.70
    )

    assert probabilities["Declined"] == 0.10