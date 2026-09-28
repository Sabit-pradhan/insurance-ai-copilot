from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ==========================================================
# EXPLAINABILITY
# ==========================================================

from src.explainability.renewal_explainer import explain_renewal
from src.explainability.fraud_explainer import explain_fraud
from src.explainability.underwriting_explainer import explain_underwriting


# ==========================================================
# AGENTS
# ==========================================================

from src.agents.sql_agent import ask_sql_agent
from src.agents.ml_agent import ask_ml_agent

from src.agents.router import (
    ask_copilot,
    clear_copilot_session,
)


# ==========================================================
# ML PREDICTORS
# ==========================================================

from src.models.underwriting_predictor import predict_underwriting
from src.models.fraud_predictor import predict_fraud
from src.models.renewal_predictor import predict_renewal


# ==========================================================
# DATABASE FEATURE SERVICES
# ==========================================================

from src.data.fraud_feature_service import (
    get_fraud_case_by_claim_id,
)

from src.data.renewal_feature_service import (
    get_renewal_case_by_policy_id,
)

from src.data.underwriting_feature_service import (
    get_underwriting_case_by_id,
)


# ==========================================================
# MONITORING
# ==========================================================

from src.monitoring.prediction_logger import log_prediction

from src.monitoring.monitoring_service import (
    get_monitoring_summary,
    get_model_summary,
    get_recent_predictions,
)

from src.monitoring.outcome_service import (
    save_prediction_outcome,
    get_prediction_outcome,
    get_prediction_outcomes,
    get_outcome_coverage,
)

from src.monitoring.performance_service import (
    get_model_performance,
    get_all_model_performance,
)


# ==========================================================
# MODEL VERSIONS
# ==========================================================

RENEWAL_MODEL_VERSION = "v1.0"
FRAUD_MODEL_VERSION = "v1.0"
UNDERWRITING_MODEL_VERSION = "v1.0"


# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="Insurance AI Copilot API",
    version="1.3.0",
    description=(
        "ML + Explainable AI + Monitoring + SQL + "
        "RAG + LangGraph Insurance AI Copilot API"
    ),
)


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def home():

    return {
        "message":
            "Insurance AI Copilot API is running",

        "version":
            "1.3.0",
    }


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy"
    }


# ==========================================================
# LATENCY HELPER
# ==========================================================

def calculate_latency_ms(
    start_time: float,
) -> float:

    latency_ms = (
        (
            perf_counter()
            -
            start_time
        )
        *
        1000
    )

    return round(
        latency_ms,
        3,
    )


# ==========================================================
# SIMPLE EXPLANATION HELPER
# ==========================================================

def build_simple_explanation(
    explanation: dict,
):

    return {

        "status":
            "success",

        "method":
            explanation.get(
                "explanation_method"
            ),

        "base_value":
            explanation.get(
                "base_value"
            ),

        "top_factors":
            explanation.get(
                "top_factors",
                [],
            ),

        "note":
            explanation.get(
                "note"
            ),
    }


# ==========================================================
# FRAUD LABEL HELPER
# ==========================================================

def get_fraud_prediction_label(
    result: dict,
):

    prediction = (
        result.get(
            "prediction"
        )
    )


    if prediction is not None:

        return prediction


    fraud_probability = (
        result.get(
            "fraud_probability"
        )
    )


    threshold = (
        result.get(
            "threshold"
        )
    )


    if (
        fraud_probability is None
        or
        threshold is None
    ):

        return "Unknown"


    if (
        float(
            fraud_probability
        )
        >=
        float(
            threshold
        )
    ):

        return (
            "Elevated Fraud Risk"
        )


    return (
        "No Elevated Fraud Risk"
    )


# ==========================================================
# RENEWAL LABEL HELPER
# ==========================================================

def get_renewal_prediction_label(
    result: dict,
):

    prediction = (
        result.get(
            "prediction"
        )
    )


    if prediction is not None:

        return prediction


    churn_probability = (
        result.get(
            "churn_probability"
        )
    )


    threshold = (
        result.get(
            "threshold"
        )
    )


    if (
        churn_probability is None
        or
        threshold is None
    ):

        return "Unknown"


    if (
        float(
            churn_probability
        )
        >=
        float(
            threshold
        )
    ):

        return (
            "Elevated Non-Renewal Risk"
        )


    return (
        "Lower Non-Renewal Risk"
    )


# ==========================================================
# UNDERWRITING REQUEST
# ==========================================================

class UnderwritingRequest(
    BaseModel
):

    AGE: int

    HEALTH_SCORE: float

    BMI: float

    CREDIT_SCORE: int

    LIFESTYLE: str

    MEDICAL_HISTORY_FLAG: str

    SMOKER_FLAG: str

    OCCUPATION_RISK: str


# ==========================================================
# FRAUD REQUEST
# ==========================================================

class FraudRequest(
    BaseModel
):

    CLAIM_ID: str | None = None


    CLAIM_AMOUNT: float

    REPORTING_DELAY_DAYS: float

    INCIDENT_MONTH: int

    AGE: int

    ANNUAL_INCOME: float

    CREDIT_SCORE: int

    SUM_INSURED: float

    ANNUAL_PREMIUM: float

    RISK_SCORE: float


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


# ==========================================================
# RENEWAL REQUEST
# ==========================================================

class RenewalRequest(
    BaseModel
):

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


    GENDER: str

    MARITAL_STATUS: str

    OCCUPATION: str

    STATE: str

    CUSTOMER_RISK_SEGMENT: str

    POLICY_TYPE: str

    SALES_CHANNEL: str

    PAYMENT_MODE: str

    RISK_BAND: str


# ==========================================================
# SQL AGENT REQUEST
# ==========================================================

class SQLAgentRequest(
    BaseModel
):

    question: str


# ==========================================================
# ML AGENT REQUEST
# ==========================================================

class MLAgentRequest(
    BaseModel
):

    question: str

    input_data: dict[
        str,
        Any,
    ]


# ==========================================================
# COPILOT REQUEST
# ==========================================================

class CopilotRequest(
    BaseModel
):

    question: str

    input_data: (
        dict[
            str,
            Any,
        ]
        |
        None
    ) = None

    session_id: (
        str
        |
        None
    ) = None


# ==========================================================
# PREDICTION OUTCOME REQUEST
# ==========================================================

class PredictionOutcomeRequest(
    BaseModel
):

    prediction_id: int

    actual_class: str

    outcome_source: (
        str
        |
        None
    ) = None

    notes: (
        str
        |
        None
    ) = None


# ==========================================================
# UNDERWRITING ENDPOINT
# MANUAL FEATURES
# ==========================================================

@app.post(
    "/predict/underwriting"
)
def underwriting_prediction(
    request: UnderwritingRequest,
):

    start_time = (
        perf_counter()
    )


    applicant_data = (
        request.model_dump()
    )


    try:

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        result = (
            predict_underwriting(
                applicant_data
            )
        )


        # --------------------------------------------------
        # SHAP
        # --------------------------------------------------

        try:

            explanation = (
                explain_underwriting(

                    input_data=
                        applicant_data,

                    top_n=
                        5,
                )
            )


            result[
                "explainability"
            ] = {

                "status":
                    "success",

                "method":
                    explanation.get(
                        "explanation_method"
                    ),

                "explained_class":
                    explanation.get(
                        "explained_class"
                    ),

                "predicted_probability":
                    explanation.get(
                        "predicted_probability"
                    ),

                "base_value":
                    explanation.get(
                        "base_value"
                    ),

                "top_factors":
                    explanation.get(
                        "top_factors",
                        [],
                    ),

                "note":
                    explanation.get(
                        "note"
                    ),
            }


        except Exception as error:

            result[
                "explainability"
            ] = {

                "status":
                    "unavailable",

                "error":
                    str(
                        error
                    ),
            }


        # --------------------------------------------------
        # LATENCY
        # --------------------------------------------------

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        # --------------------------------------------------
        # DECISION
        # --------------------------------------------------

        decision = (

            result.get(
                "decision"
            )

            or

            result.get(
                "prediction"
            )

            or

            result.get(
                "predicted_class"
            )

            or

            "Unknown"
        )


        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        probability = None


        probabilities = (

            result.get(
                "probabilities",
                {},
            )

            or

            {}
        )


        if (
            decision
            in
            probabilities
        ):

            probability = (
                probabilities[
                    decision
                ]
            )


        if probability is None:

            probability = (

                result
                .get(
                    "explainability",
                    {},
                )
                .get(
                    "predicted_probability"
                )
            )


        # --------------------------------------------------
        # LOG PREDICTION
        # --------------------------------------------------

        prediction_id = (
            log_prediction(

                model_name=
                    "underwriting",

                model_version=
                    UNDERWRITING_MODEL_VERSION,

                endpoint=
                    "/predict/underwriting",

                predicted_class=
                    str(
                        decision
                    ),

                probability=
                    probability,

                threshold=
                    None,

                feature_snapshot=
                    applicant_data,

                explainability=
                    result.get(
                        "explainability",
                        {},
                    ),

                latency_ms=
                    latency_ms,

                status=
                    "success",
            )
        )


        # --------------------------------------------------
        # MONITORING
        # --------------------------------------------------

        result[
            "monitoring"
        ] = {

            "prediction_id":
                prediction_id,

            "model_version":
                UNDERWRITING_MODEL_VERSION,

            "latency_ms":
                latency_ms,
        }


        return result


    except Exception as error:

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        log_prediction(

            model_name=
                "underwriting",

            model_version=
                UNDERWRITING_MODEL_VERSION,

            endpoint=
                "/predict/underwriting",

            feature_snapshot=
                applicant_data,

            latency_ms=
                latency_ms,

            status=
                "error",

            error_message=
                str(
                    error
                ),
        )


        raise


# ==========================================================
# UNDERWRITING ENDPOINT
# REAL DATABASE FEATURES BY UNDERWRITING ID
# ==========================================================

@app.post(
    "/predict/underwriting/by-id/{underwriting_id}"
)
def underwriting_prediction_by_id(
    underwriting_id: str,
):

    """
    Only UNDERWRITING_ID is supplied.

    Features come from database.

    DECISION is kept separate
    as ground truth.
    """


    start_time = (
        perf_counter()
    )


    # ------------------------------------------------------
    # LOAD REAL DATABASE CASE
    # ------------------------------------------------------

    try:

        underwriting_case = (
            get_underwriting_case_by_id(
                underwriting_id
            )
        )


    except ValueError as error:

        raise HTTPException(

            status_code=
                404,

            detail=
                str(
                    error
                ),
        )


    # ------------------------------------------------------
    # MODEL FEATURES ONLY
    # ------------------------------------------------------

    applicant_data = (
        underwriting_case[
            "model_features"
        ]
    )


    # actual_decision is intentionally
    # NOT passed to the model.
    #
    # It is ground truth.


    try:

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        result = (
            predict_underwriting(
                applicant_data
            )
        )


        # --------------------------------------------------
        # SHAP
        # --------------------------------------------------

        try:

            explanation = (
                explain_underwriting(

                    input_data=
                        applicant_data,

                    top_n=
                        5,
                )
            )


            result[
                "explainability"
            ] = {

                "status":
                    "success",

                "method":
                    explanation.get(
                        "explanation_method"
                    ),

                "explained_class":
                    explanation.get(
                        "explained_class"
                    ),

                "predicted_probability":
                    explanation.get(
                        "predicted_probability"
                    ),

                "base_value":
                    explanation.get(
                        "base_value"
                    ),

                "top_factors":
                    explanation.get(
                        "top_factors",
                        [],
                    ),

                "note":
                    explanation.get(
                        "note"
                    ),
            }


        except Exception as error:

            result[
                "explainability"
            ] = {

                "status":
                    "unavailable",

                "error":
                    str(
                        error
                    ),
            }


        # --------------------------------------------------
        # LATENCY
        # --------------------------------------------------

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        # --------------------------------------------------
        # DECISION
        # --------------------------------------------------

        decision = (

            result.get(
                "decision"
            )

            or

            result.get(
                "prediction"
            )

            or

            result.get(
                "predicted_class"
            )

            or

            "Unknown"
        )


        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        probability = None


        probabilities = (

            result.get(
                "probabilities",
                {},
            )

            or

            {}
        )


        if (
            decision
            in
            probabilities
        ):

            probability = (
                probabilities[
                    decision
                ]
            )


        if probability is None:

            probability = (

                result
                .get(
                    "explainability",
                    {},
                )
                .get(
                    "predicted_probability"
                )
            )


        # --------------------------------------------------
        # FEATURE SNAPSHOT
        # --------------------------------------------------

        feature_snapshot = {

            "UNDERWRITING_ID":
                underwriting_case[
                    "underwriting_id"
                ],

            "POLICY_ID":
                underwriting_case[
                    "policy_id"
                ],

            "CUSTOMER_ID":
                underwriting_case[
                    "customer_id"
                ],

            **applicant_data,
        }


        # --------------------------------------------------
        # LOG PREDICTION
        # --------------------------------------------------

        prediction_id = (
            log_prediction(

                model_name=
                    "underwriting",

                model_version=
                    UNDERWRITING_MODEL_VERSION,

                endpoint=
                    "/predict/underwriting/by-id",

                predicted_class=
                    str(
                        decision
                    ),

                probability=
                    probability,

                threshold=
                    None,

                feature_snapshot=
                    feature_snapshot,

                explainability=
                    result.get(
                        "explainability",
                        {},
                    ),

                latency_ms=
                    latency_ms,

                status=
                    "success",
            )
        )


        # --------------------------------------------------
        # MONITORING
        # --------------------------------------------------

        result[
            "monitoring"
        ] = {

            "prediction_id":
                prediction_id,

            "model_version":
                UNDERWRITING_MODEL_VERSION,

            "underwriting_id":
                underwriting_case[
                    "underwriting_id"
                ],

            "policy_id":
                underwriting_case[
                    "policy_id"
                ],

            "customer_id":
                underwriting_case[
                    "customer_id"
                ],

            "latency_ms":
                latency_ms,

            "feature_source":
                "database",
        }


        return result


    except HTTPException:

        raise


    except Exception as error:

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        log_prediction(

            model_name=
                "underwriting",

            model_version=
                UNDERWRITING_MODEL_VERSION,

            endpoint=
                "/predict/underwriting/by-id",

            predicted_class=
                None,

            probability=
                None,

            threshold=
                None,

            feature_snapshot={

                "UNDERWRITING_ID":
                    underwriting_id,

                **applicant_data,
            },

            explainability=
                None,

            latency_ms=
                latency_ms,

            status=
                "error",

            error_message=
                str(
                    error
                ),
        )


        raise HTTPException(

            status_code=
                500,

            detail=
                str(
                    error
                ),
        )


# ==========================================================
# FRAUD COMMON PIPELINE
# ==========================================================

def run_fraud_prediction(
    claim_data: dict,
    claim_id: str | None,
    endpoint: str,
):

    """
    Common fraud prediction pipeline.

    claim_data contains model features only.

    CLAIM_ID is used for monitoring
    and ground-truth matching.
    """


    start_time = (
        perf_counter()
    )


    try:

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        result = (
            predict_fraud(
                claim_data
            )
        )


        # --------------------------------------------------
        # SHAP
        # --------------------------------------------------

        try:

            explanation = (
                explain_fraud(

                    input_data=
                        claim_data,

                    top_n=
                        5,
                )
            )


            result[
                "explainability"
            ] = (
                build_simple_explanation(
                    explanation
                )
            )


        except Exception as error:

            result[
                "explainability"
            ] = {

                "status":
                    "unavailable",

                "error":
                    str(
                        error
                    ),
            }


        # --------------------------------------------------
        # LATENCY
        # --------------------------------------------------

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        # --------------------------------------------------
        # VALUES
        # --------------------------------------------------

        fraud_probability = (
            result.get(
                "fraud_probability"
            )
        )


        threshold = (
            result.get(
                "threshold"
            )
        )


        prediction = (
            get_fraud_prediction_label(
                result
            )
        )


        # --------------------------------------------------
        # LOG
        # --------------------------------------------------

        prediction_id = (
            log_prediction(

                model_name=
                    "fraud",

                model_version=
                    FRAUD_MODEL_VERSION,

                endpoint=
                    endpoint,

                predicted_class=
                    str(
                        prediction
                    ),

                probability=
                    fraud_probability,

                threshold=
                    threshold,

                feature_snapshot={

                    "CLAIM_ID":
                        claim_id,

                    **claim_data,
                },

                explainability=
                    result.get(
                        "explainability",
                        {},
                    ),

                latency_ms=
                    latency_ms,

                status=
                    "success",
            )
        )


        result[
            "monitoring"
        ] = {

            "prediction_id":
                prediction_id,

            "model_version":
                FRAUD_MODEL_VERSION,

            "claim_id":
                claim_id,

            "latency_ms":
                latency_ms,
        }


        return result


    except Exception as error:

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        log_prediction(

            model_name=
                "fraud",

            model_version=
                FRAUD_MODEL_VERSION,

            endpoint=
                endpoint,

            feature_snapshot={

                "CLAIM_ID":
                    claim_id,

                **claim_data,
            },

            latency_ms=
                latency_ms,

            status=
                "error",

            error_message=
                str(
                    error
                ),
        )


        raise


# ==========================================================
# FRAUD ENDPOINT
# MANUAL FEATURES
# ==========================================================

@app.post(
    "/predict/fraud"
)
def fraud_prediction(
    request: FraudRequest,
):

    request_data = (
        request.model_dump()
    )


    claim_id = (
        request_data.pop(
            "CLAIM_ID",
            None,
        )
    )


    claim_data = (
        request_data
    )


    return (
        run_fraud_prediction(

            claim_data=
                claim_data,

            claim_id=
                claim_id,

            endpoint=
                "/predict/fraud",
        )
    )


# ==========================================================
# FRAUD ENDPOINT
# REAL DATABASE FEATURES
# ==========================================================

@app.post(
    "/predict/fraud/by-claim/{claim_id}"
)
def fraud_prediction_by_claim(
    claim_id: str,
):

    """
    Only CLAIM_ID is supplied.

    Features come from database.

    fraud_flag is kept separate
    as ground truth.
    """


    try:

        fraud_case = (
            get_fraud_case_by_claim_id(
                claim_id
            )
        )


    except ValueError as error:

        raise HTTPException(

            status_code=
                404,

            detail=
                str(
                    error
                ),
        )


    claim_data = (
        fraud_case[
            "model_features"
        ]
    )


    try:

        result = (
            run_fraud_prediction(

                claim_data=
                    claim_data,

                claim_id=
                    claim_id,

                endpoint=
                    "/predict/fraud/by-claim",
            )
        )


        result[
            "monitoring"
        ][
            "feature_source"
        ] = (
            "database"
        )


        return result


    except HTTPException:

        raise


    except Exception as error:

        raise HTTPException(

            status_code=
                500,

            detail=
                str(
                    error
                ),
        )


# ==========================================================
# RENEWAL ENDPOINT
# MANUAL FEATURES
# ==========================================================

@app.post(
    "/predict/renewal"
)
def renewal_prediction(
    request: RenewalRequest,
):

    start_time = (
        perf_counter()
    )


    customer_data = (
        request.model_dump()
    )


    try:

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        result = (
            predict_renewal(
                customer_data
            )
        )


        # --------------------------------------------------
        # SHAP
        # --------------------------------------------------

        try:

            explanation = (
                explain_renewal(

                    input_data=
                        customer_data,

                    top_n=
                        5,
                )
            )


            result[
                "explainability"
            ] = (
                build_simple_explanation(
                    explanation
                )
            )


        except Exception as error:

            result[
                "explainability"
            ] = {

                "status":
                    "unavailable",

                "error":
                    str(
                        error
                    ),
            }


        # --------------------------------------------------
        # LATENCY
        # --------------------------------------------------

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        # --------------------------------------------------
        # VALUES
        # --------------------------------------------------

        churn_probability = (
            result.get(
                "churn_probability"
            )
        )


        threshold = (
            result.get(
                "threshold"
            )
        )


        prediction = (
            get_renewal_prediction_label(
                result
            )
        )


        # --------------------------------------------------
        # LOG
        # --------------------------------------------------

        prediction_id = (
            log_prediction(

                model_name=
                    "renewal",

                model_version=
                    RENEWAL_MODEL_VERSION,

                endpoint=
                    "/predict/renewal",

                predicted_class=
                    str(
                        prediction
                    ),

                probability=
                    churn_probability,

                threshold=
                    threshold,

                feature_snapshot=
                    customer_data,

                explainability=
                    result.get(
                        "explainability",
                        {},
                    ),

                latency_ms=
                    latency_ms,

                status=
                    "success",
            )
        )


        result[
            "monitoring"
        ] = {

            "prediction_id":
                prediction_id,

            "model_version":
                RENEWAL_MODEL_VERSION,

            "latency_ms":
                latency_ms,
        }


        return result


    except Exception as error:

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        log_prediction(

            model_name=
                "renewal",

            model_version=
                RENEWAL_MODEL_VERSION,

            endpoint=
                "/predict/renewal",

            feature_snapshot=
                customer_data,

            latency_ms=
                latency_ms,

            status=
                "error",

            error_message=
                str(
                    error
                ),
        )


        raise


# ==========================================================
# RENEWAL ENDPOINT
# REAL DATABASE FEATURES BY POLICY ID
# ==========================================================

@app.post(
    "/predict/renewal/by-policy/{policy_id}"
)
def renewal_prediction_by_policy(
    policy_id: str,
):

    """
    Only POLICY_ID is supplied.

    Features are loaded automatically from:

    renewals
    + customers
    + policies
    + premium history
    + claim history

    renewal_status is NOT sent to model.

    It remains ground truth.
    """


    start_time = (
        perf_counter()
    )


    # ------------------------------------------------------
    # LOAD DATABASE FEATURES
    # ------------------------------------------------------

    try:

        renewal_case = (
            get_renewal_case_by_policy_id(
                policy_id
            )
        )


    except ValueError as error:

        raise HTTPException(

            status_code=
                404,

            detail=
                str(
                    error
                ),
        )


    # ------------------------------------------------------
    # MODEL FEATURES ONLY
    # ------------------------------------------------------

    customer_data = (
        renewal_case[
            "model_features"
        ]
    )


    try:

        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        result = (
            predict_renewal(
                customer_data
            )
        )


        # --------------------------------------------------
        # SHAP
        # --------------------------------------------------

        try:

            explanation = (
                explain_renewal(

                    input_data=
                        customer_data,

                    top_n=
                        5,
                )
            )


            result[
                "explainability"
            ] = (
                build_simple_explanation(
                    explanation
                )
            )


        except Exception as error:

            result[
                "explainability"
            ] = {

                "status":
                    "unavailable",

                "error":
                    str(
                        error
                    ),
            }


        # --------------------------------------------------
        # LATENCY
        # --------------------------------------------------

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        # --------------------------------------------------
        # VALUES
        # --------------------------------------------------

        churn_probability = (
            result.get(
                "churn_probability"
            )
        )


        threshold = (
            result.get(
                "threshold"
            )
        )


        prediction = (
            get_renewal_prediction_label(
                result
            )
        )


        # --------------------------------------------------
        # FEATURE SNAPSHOT
        # --------------------------------------------------

        feature_snapshot = {

            "POLICY_ID":
                renewal_case[
                    "policy_id"
                ],

            "RENEWAL_ID":
                renewal_case[
                    "renewal_id"
                ],

            "CUSTOMER_ID":
                renewal_case[
                    "customer_id"
                ],

            **customer_data,
        }


        # --------------------------------------------------
        # LOG PREDICTION
        # --------------------------------------------------

        prediction_id = (
            log_prediction(

                model_name=
                    "renewal",

                model_version=
                    RENEWAL_MODEL_VERSION,

                endpoint=
                    "/predict/renewal/by-policy",

                predicted_class=
                    str(
                        prediction
                    ),

                probability=
                    churn_probability,

                threshold=
                    threshold,

                feature_snapshot=
                    feature_snapshot,

                explainability=
                    result.get(
                        "explainability",
                        {},
                    ),

                latency_ms=
                    latency_ms,

                status=
                    "success",
            )
        )


        # --------------------------------------------------
        # MONITORING
        # --------------------------------------------------

        result[
            "monitoring"
        ] = {

            "prediction_id":
                prediction_id,

            "model_version":
                RENEWAL_MODEL_VERSION,

            "policy_id":
                renewal_case[
                    "policy_id"
                ],

            "renewal_id":
                renewal_case[
                    "renewal_id"
                ],

            "customer_id":
                renewal_case[
                    "customer_id"
                ],

            "latency_ms":
                latency_ms,

            "feature_source":
                "database",
        }


        return result


    except HTTPException:

        raise


    except Exception as error:

        latency_ms = (
            calculate_latency_ms(
                start_time
            )
        )


        log_prediction(

            model_name=
                "renewal",

            model_version=
                RENEWAL_MODEL_VERSION,

            endpoint=
                "/predict/renewal/by-policy",

            predicted_class=
                None,

            probability=
                None,

            threshold=
                None,

            feature_snapshot={

                "POLICY_ID":
                    policy_id,

                **customer_data,
            },

            explainability=
                None,

            latency_ms=
                latency_ms,

            status=
                "error",

            error_message=
                str(
                    error
                ),
        )


        raise HTTPException(

            status_code=
                500,

            detail=
                str(
                    error
                ),
        )


# ==========================================================
# SQL AGENT
# ==========================================================

@app.post(
    "/ask/sql"
)
def sql_agent_endpoint(
    request: SQLAgentRequest,
):

    return (
        ask_sql_agent(
            request.question
        )
    )


# ==========================================================
# ML AGENT
# ==========================================================

@app.post(
    "/ask/ml"
)
def ml_agent_endpoint(
    request: MLAgentRequest,
):

    return (
        ask_ml_agent(

            question=
                request.question,

            input_data=
                request.input_data,
        )
    )


# ==========================================================
# COPILOT
# ==========================================================

@app.post(
    "/ask/copilot"
)
def copilot_endpoint(
    request: CopilotRequest,
):

    return (
        ask_copilot(

            question=
                request.question,

            input_data=
                request.input_data,

            session_id=
                request.session_id,
        )
    )


# ==========================================================
# CLEAR COPILOT SESSION
# ==========================================================

@app.delete(
    "/ask/copilot/session/{session_id}"
)
def clear_copilot_session_endpoint(
    session_id: str,
):

    cleared = (
        clear_copilot_session(
            session_id
        )
    )


    return {

        "session_id":
            session_id,

        "cleared":
            cleared,

        "message": (

            "Conversation cleared."

            if cleared

            else

            (
                "Session had no stored "
                "conversation."
            )
        ),
    }


# ==========================================================
# MONITORING SUMMARY
# ==========================================================

@app.get(
    "/monitoring/summary"
)
def monitoring_summary():

    return (
        get_monitoring_summary()
    )


# ==========================================================
# MODEL-WISE MONITORING
# ==========================================================

@app.get(
    "/monitoring/models"
)
def monitoring_models():

    return (
        get_model_summary()
    )


# ==========================================================
# RECENT PREDICTIONS
# ==========================================================

@app.get(
    "/monitoring/recent"
)
def monitoring_recent(
    limit: int = 20,
):

    limit = max(
        1,
        min(
            limit,
            100,
        ),
    )


    return (
        get_recent_predictions(
            limit=limit
        )
    )


# ==========================================================
# SAVE PREDICTION OUTCOME
# ==========================================================

@app.post(
    "/monitoring/outcomes"
)
def create_prediction_outcome(
    request: PredictionOutcomeRequest,
):

    return (
        save_prediction_outcome(

            prediction_id=
                request.prediction_id,

            actual_class=
                request.actual_class,

            outcome_source=
                request.outcome_source,

            notes=
                request.notes,
        )
    )


# ==========================================================
# GET ALL OUTCOMES
# ==========================================================

@app.get(
    "/monitoring/outcomes"
)
def prediction_outcomes(
    model_name: str | None = None,
):

    return (
        get_prediction_outcomes(

            model_name=
                model_name
        )
    )


# ==========================================================
# GET ONE OUTCOME
# ==========================================================

@app.get(
    "/monitoring/outcomes/{prediction_id}"
)
def prediction_outcome(
    prediction_id: int,
):

    return (
        get_prediction_outcome(
            prediction_id
        )
    )


# ==========================================================
# OUTCOME COVERAGE
# ==========================================================

@app.get(
    "/monitoring/outcome-coverage"
)
def prediction_outcome_coverage():

    return (
        get_outcome_coverage()
    )


# ==========================================================
# ALL MODEL PERFORMANCE
# ==========================================================

@app.get(
    "/monitoring/performance"
)
def monitoring_performance():

    return (
        get_all_model_performance()
    )


# ==========================================================
# SINGLE MODEL PERFORMANCE
# ==========================================================

@app.get(
    "/monitoring/performance/{model_name}"
)
def monitoring_model_performance(
    model_name: str,
):

    try:

        return (
            get_model_performance(
                model_name
            )
        )


    except ValueError as error:

        raise HTTPException(

            status_code=
                400,

            detail=
                str(
                    error
                ),
        )