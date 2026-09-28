# src/monitoring/prediction_logger.py

import json
import os

from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.core.logger import get_logger


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(
    override=True
)


# ==========================================================
# LOGGER
# ==========================================================

logger = get_logger(
    __name__
)


# ==========================================================
# MONITORING DATABASE CONNECTION
# ==========================================================

MONITORING_DATABASE_URL = os.getenv(
    "MONITORING_DATABASE_URL"
)


if not MONITORING_DATABASE_URL:

    raise RuntimeError(
        "MONITORING_DATABASE_URL is not configured in .env"
    )


monitoring_engine = create_engine(
    MONITORING_DATABASE_URL,
    pool_pre_ping=True,
)


# ==========================================================
# TEST MONITORING CONNECTION
# ==========================================================

def test_monitoring_connection():
    """
    Check whether the monitoring database connection works.
    """

    try:

        with monitoring_engine.connect() as connection:

            result = connection.execute(
                text(
                    """
                    SELECT
                        current_user,
                        current_database()
                    """
                )
            ).fetchone()


        return {
            "status": "success",
            "user": result[0],
            "database": result[1],
        }


    except Exception as error:

        return {
            "status": "error",
            "error": str(error),
        }


# ==========================================================
# JSON SAFE CONVERTER
# ==========================================================

def make_json_safe(
    value,
):
    """
    Convert Python / NumPy-like values into values
    PostgreSQL JSONB can safely store.
    """

    if value is None:

        return None


    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value


    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):

        return value.isoformat()


    # ------------------------------------------------------
    # NUMPY SCALAR
    # ------------------------------------------------------

    if hasattr(
        value,
        "item",
    ):

        try:

            return make_json_safe(
                value.item()
            )

        except Exception:

            pass


    # ------------------------------------------------------
    # DICTIONARY
    # ------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        return {

            str(key):
                make_json_safe(
                    item
                )

            for key, item
            in value.items()
        }


    # ------------------------------------------------------
    # LIST / TUPLE / SET
    # ------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [

            make_json_safe(
                item
            )

            for item
            in value
        ]


    # ------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------

    return str(
        value
    )


# ==========================================================
# LOG PREDICTION
# ==========================================================

def log_prediction(
    model_name: str,
    model_version: str,
    endpoint: str,
    predicted_class: str | None = None,
    probability: float | None = None,
    threshold: float | None = None,
    feature_snapshot: dict | None = None,
    explainability: dict | None = None,
    latency_ms: float | None = None,
    status: str = "success",
    error_message: str | None = None,
):
    """
    Store one ML prediction event in PostgreSQL.

    Monitoring failure must not break the actual
    ML prediction API.

    Therefore this function returns None
    if database logging fails.
    """

    # ======================================================
    # CLEAN FEATURE VALUES
    # ======================================================

    safe_features = (
        make_json_safe(
            feature_snapshot
        )
        if feature_snapshot
        else {}
    )


    # ======================================================
    # CLEAN EXPLAINABILITY VALUES
    # ======================================================

    safe_explainability = (
        make_json_safe(
            explainability
        )
        if explainability
        else {}
    )


    # ======================================================
    # CONVERT NUMERIC VALUES
    # ======================================================

    if probability is not None:

        probability = float(
            probability
        )


    if threshold is not None:

        threshold = float(
            threshold
        )


    if latency_ms is not None:

        latency_ms = float(
            latency_ms
        )


    # ======================================================
    # SQL QUERY
    # ======================================================

    query = text(
        """
        INSERT INTO monitoring.prediction_logs
        (
            model_name,
            model_version,
            endpoint,
            predicted_class,
            probability,
            threshold,
            feature_snapshot,
            explainability,
            latency_ms,
            status,
            error_message
        )
        VALUES
        (
            :model_name,
            :model_version,
            :endpoint,
            :predicted_class,
            :probability,
            :threshold,
            CAST(:feature_snapshot AS JSONB),
            CAST(:explainability AS JSONB),
            :latency_ms,
            :status,
            :error_message
        )
        RETURNING prediction_id;
        """
    )


    # ======================================================
    # PARAMETERS
    # ======================================================

    parameters = {

        "model_name":
            model_name,

        "model_version":
            model_version,

        "endpoint":
            endpoint,

        "predicted_class":
            predicted_class,

        "probability":
            probability,

        "threshold":
            threshold,

        "feature_snapshot":
            json.dumps(
                safe_features
            ),

        "explainability":
            json.dumps(
                safe_explainability
            ),

        "latency_ms":
            latency_ms,

        "status":
            status,

        "error_message":
            error_message,
    }


    # ======================================================
    # INSERT INTO POSTGRESQL
    # ======================================================

    try:

        with monitoring_engine.begin() as connection:

            prediction_id = (
                connection.execute(
                    query,
                    parameters,
                )
                .scalar_one()
            )


        logger.info(
            (
                "Prediction logged successfully | "
                f"model={model_name} | "
                f"prediction_id={prediction_id}"
            )
        )


        return prediction_id


    # ======================================================
    # MONITORING ERROR
    # ======================================================

    except Exception as error:

        logger.exception(
            (
                "Prediction monitoring log failed | "
                f"model={model_name} | "
                f"error={error}"
            )
        )


        return None


# ==========================================================
# MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    connection_result = (
        test_monitoring_connection()
    )


    print(
        "Monitoring connection:"
    )

    print(
        connection_result
    )


    if (
        connection_result.get(
            "status"
        )
        ==
        "success"
    ):

        prediction_id = log_prediction(

            model_name=
                "test_model",

            model_version=
                "v1.0",

            endpoint=
                "/test",

            predicted_class=
                "Test",

            probability=
                0.75,

            threshold=
                0.50,

            feature_snapshot=
                {
                    "AGE": 35,
                },

            explainability=
                {
                    "method":
                        "SHAP",
                },

            latency_ms=
                120.5,

            status=
                "success",
        )


        print(
            "Prediction ID:",
            prediction_id,
        )