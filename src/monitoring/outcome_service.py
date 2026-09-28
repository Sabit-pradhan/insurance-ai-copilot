# src/monitoring/outcome_service.py

from sqlalchemy import text

from src.monitoring.prediction_logger import (
    monitoring_engine,
    make_json_safe,
)


# ==========================================================
# CHECK PREDICTION
# ==========================================================

def get_prediction(
    prediction_id: int,
):
    """
    Get one prediction from monitoring logs.
    """

    query = text(
        """
        SELECT
            prediction_id,
            model_name,
            model_version,
            predicted_class,
            probability,
            threshold,
            status,
            created_at

        FROM monitoring.prediction_logs

        WHERE prediction_id = :prediction_id;
        """
    )


    with monitoring_engine.connect() as connection:

        row = (
            connection.execute(
                query,
                {
                    "prediction_id":
                        prediction_id
                },
            )
            .mappings()
            .first()
        )


    if not row:

        return None


    return make_json_safe(
        dict(
            row
        )
    )


# ==========================================================
# SAVE ACTUAL OUTCOME
# ==========================================================

def save_prediction_outcome(
    prediction_id: int,
    actual_class: str,
    outcome_source: str | None = None,
    notes: str | None = None,
):
    """
    Store the real-world outcome for one prediction.

    If an outcome already exists for the prediction,
    update it instead of creating a duplicate.
    """

    # ------------------------------------------------------
    # VALIDATE PREDICTION
    # ------------------------------------------------------

    prediction = get_prediction(
        prediction_id
    )


    if prediction is None:

        raise ValueError(
            f"Prediction ID {prediction_id} does not exist."
        )


    # ------------------------------------------------------
    # UPSERT OUTCOME
    # ------------------------------------------------------

    query = text(
        """
        INSERT INTO monitoring.prediction_outcomes
        (
            prediction_id,
            actual_class,
            outcome_source,
            notes
        )
        VALUES
        (
            :prediction_id,
            :actual_class,
            :outcome_source,
            :notes
        )

        ON CONFLICT (prediction_id)

        DO UPDATE SET

            actual_class =
                EXCLUDED.actual_class,

            outcome_source =
                EXCLUDED.outcome_source,

            notes =
                EXCLUDED.notes,

            outcome_date =
                NOW()

        RETURNING
            outcome_id,
            prediction_id,
            actual_class,
            outcome_source,
            outcome_date,
            notes,
            created_at;
        """
    )


    parameters = {

        "prediction_id":
            prediction_id,

        "actual_class":
            actual_class,

        "outcome_source":
            outcome_source,

        "notes":
            notes,
    }


    with monitoring_engine.begin() as connection:

        row = (
            connection.execute(
                query,
                parameters,
            )
            .mappings()
            .one()
        )


    return make_json_safe(
        dict(
            row
        )
    )


# ==========================================================
# GET OUTCOME
# ==========================================================

def get_prediction_outcome(
    prediction_id: int,
):
    """
    Return prediction + actual outcome.
    """

    query = text(
        """
        SELECT

            p.prediction_id,

            p.model_name,

            p.model_version,

            p.predicted_class,

            p.probability,

            p.threshold,

            o.actual_class,

            o.outcome_source,

            o.outcome_date,

            o.notes

        FROM monitoring.prediction_logs p

        LEFT JOIN monitoring.prediction_outcomes o

            ON p.prediction_id =
               o.prediction_id

        WHERE p.prediction_id =
              :prediction_id;
        """
    )


    with monitoring_engine.connect() as connection:

        row = (
            connection.execute(
                query,
                {
                    "prediction_id":
                        prediction_id
                },
            )
            .mappings()
            .first()
        )


    if not row:

        return None


    return make_json_safe(
        dict(
            row
        )
    )


# ==========================================================
# GET ALL OUTCOMES
# ==========================================================

def get_prediction_outcomes(
    model_name: str | None = None,
):
    """
    Return predictions that already have
    real-world outcomes.
    """

    query = text(
        """
        SELECT

            p.prediction_id,

            p.model_name,

            p.model_version,

            p.predicted_class,

            p.probability,

            p.threshold,

            o.actual_class,

            o.outcome_source,

            o.outcome_date,

            o.notes

        FROM monitoring.prediction_logs p

        INNER JOIN monitoring.prediction_outcomes o

            ON p.prediction_id =
               o.prediction_id

        WHERE
            (
                :model_name IS NULL
                OR
                p.model_name = :model_name
            )

        ORDER BY
            o.outcome_date DESC;
        """
    )


    with monitoring_engine.connect() as connection:

        rows = (
            connection.execute(
                query,
                {
                    "model_name":
                        model_name
                },
            )
            .mappings()
            .all()
        )


    return make_json_safe(
        [
            dict(row)
            for row in rows
        ]
    )


# ==========================================================
# COUNT OUTCOME COVERAGE
# ==========================================================

def get_outcome_coverage():
    """
    How many predictions already have
    actual outcomes attached?
    """

    query = text(
        """
        SELECT

            COUNT(*) AS total_predictions,

            COUNT(o.outcome_id)
                AS predictions_with_outcomes,

            COUNT(*)
            -
            COUNT(o.outcome_id)
                AS pending_outcomes,

            ROUND(
                (
                    COUNT(o.outcome_id)::numeric
                    /
                    NULLIF(
                        COUNT(*),
                        0
                    )
                )
                * 100,
                2
            ) AS outcome_coverage_pct

        FROM monitoring.prediction_logs p

        LEFT JOIN monitoring.prediction_outcomes o

            ON p.prediction_id =
               o.prediction_id

        WHERE p.model_name IN
        (
            'renewal',
            'fraud',
            'underwriting'
        );
        """
    )


    with monitoring_engine.connect() as connection:

        row = (
            connection.execute(
                query
            )
            .mappings()
            .first()
        )


    if not row:

        return {}


    return make_json_safe(
        dict(
            row
        )
    )