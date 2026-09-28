from sqlalchemy import text

from src.data.db import engine

from src.monitoring.prediction_logger import (
    monitoring_engine,
)

from src.monitoring.outcome_service import (
    save_prediction_outcome,
)


# ==========================================================
# GET PREDICTIONS WAITING FOR GROUND TRUTH
# ==========================================================

def get_pending_ground_truth(
    model_name: str | None = None,
    endpoint: str | None = None,
    limit: int = 100,
):

    query = """
        SELECT

            p.prediction_id,
            p.model_name,
            p.endpoint,
            p.predicted_class,
            p.probability,
            p.threshold,
            p.feature_snapshot,
            p.created_at

        FROM monitoring.prediction_logs p

        LEFT JOIN monitoring.prediction_outcomes o

            ON p.prediction_id
            =
            o.prediction_id

        WHERE
            p.status = 'success'

            AND
            o.prediction_id IS NULL
    """


    params = {
        "limit":
            limit
    }


    if model_name is not None:

        query += """

            AND
            p.model_name = :model_name
        """

        params[
            "model_name"
        ] = model_name


    if endpoint is not None:

        query += """

            AND
            p.endpoint = :endpoint
        """

        params[
            "endpoint"
        ] = endpoint


    query += """

        ORDER BY
            p.prediction_id

        LIMIT :limit
    """


    with monitoring_engine.connect() as connection:

        rows = (
            connection.execute(
                text(
                    query
                ),
                params,
            )
            .mappings()
            .all()
        )


    return [
        dict(
            row
        )
        for row in rows
    ]


# ==========================================================
# SAVE GROUND TRUTH
# ==========================================================

def save_ground_truth(
    prediction_id: int,
    actual_class: str,
    outcome_source: str,
    notes: str | None = None,
):

    return (
        save_prediction_outcome(

            prediction_id=
                prediction_id,

            actual_class=
                actual_class,

            outcome_source=
                outcome_source,

            notes=
                notes,
        )
    )


# ==========================================================
# FRAUD GROUND TRUTH LOOKUP
# ==========================================================

def get_claim_fraud_flag(
    claim_id: str,
):

    query = text(
        """
        SELECT

            fraud_flag

        FROM public.claims

        WHERE
            claim_id = :claim_id
        """
    )


    with engine.connect() as connection:

        row = (
            connection.execute(
                query,
                {
                    "claim_id":
                        claim_id
                },
            )
            .mappings()
            .first()
        )


    if row is None:

        return None


    return row[
        "fraud_flag"
    ]


# ==========================================================
# FRAUD LABEL MAPPING
# ==========================================================

def map_fraud_ground_truth(
    fraud_flag,
):

    if fraud_flag is None:

        return None


    fraud_flag = (
        str(
            fraud_flag
        )
        .strip()
        .lower()
    )


    if fraud_flag == "confirmed":

        return (
            "Fraud Confirmed"
        )


    if fraud_flag == "no":

        return (
            "Fraud Not Confirmed"
        )


    # Suspected is not final ground truth.

    if fraud_flag == "suspected":

        return None


    return None


# ==========================================================
# SYNC FRAUD GROUND TRUTH
# ==========================================================

def sync_fraud_ground_truth(
    limit: int = 100,
):

    pending_predictions = (
        get_pending_ground_truth(

            model_name=
                "fraud",

            endpoint=
                "/predict/fraud/by-claim",

            limit=
                limit,
        )
    )


    summary = {

        "model_name":
            "fraud",

        "checked":
            0,

        "labeled":
            0,

        "skipped":
            0,

        "details":
            [],
    }


    for prediction in pending_predictions:

        summary[
            "checked"
        ] += 1


        prediction_id = (
            prediction[
                "prediction_id"
            ]
        )


        feature_snapshot = (
            prediction.get(
                "feature_snapshot"
            )
            or
            {}
        )


        claim_id = (
            feature_snapshot.get(
                "CLAIM_ID"
            )
        )


        # --------------------------------------------------
        # CLAIM ID MISSING
        # --------------------------------------------------

        if not claim_id:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "status":
                        "skipped",

                    "reason":
                        "CLAIM_ID missing",
                }
            )


            continue


        # --------------------------------------------------
        # GET ACTUAL FRAUD FLAG
        # --------------------------------------------------

        fraud_flag = (
            get_claim_fraud_flag(
                claim_id
            )
        )


        if fraud_flag is None:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "claim_id":
                        claim_id,

                    "status":
                        "skipped",

                    "reason":
                        "Claim not found",
                }
            )


            continue


        # --------------------------------------------------
        # MAP LABEL
        # --------------------------------------------------

        actual_class = (
            map_fraud_ground_truth(
                fraud_flag
            )
        )


        # Suspected = unfinished investigation

        if actual_class is None:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "claim_id":
                        claim_id,

                    "fraud_flag":
                        fraud_flag,

                    "status":
                        "skipped",

                    "reason":
                        (
                            "Fraud outcome "
                            "not finalized"
                        ),
                }
            )


            continue


        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------

        outcome = (
            save_ground_truth(

                prediction_id=
                    prediction_id,

                actual_class=
                    actual_class,

                outcome_source=
                    "public.claims.fraud_flag",

                notes=(
                    "Automatically synchronized "
                    "from claim ground truth."
                ),
            )
        )


        summary[
            "labeled"
        ] += 1


        summary[
            "details"
        ].append(
            {
                "prediction_id":
                    prediction_id,

                "claim_id":
                    claim_id,

                "fraud_flag":
                    fraud_flag,

                "actual_class":
                    actual_class,

                "status":
                    "labeled",

                "outcome":
                    outcome,
            }
        )


    return summary


# ==========================================================
# RENEWAL GROUND TRUTH LOOKUP
# ==========================================================

def get_policy_renewal_status(
    policy_id: str,
):

    query = text(
        """
        SELECT

            renewal_id,
            renewal_status

        FROM public.renewals

        WHERE
            policy_id = :policy_id
        """
    )


    with engine.connect() as connection:

        row = (
            connection.execute(
                query,
                {
                    "policy_id":
                        policy_id
                },
            )
            .mappings()
            .first()
        )


    if row is None:

        return None


    return dict(
        row
    )


# ==========================================================
# RENEWAL LABEL MAPPING
# ==========================================================

def map_renewal_ground_truth(
    renewal_status,
):

    if renewal_status is None:

        return None


    renewal_status = (
        str(
            renewal_status
        )
        .strip()
        .lower()
    )


    # Customer renewed policy

    if renewal_status == "renewed":

        return (
            "Renewed"
        )


    # Final non-renewal outcomes

    if renewal_status in {
        "lapsed",
        "cancelled",
    }:

        return (
            "Not Renewed"
        )


    # Pending is not final.

    if renewal_status == "pending":

        return None


    return None


# ==========================================================
# SYNC RENEWAL GROUND TRUTH
# ==========================================================

def sync_renewal_ground_truth(
    limit: int = 100,
):

    # Only evaluate the real database-driven endpoint.

    pending_predictions = (
        get_pending_ground_truth(

            model_name=
                "renewal",

            endpoint=
                "/predict/renewal/by-policy",

            limit=
                limit,
        )
    )


    summary = {

        "model_name":
            "renewal",

        "checked":
            0,

        "labeled":
            0,

        "skipped":
            0,

        "details":
            [],
    }


    for prediction in pending_predictions:

        summary[
            "checked"
        ] += 1


        prediction_id = (
            prediction[
                "prediction_id"
            ]
        )


        feature_snapshot = (
            prediction.get(
                "feature_snapshot"
            )
            or
            {}
        )


        policy_id = (
            feature_snapshot.get(
                "POLICY_ID"
            )
        )


        # --------------------------------------------------
        # POLICY ID MISSING
        # --------------------------------------------------

        if not policy_id:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "status":
                        "skipped",

                    "reason":
                        "POLICY_ID missing",
                }
            )


            continue


        # --------------------------------------------------
        # GET ACTUAL RENEWAL
        # --------------------------------------------------

        renewal_record = (
            get_policy_renewal_status(
                policy_id
            )
        )


        if renewal_record is None:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "policy_id":
                        policy_id,

                    "status":
                        "skipped",

                    "reason":
                        "Renewal record not found",
                }
            )


            continue


        renewal_status = (
            renewal_record[
                "renewal_status"
            ]
        )


        # --------------------------------------------------
        # MAP ACTUAL LABEL
        # --------------------------------------------------

        actual_class = (
            map_renewal_ground_truth(
                renewal_status
            )
        )


        # Pending is not a final outcome.

        if actual_class is None:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "policy_id":
                        policy_id,

                    "renewal_status":
                        renewal_status,

                    "status":
                        "skipped",

                    "reason":
                        (
                            "Renewal outcome "
                            "not finalized"
                        ),
                }
            )


            continue


        # --------------------------------------------------
        # SAVE OUTCOME
        # --------------------------------------------------

        outcome = (
            save_ground_truth(

                prediction_id=
                    prediction_id,

                actual_class=
                    actual_class,

                outcome_source=
                    "public.renewals.renewal_status",

                notes=(
                    "Automatically synchronized "
                    "from renewal ground truth."
                ),
            )
        )


        summary[
            "labeled"
        ] += 1


        summary[
            "details"
        ].append(
            {
                "prediction_id":
                    prediction_id,

                "policy_id":
                    policy_id,

                "renewal_id":
                    renewal_record[
                        "renewal_id"
                    ],

                "renewal_status":
                    renewal_status,

                "actual_class":
                    actual_class,

                "status":
                    "labeled",

                "outcome":
                    outcome,
            }
        )


    return summary


# ==========================================================
# UNDERWRITING GROUND TRUTH LOOKUP
# ==========================================================

def get_underwriting_decision(
    underwriting_id: str,
):

    query = text(
        """
        SELECT

            underwriting_id,
            policy_id,
            customer_id,
            decision

        FROM public.underwriting

        WHERE
            underwriting_id = :underwriting_id
        """
    )


    with engine.connect() as connection:

        row = (
            connection.execute(
                query,
                {
                    "underwriting_id":
                        underwriting_id
                },
            )
            .mappings()
            .first()
        )


    if row is None:

        return None


    return dict(
        row
    )


# ==========================================================
# UNDERWRITING LABEL MAPPING
# ==========================================================

def map_underwriting_ground_truth(
    decision,
):

    if decision is None:

        return None


    decision = (
        str(
            decision
        )
        .strip()
        .lower()
    )


    # ------------------------------------------------------
    # APPROVED
    # ------------------------------------------------------

    if decision == "approved":

        return (
            "Approved"
        )


    # ------------------------------------------------------
    # APPROVED WITH LOADING
    # ------------------------------------------------------

    if decision in {
        "approved with loading",
        "loading",
    }:

        return (
            "Approved with Loading"
        )


    # ------------------------------------------------------
    # DECLINED
    # ------------------------------------------------------

    if decision in {
        "declined",
        "rejected",
    }:

        return (
            "Declined"
        )


    return None


# ==========================================================
# SYNC UNDERWRITING GROUND TRUTH
# ==========================================================

def sync_underwriting_ground_truth(
    limit: int = 100,
):

    # Only evaluate the real database-driven endpoint.

    pending_predictions = (
        get_pending_ground_truth(

            model_name=
                "underwriting",

            endpoint=
                "/predict/underwriting/by-id",

            limit=
                limit,
        )
    )


    summary = {

        "model_name":
            "underwriting",

        "checked":
            0,

        "labeled":
            0,

        "skipped":
            0,

        "details":
            [],
    }


    for prediction in pending_predictions:

        summary[
            "checked"
        ] += 1


        prediction_id = (
            prediction[
                "prediction_id"
            ]
        )


        feature_snapshot = (
            prediction.get(
                "feature_snapshot"
            )
            or
            {}
        )


        underwriting_id = (
            feature_snapshot.get(
                "UNDERWRITING_ID"
            )
        )


        # --------------------------------------------------
        # UNDERWRITING ID MISSING
        # --------------------------------------------------

        if not underwriting_id:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "status":
                        "skipped",

                    "reason":
                        (
                            "UNDERWRITING_ID "
                            "missing"
                        ),
                }
            )


            continue


        # --------------------------------------------------
        # GET ACTUAL DECISION
        # --------------------------------------------------

        underwriting_record = (
            get_underwriting_decision(
                underwriting_id
            )
        )


        if underwriting_record is None:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "underwriting_id":
                        underwriting_id,

                    "status":
                        "skipped",

                    "reason":
                        (
                            "Underwriting record "
                            "not found"
                        ),
                }
            )


            continue


        actual_decision = (
            underwriting_record[
                "decision"
            ]
        )


        # --------------------------------------------------
        # MAP ACTUAL LABEL
        # --------------------------------------------------

        actual_class = (
            map_underwriting_ground_truth(
                actual_decision
            )
        )


        if actual_class is None:

            summary[
                "skipped"
            ] += 1


            summary[
                "details"
            ].append(
                {
                    "prediction_id":
                        prediction_id,

                    "underwriting_id":
                        underwriting_id,

                    "decision":
                        actual_decision,

                    "status":
                        "skipped",

                    "reason":
                        (
                            "Unknown underwriting "
                            "decision"
                        ),
                }
            )


            continue


        # --------------------------------------------------
        # SAVE OUTCOME
        # --------------------------------------------------

        outcome = (
            save_ground_truth(

                prediction_id=
                    prediction_id,

                actual_class=
                    actual_class,

                outcome_source=
                    "public.underwriting.decision",

                notes=(
                    "Automatically synchronized "
                    "from underwriting ground truth."
                ),
            )
        )


        summary[
            "labeled"
        ] += 1


        summary[
            "details"
        ].append(
            {
                "prediction_id":
                    prediction_id,

                "underwriting_id":
                    underwriting_id,

                "policy_id":
                    underwriting_record[
                        "policy_id"
                    ],

                "customer_id":
                    underwriting_record[
                        "customer_id"
                    ],

                "decision":
                    actual_decision,

                "actual_class":
                    actual_class,

                "status":
                    "labeled",

                "outcome":
                    outcome,
            }
        )


    return summary


# ==========================================================
# GROUND TRUTH STATUS
# ==========================================================

def get_ground_truth_status():

    query = text(
        """
        SELECT

            p.model_name,

            COUNT(*) AS total_predictions,

            COUNT(
                o.prediction_id
            ) AS labeled_predictions,

            COUNT(*) -
            COUNT(
                o.prediction_id
            ) AS pending_predictions

        FROM monitoring.prediction_logs p

        LEFT JOIN monitoring.prediction_outcomes o

            ON p.prediction_id
            =
            o.prediction_id

        WHERE
            p.status = 'success'

        GROUP BY
            p.model_name

        ORDER BY
            p.model_name
        """
    )


    with monitoring_engine.connect() as connection:

        rows = (
            connection.execute(
                query
            )
            .mappings()
            .all()
        )


    return [
        dict(
            row
        )
        for row in rows
    ]


# ==========================================================
# MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    from pprint import pprint


    # ======================================================
    # FRAUD
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        "FRAUD GROUND TRUTH SYNC"
    )

    print(
        "=============================="
    )


    fraud_result = (
        sync_fraud_ground_truth(
            limit=500
        )
    )


    pprint(
        fraud_result
    )


    # ======================================================
    # RENEWAL
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        "RENEWAL GROUND TRUTH SYNC"
    )

    print(
        "=============================="
    )


    renewal_result = (
        sync_renewal_ground_truth(
            limit=500
        )
    )


    pprint(
        renewal_result
    )


    # ======================================================
    # UNDERWRITING
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        "UNDERWRITING GROUND TRUTH SYNC"
    )

    print(
        "=============================="
    )


    underwriting_result = (
        sync_underwriting_ground_truth(
            limit=500
        )
    )


    pprint(
        underwriting_result
    )


    # ======================================================
    # STATUS
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        "GROUND TRUTH STATUS"
    )

    print(
        "=============================="
    )


    pprint(
        get_ground_truth_status()
    )