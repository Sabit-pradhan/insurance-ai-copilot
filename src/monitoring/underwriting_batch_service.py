import requests

from sqlalchemy import text

from src.data.db import engine

from src.monitoring.prediction_logger import (
    monitoring_engine,
)

from src.monitoring.ground_truth_service import (
    sync_underwriting_ground_truth,
)


# ==========================================================
# API CONFIG
# ==========================================================

API_BASE_URL = "http://127.0.0.1:8001"


# ==========================================================
# GET ALREADY SUCCESSFULLY SCORED IDS
# ==========================================================

def get_already_scored_underwriting_ids():

    query = text(
        """
        SELECT DISTINCT

            feature_snapshot
            ->> 'UNDERWRITING_ID'
            AS underwriting_id

        FROM monitoring.prediction_logs

        WHERE
            model_name = 'underwriting'

            AND
            status = 'success'

            AND
            endpoint = '/predict/underwriting/by-id'

            AND
            feature_snapshot
            ->> 'UNDERWRITING_ID'
            IS NOT NULL
        """
    )


    with monitoring_engine.connect() as connection:

        rows = (
            connection.execute(
                query
            )
            .scalars()
            .all()
        )


    return set(rows)


# ==========================================================
# GET IDS FOR ONE DECISION CLASS
# ==========================================================

def get_ids_by_decision(
    decision: str,
    limit: int,
    already_scored: set,
):

    if limit <= 0:

        return []


    # Fetch more because some rows
    # may already be successfully scored.

    fetch_limit = max(
        limit * 20,
        100,
    )


    query = text(
        """
        SELECT
            underwriting_id

        FROM public.underwriting

        WHERE
            decision = :decision

        ORDER BY
            underwriting_id

        LIMIT :limit
        """
    )


    with engine.connect() as connection:

        candidates = (
            connection.execute(
                query,
                {
                    "decision":
                        decision,

                    "limit":
                        fetch_limit,
                },
            )
            .scalars()
            .all()
        )


    selected = []


    for underwriting_id in candidates:

        if underwriting_id in already_scored:

            continue


        selected.append(
            underwriting_id
        )


        if len(selected) >= limit:

            break


    return selected


# ==========================================================
# SELECT EVALUATION CASES
# ==========================================================

def get_evaluation_underwriting_ids(
    approved_limit: int = 10,
    loading_limit: int = 10,
    declined_limit: int = 10,
):

    already_scored = (
        get_already_scored_underwriting_ids()
    )


    approved_ids = (
        get_ids_by_decision(

            decision=
                "Approved",

            limit=
                approved_limit,

            already_scored=
                already_scored,
        )
    )


    loading_ids = (
        get_ids_by_decision(

            decision=
                "Approved with Loading",

            limit=
                loading_limit,

            already_scored=
                already_scored,
        )
    )


    declined_ids = (
        get_ids_by_decision(

            decision=
                "Declined",

            limit=
                declined_limit,

            already_scored=
                already_scored,
        )
    )


    return {

        "Approved":
            approved_ids,

        "Approved with Loading":
            loading_ids,

        "Declined":
            declined_ids,
    }


# ==========================================================
# SCORE ONE UNDERWRITING CASE
# ==========================================================

def score_underwriting_case(
    underwriting_id: str,
):

    url = (
        f"{API_BASE_URL}"
        f"/predict/underwriting/"
        f"by-id/{underwriting_id}"
    )


    try:

        response = (
            requests.post(
                url,
                timeout=60,
            )
        )


        if response.status_code != 200:

            return {

                "underwriting_id":
                    underwriting_id,

                "status":
                    "failed",

                "status_code":
                    response.status_code,

                "error":
                    response.text,
            }


        result = (
            response.json()
        )


        monitoring = (
            result.get(
                "monitoring",
                {},
            )
            or
            {}
        )


        predicted_decision = (

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


        return {

            "underwriting_id":
                underwriting_id,

            "status":
                "success",

            "prediction_id":
                monitoring.get(
                    "prediction_id"
                ),

            "predicted_decision":
                predicted_decision,

            "probabilities":
                result.get(
                    "probabilities",
                    {},
                ),
        }


    except Exception as error:

        return {

            "underwriting_id":
                underwriting_id,

            "status":
                "failed",

            "error":
                str(error),
        }


# ==========================================================
# RUN UNDERWRITING EVALUATION BATCH
# ==========================================================

def run_underwriting_evaluation_batch(
    approved_limit: int = 10,
    loading_limit: int = 10,
    declined_limit: int = 10,
):

    selected = (
        get_evaluation_underwriting_ids(

            approved_limit=
                approved_limit,

            loading_limit=
                loading_limit,

            declined_limit=
                declined_limit,
        )
    )


    cases = []


    for actual_decision, underwriting_ids in selected.items():

        for underwriting_id in underwriting_ids:

            cases.append(
                {
                    "underwriting_id":
                        underwriting_id,

                    "actual_decision":
                        actual_decision,
                }
            )


    # ======================================================
    # SHOW SELECTION
    # ======================================================

    print(
        "\n=============================="
    )

    print(
        "UNDERWRITING BATCH SCORING"
    )

    print(
        "=============================="
    )


    print(
        "Selected:",
        len(cases),
    )


    print(
        "Approved:",
        len(
            selected[
                "Approved"
            ]
        ),
    )


    print(
        "Approved with Loading:",
        len(
            selected[
                "Approved with Loading"
            ]
        ),
    )


    print(
        "Declined:",
        len(
            selected[
                "Declined"
            ]
        ),
    )


    results = []


    # ======================================================
    # SCORE CASES
    # ======================================================

    for index, case in enumerate(
        cases,
        start=1,
    ):

        underwriting_id = (
            case[
                "underwriting_id"
            ]
        )


        actual_decision = (
            case[
                "actual_decision"
            ]
        )


        print(
            f"\n[{index}/{len(cases)}] "
            f"{underwriting_id}"
        )


        print(
            "Actual:",
            actual_decision,
        )


        result = (
            score_underwriting_case(
                underwriting_id
            )
        )


        result[
            "actual_decision"
        ] = (
            actual_decision
        )


        results.append(
            result
        )


        if result["status"] == "success":

            print(
                "Predicted:",
                result.get(
                    "predicted_decision"
                ),
            )


            print(
                "Prediction ID:",
                result.get(
                    "prediction_id"
                ),
            )


        else:

            print(
                "FAILED:",
                result.get(
                    "error"
                ),
            )


    # ======================================================
    # GROUND TRUTH SYNC
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


    ground_truth_result = (
        sync_underwriting_ground_truth(
            limit=500
        )
    )


    # ======================================================
    # COUNTS
    # ======================================================

    successful = sum(

        1

        for result in results

        if result[
            "status"
        ] == "success"
    )


    failed = (
        len(results)
        -
        successful
    )


    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    summary = {

        "selected":
            len(cases),

        "successful":
            successful,

        "failed":
            failed,

        "approved_selected":
            len(
                selected[
                    "Approved"
                ]
            ),

        "loading_selected":
            len(
                selected[
                    "Approved with Loading"
                ]
            ),

        "declined_selected":
            len(
                selected[
                    "Declined"
                ]
            ),

        "ground_truth_sync":
            ground_truth_result,

        "results":
            results,
    }


    return summary


# ==========================================================
# MANUAL RUN
# ==========================================================

if __name__ == "__main__":

    from pprint import pprint


    result = (
        run_underwriting_evaluation_batch(

            approved_limit=
                10,

            loading_limit=
                10,

            declined_limit=
                10,
        )
    )


    print(
        "\n=============================="
    )

    print(
        "FINAL BATCH SUMMARY"
    )

    print(
        "=============================="
    )


    pprint(
        result
    )