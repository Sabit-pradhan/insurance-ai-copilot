# src/monitoring/renewal_batch_service.py

import requests

from sqlalchemy import text

from src.data.db import engine

from src.monitoring.prediction_logger import (
    monitoring_engine,
)

from src.monitoring.ground_truth_service import (
    sync_renewal_ground_truth,
)


# ==========================================================
# API
# ==========================================================

API_BASE_URL = (
    "http://127.0.0.1:8001"
)


# ==========================================================
# GET POLICIES ALREADY SCORED
# ==========================================================

def get_already_scored_policy_ids():

    query = text(
        """
        SELECT DISTINCT

            feature_snapshot
            ->> 'POLICY_ID'
            AS policy_id

        FROM
            monitoring.prediction_logs

        WHERE
            model_name = 'renewal'

            AND
            status = 'success'

            AND
            endpoint =
            '/predict/renewal/by-policy'

            AND
            feature_snapshot
            ->> 'POLICY_ID'
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


    return set(
        rows
    )


# ==========================================================
# GET EVALUATION POLICY IDS
# ==========================================================

def get_evaluation_policy_ids(
    renewed_limit: int = 10,
    non_renewed_limit: int = 10,
):

    already_scored = (
        get_already_scored_policy_ids()
    )


    # Fetch extra candidates because some
    # may already have been scored.

    fetch_limit = (
        max(
            renewed_limit,
            non_renewed_limit,
        )
        *
        10
    )


    # ======================================================
    # RENEWED POLICIES
    # ======================================================

    renewed_query = text(
        """
        SELECT
            policy_id

        FROM public.renewals

        WHERE
            renewal_status = 'Renewed'

        ORDER BY
            policy_id

        LIMIT :limit
        """
    )


    # ======================================================
    # NON-RENEWED POLICIES
    # ======================================================

    non_renewed_query = text(
        """
        SELECT
            policy_id

        FROM public.renewals

        WHERE
            renewal_status
            IN (
                'Lapsed',
                'Cancelled'
            )

        ORDER BY
            policy_id

        LIMIT :limit
        """
    )


    with engine.connect() as connection:

        renewed_candidates = (
            connection.execute(
                renewed_query,
                {
                    "limit":
                        fetch_limit
                },
            )
            .scalars()
            .all()
        )


        non_renewed_candidates = (
            connection.execute(
                non_renewed_query,
                {
                    "limit":
                        fetch_limit
                },
            )
            .scalars()
            .all()
        )


    # ======================================================
    # REMOVE ALREADY SCORED IDS
    # ======================================================

    renewed_ids = [

        policy_id

        for policy_id
        in renewed_candidates

        if policy_id
        not in already_scored

    ][
        :renewed_limit
    ]


    non_renewed_ids = [

        policy_id

        for policy_id
        in non_renewed_candidates

        if policy_id
        not in already_scored

    ][
        :non_renewed_limit
    ]


    return {

        "renewed":
            renewed_ids,

        "non_renewed":
            non_renewed_ids,
    }


# ==========================================================
# SCORE ONE POLICY
# ==========================================================

def score_policy(
    policy_id: str,
):

    url = (
        f"{API_BASE_URL}"
        f"/predict/renewal/"
        f"by-policy/{policy_id}"
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

                "policy_id":
                    policy_id,

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


        return {

            "policy_id":
                policy_id,

            "status":
                "success",

            "prediction_id":
                (
                    result
                    .get(
                        "monitoring",
                        {},
                    )
                    .get(
                        "prediction_id"
                    )
                ),

            "renewal_probability":
                result.get(
                    "renewal_probability"
                ),

            "churn_probability":
                result.get(
                    "churn_probability"
                ),

            "prediction":
                result.get(
                    "prediction"
                ),

            "threshold":
                result.get(
                    "threshold"
                ),
        }


    except Exception as error:

        return {

            "policy_id":
                policy_id,

            "status":
                "failed",

            "error":
                str(
                    error
                ),
        }


# ==========================================================
# RUN RENEWAL EVALUATION BATCH
# ==========================================================

def run_renewal_evaluation_batch(
    renewed_limit: int = 10,
    non_renewed_limit: int = 10,
):

    selected = (
        get_evaluation_policy_ids(

            renewed_limit=
                renewed_limit,

            non_renewed_limit=
                non_renewed_limit,
        )
    )


    renewed_ids = (
        selected[
            "renewed"
        ]
    )


    non_renewed_ids = (
        selected[
            "non_renewed"
        ]
    )


    evaluation_cases = []


    for policy_id in renewed_ids:

        evaluation_cases.append(
            {
                "policy_id":
                    policy_id,

                "actual_group":
                    "Renewed",
            }
        )


    for policy_id in non_renewed_ids:

        evaluation_cases.append(
            {
                "policy_id":
                    policy_id,

                "actual_group":
                    "Not Renewed",
            }
        )


    results = []


    print(
        "\n=============================="
    )

    print(
        "RENEWAL BATCH SCORING"
    )

    print(
        "=============================="
    )


    print(
        "\nSelected:",
        len(
            evaluation_cases
        ),
    )


    print(
        "Renewed:",
        len(
            renewed_ids
        ),
    )


    print(
        "Not Renewed:",
        len(
            non_renewed_ids
        ),
    )


    # ======================================================
    # SCORE POLICIES
    # ======================================================

    for index, case in enumerate(
        evaluation_cases,
        start=1,
    ):

        policy_id = (
            case[
                "policy_id"
            ]
        )


        actual_group = (
            case[
                "actual_group"
            ]
        )


        print(
            f"\n[{index}/"
            f"{len(evaluation_cases)}] "
            f"{policy_id} "
            f"Actual={actual_group}"
        )


        result = (
            score_policy(
                policy_id
            )
        )


        result[
            "actual_group"
        ] = (
            actual_group
        )


        results.append(
            result
        )


        if (
            result[
                "status"
            ]
            ==
            "success"
        ):

            print(
                "Prediction:",
                result.get(
                    "prediction"
                ),
            )


            print(
                "Churn probability:",
                result.get(
                    "churn_probability"
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
        "GROUND TRUTH SYNC"
    )

    print(
        "=============================="
    )


    ground_truth_result = (
        sync_renewal_ground_truth(
            limit=500
        )
    )


    # ======================================================
    # SUMMARY
    # ======================================================

    successful = sum(

        1

        for item in results

        if item[
            "status"
        ]
        ==
        "success"
    )


    failed = (
        len(
            results
        )
        -
        successful
    )


    summary = {

        "selected":
            len(
                evaluation_cases
            ),

        "successful":
            successful,

        "failed":
            failed,

        "renewed_selected":
            len(
                renewed_ids
            ),

        "non_renewed_selected":
            len(
                non_renewed_ids
            ),

        "results":
            results,

        "ground_truth_sync":
            ground_truth_result,
    }


    return summary


# ==========================================================
# MANUAL RUN
# ==========================================================

if __name__ == "__main__":

    from pprint import pprint


    result = (
        run_renewal_evaluation_batch(

            renewed_limit=
                10,

            non_renewed_limit=
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