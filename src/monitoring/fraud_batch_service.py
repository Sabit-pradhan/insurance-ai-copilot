# src/monitoring/fraud_batch_service.py

import requests

from sqlalchemy import text

from src.data.db import engine

from src.monitoring.ground_truth_service import (
    sync_fraud_ground_truth,
)


# ==========================================================
# API CONFIGURATION
# ==========================================================

API_BASE_URL = (
    "http://127.0.0.1:8001"
)


# ==========================================================
# GET REAL CLAIM IDS FOR EVALUATION
# ==========================================================

def get_evaluation_claim_ids(
    normal_limit=10,
    confirmed_limit=10,
):

    """
    Get finalized real claims for evaluation.

    No
        -> Not Fraud

    Confirmed
        -> Fraud

    Suspected is intentionally excluded
    because it is not a finalized outcome.

    Important:
    fraud_flag is used here only to construct
    an evaluation sample.

    It is NEVER passed to the ML model.
    """

    query = text(
        """
        (
            SELECT

                claim_id,

                fraud_flag

            FROM public.claims

            WHERE
                fraud_flag = 'No'

            ORDER BY
                claim_id

            LIMIT :normal_limit
        )

        UNION ALL

        (
            SELECT

                claim_id,

                fraud_flag

            FROM public.claims

            WHERE
                fraud_flag = 'Confirmed'

            ORDER BY
                claim_id

            LIMIT :confirmed_limit
        );
        """
    )


    with engine.connect() as connection:

        rows = (
            connection.execute(
                query,
                {
                    "normal_limit":
                        normal_limit,

                    "confirmed_limit":
                        confirmed_limit,
                },
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
# SCORE ONE CLAIM THROUGH API
# ==========================================================

def score_claim(
    claim_id: str,
):

    """
    Score one real claim using the existing
    database-driven FastAPI endpoint.
    """

    url = (
        f"{API_BASE_URL}"
        f"/predict/fraud/by-claim/"
        f"{claim_id}"
    )


    response = (
        requests.post(
            url,
            timeout=60,
        )
    )


    if response.status_code != 200:

        return {

            "claim_id":
                claim_id,

            "status":
                "error",

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
            {}
        )
        or
        {}
    )


    return {

        "claim_id":
            claim_id,

        "status":
            "success",

        "prediction_id":
            monitoring.get(
                "prediction_id"
            ),

        "fraud_probability":
            result.get(
                "fraud_probability"
            ),

        "threshold":
            result.get(
                "threshold"
            ),

        "prediction":
            result.get(
                "prediction"
            ),

        "feature_source":
            monitoring.get(
                "feature_source"
            ),
    }


# ==========================================================
# RUN FRAUD EVALUATION BATCH
# ==========================================================

def run_fraud_evaluation_batch(
    normal_limit=10,
    confirmed_limit=10,
):

    """
    Score multiple real claims.

    Step 1:
        Load real finalized claim IDs.

    Step 2:
        Score each claim using real DB features.

    Step 3:
        Sync actual fraud outcomes.

    Step 4:
        Return summary.
    """

    claims = (
        get_evaluation_claim_ids(

            normal_limit=
                normal_limit,

            confirmed_limit=
                confirmed_limit,
        )
    )


    results = []


    print(
        "\n"
        "=========================================="
    )

    print(
        "FRAUD EVALUATION BATCH STARTED"
    )

    print(
        "=========================================="
    )


    print(
        f"\nClaims selected: {len(claims)}"
    )


    # ======================================================
    # SCORE CLAIMS
    # ======================================================

    for index, claim in enumerate(
        claims,
        start=1,
    ):

        claim_id = (
            claim[
                "claim_id"
            ]
        )


        actual_flag = (
            claim[
                "fraud_flag"
            ]
        )


        print(
            f"\n[{index}/{len(claims)}] "
            f"Scoring {claim_id}"
        )


        try:

            score_result = (
                score_claim(
                    claim_id
                )
            )


            score_result[
                "evaluation_flag"
            ] = (
                actual_flag
            )


            results.append(
                score_result
            )


            if (
                score_result[
                    "status"
                ]
                ==
                "success"
            ):

                print(
                    "Prediction ID:",
                    score_result[
                        "prediction_id"
                    ],
                )


                print(
                    "Probability:",
                    score_result[
                        "fraud_probability"
                    ],
                )


                print(
                    "Prediction:",
                    score_result[
                        "prediction"
                    ],
                )


                print(
                    "Actual flag:",
                    actual_flag,
                )


            else:

                print(
                    "ERROR:",
                    score_result.get(
                        "error"
                    ),
                )


        except Exception as error:

            results.append(
                {

                    "claim_id":
                        claim_id,

                    "status":
                        "error",

                    "evaluation_flag":
                        actual_flag,

                    "error":
                        str(
                            error
                        ),
                }
            )


            print(
                "ERROR:",
                error,
            )


    # ======================================================
    # SYNC GROUND TRUTH
    # ======================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "SYNCING GROUND TRUTH"
    )

    print(
        "=========================================="
    )


    sync_result = (
        sync_fraud_ground_truth(
            limit=500
        )
    )


    # ======================================================
    # BATCH COUNTS
    # ======================================================

    successful = sum(

        1

        for result in results

        if (
            result.get(
                "status"
            )
            ==
            "success"
        )
    )


    failed = (
        len(
            results
        )
        -
        successful
    )


    normal_claims = sum(

        1

        for claim in claims

        if (
            claim[
                "fraud_flag"
            ]
            ==
            "No"
        )
    )


    confirmed_claims = sum(

        1

        for claim in claims

        if (
            claim[
                "fraud_flag"
            ]
            ==
            "Confirmed"
        )
    )


    # ======================================================
    # FINAL RESULT
    # ======================================================

    summary = {

        "selected_claims":
            len(
                claims
            ),

        "normal_claims":
            normal_claims,

        "confirmed_fraud_claims":
            confirmed_claims,

        "successful_predictions":
            successful,

        "failed_predictions":
            failed,

        "ground_truth_sync":
            sync_result,

        "predictions":
            results,
    }


    print(
        "\n"
        "=========================================="
    )

    print(
        "BATCH COMPLETE"
    )

    print(
        "=========================================="
    )


    print(
        "\nSelected:",
        len(
            claims
        ),
    )


    print(
        "Successful:",
        successful,
    )


    print(
        "Failed:",
        failed,
    )


    print(
        "Normal claims:",
        normal_claims,
    )


    print(
        "Confirmed fraud:",
        confirmed_claims,
    )


    return summary


# ==========================================================
# MANUAL RUN
# ==========================================================

if __name__ == "__main__":

    run_fraud_evaluation_batch(

        normal_limit=10,

        confirmed_limit=10,
    )