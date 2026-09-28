# src/data/underwriting_feature_service.py

from sqlalchemy import text

from src.data.db import engine


# ==========================================================
# UNDERWRITING MODEL FEATURES
# ==========================================================

UNDERWRITING_MODEL_FEATURES = [

    "AGE",
    "HEALTH_SCORE",
    "BMI",
    "CREDIT_SCORE",

    "LIFESTYLE",
    "MEDICAL_HISTORY_FLAG",
    "SMOKER_FLAG",
    "OCCUPATION_RISK",
]


# ==========================================================
# GET UNDERWRITING CASE BY ID
# ==========================================================

def get_underwriting_case_by_id(
    underwriting_id: str,
):

    # ------------------------------------------------------
    # CLEAN ID
    # ------------------------------------------------------

    underwriting_id = (
        underwriting_id
        .strip()
        .upper()
    )


    # ------------------------------------------------------
    # QUERY
    # ------------------------------------------------------

    query = text(
        """
        SELECT

            underwriting_id,
            policy_id,
            customer_id,

            age,
            health_score,
            lifestyle,
            medical_history_flag,
            smoker_flag,
            bmi,
            occupation_risk,
            credit_score,

            underwriting_score,
            decision,
            premium_loading_pct

        FROM public.underwriting

        WHERE
            underwriting_id = :underwriting_id
        """
    )


    # ------------------------------------------------------
    # RUN QUERY
    # ------------------------------------------------------

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


    # ======================================================
    # NOT FOUND
    # ======================================================

    if row is None:

        raise ValueError(
            (
                "Underwriting record not found "
                f"for ID: {underwriting_id}"
            )
        )


    row = dict(
        row
    )


    # ======================================================
    # BUILD EXACT MODEL INPUT
    # ======================================================

    model_features = {

        # --------------------------------------------------
        # NUMERIC
        # --------------------------------------------------

        "AGE":
            int(
                row[
                    "age"
                ]
            ),

        "HEALTH_SCORE":
            float(
                row[
                    "health_score"
                ]
            ),

        "BMI":
            float(
                row[
                    "bmi"
                ]
            ),

        "CREDIT_SCORE":
            int(
                row[
                    "credit_score"
                ]
            ),


        # --------------------------------------------------
        # CATEGORICAL
        # --------------------------------------------------

        "LIFESTYLE":
            str(
                row[
                    "lifestyle"
                ]
            ),

        "MEDICAL_HISTORY_FLAG":
            str(
                row[
                    "medical_history_flag"
                ]
            ),

        "SMOKER_FLAG":
            str(
                row[
                    "smoker_flag"
                ]
            ),

        "OCCUPATION_RISK":
            str(
                row[
                    "occupation_risk"
                ]
            ),
    }


    # ======================================================
    # RETURN
    # ======================================================

    return {

        "underwriting_id":
            row[
                "underwriting_id"
            ],

        "policy_id":
            row[
                "policy_id"
            ],

        "customer_id":
            row[
                "customer_id"
            ],

        "model_features":
            model_features,


        # --------------------------------------------------
        # GROUND TRUTH
        # --------------------------------------------------
        #
        # IMPORTANT:
        #
        # DECISION must NEVER be sent
        # into the model.
        #
        # It will be used later only
        # for performance evaluation.
        # --------------------------------------------------

        "actual_decision":
            row[
                "decision"
            ],


        # --------------------------------------------------
        # REFERENCE DATA
        # --------------------------------------------------
        #
        # These fields are useful for analytics,
        # but are not model inputs.
        # --------------------------------------------------

        "underwriting_score":
            float(
                row[
                    "underwriting_score"
                ]
            ),

        "premium_loading_pct": None if row["premium_loading_pct"] is None else float(row["premium_loading_pct"]),
    }


# ==========================================================
# SIMPLE FEATURE FUNCTION
# ==========================================================

def get_underwriting_features_by_id(
    underwriting_id: str,
):

    result = (
        get_underwriting_case_by_id(
            underwriting_id
        )
    )


    return result[
        "model_features"
    ]


# ==========================================================
# MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    from pprint import pprint


    # ------------------------------------------------------
    # GET ONE REAL UNDERWRITING ID
    # ------------------------------------------------------

    query = text(
        """
        SELECT underwriting_id

        FROM public.underwriting

        ORDER BY underwriting_id

        LIMIT 1
        """
    )


    with engine.connect() as connection:

        underwriting_id = (
            connection.execute(
                query
            )
            .scalar()
        )


    print(
        "\nTesting Underwriting ID:",
        underwriting_id,
    )


    print(
        "\n=============================="
    )

    print(
        "UNDERWRITING REAL-ID FEATURES"
    )

    print(
        "=============================="
    )


    result = (
        get_underwriting_case_by_id(
            underwriting_id
        )
    )


    pprint(
        result
    )

