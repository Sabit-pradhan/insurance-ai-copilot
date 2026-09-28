# src/data/fraud_feature_service.py

from sqlalchemy import text

from src.data.db import engine


# ==========================================================
# GET FRAUD CASE FROM DATABASE
# ==========================================================

def get_fraud_case_by_claim_id(
    claim_id: str,
):

    """
    Load a real claim and all fraud-model features
    from claims + customers + policies.

    Important:
    fraud_flag is returned separately as historical
    ground truth. It is NOT a model feature.
    """

    query = text(
        """
        SELECT

            -- =================================================
            -- IDENTIFIER
            -- =================================================

            c.claim_id,


            -- =================================================
            -- CLAIM FEATURES
            -- =================================================

            c.claim_amount,

            EXTRACT(
                DAY FROM (
                    c.reported_date
                    -
                    c.incident_date
                )
            )::INT
                AS reporting_delay_days,

            EXTRACT(
                MONTH FROM
                    c.incident_date
            )::INT
                AS incident_month,

            c.claim_type,

            c.source,

            c.claim_severity,


            -- =================================================
            -- CUSTOMER FEATURES
            -- =================================================

            cu.age,

            cu.annual_income,

            cu.credit_score,

            cu.gender,

            cu.marital_status,

            cu.occupation,

            cu.state,

            cu.customer_risk_segment,


            -- =================================================
            -- POLICY FEATURES
            -- =================================================

            p.sum_insured,

            p.annual_premium,

            p.risk_score,

            p.policy_type,

            p.payment_mode,

            p.risk_band,


            -- =================================================
            -- GROUND TRUTH
            -- NOT USED AS MODEL INPUT
            -- =================================================

            c.fraud_flag


        FROM public.claims c


        LEFT JOIN public.customers cu

            ON c.customer_id
            =
            cu.customer_id


        LEFT JOIN public.policies p

            ON c.policy_id
            =
            p.policy_id


        WHERE
            c.claim_id = :claim_id;
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

        raise ValueError(
            f"Claim not found: {claim_id}"
        )


    row = dict(
        row
    )


    # ==========================================================
    # BUILD MODEL FEATURES
    # ==========================================================

    model_features = {

        # ------------------------------------------------------
        # NUMERIC
        # ------------------------------------------------------

        "CLAIM_AMOUNT":
            float(
                row["claim_amount"]
            ),

        "REPORTING_DELAY_DAYS":
            float(
                row["reporting_delay_days"]
            ),

        "INCIDENT_MONTH":
            int(
                row["incident_month"]
            ),

        "AGE":
            int(
                row["age"]
            ),

        "ANNUAL_INCOME":
            float(
                row["annual_income"]
            ),

        "CREDIT_SCORE":
            int(
                row["credit_score"]
            ),

        "SUM_INSURED":
            float(
                row["sum_insured"]
            ),

        "ANNUAL_PREMIUM":
            float(
                row["annual_premium"]
            ),

        "RISK_SCORE":
            float(
                row["risk_score"]
            ),


        # ------------------------------------------------------
        # CATEGORICAL
        # ------------------------------------------------------

        "CLAIM_TYPE":
            row["claim_type"],

        "SOURCE":
            row["source"],

        "CLAIM_SEVERITY":
            row["claim_severity"],

        "GENDER":
            row["gender"],

        "MARITAL_STATUS":
            row["marital_status"],

        "OCCUPATION":
            row["occupation"],

        "STATE":
            row["state"],

        "CUSTOMER_RISK_SEGMENT":
            row["customer_risk_segment"],

        "POLICY_TYPE":
            row["policy_type"],

        "PAYMENT_MODE":
            row["payment_mode"],

        "RISK_BAND":
            row["risk_band"],
    }


    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    return {

        "claim_id":
            row["claim_id"],

        "model_features":
            model_features,

        # Historical outcome only.
        # Never send this into predict_fraud().
        "actual_fraud_flag":
            row["fraud_flag"],
    }


# ==========================================================
# GET ONLY MODEL FEATURES
# ==========================================================

def get_fraud_features_by_claim_id(
    claim_id: str,
):

    fraud_case = (
        get_fraud_case_by_claim_id(
            claim_id
        )
    )


    return (
        fraud_case[
            "model_features"
        ]
    )


# ==========================================================
# MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    claim_id = (
        "CLM000000001"
    )


    result = (
        get_fraud_case_by_claim_id(
            claim_id
        )
    )


    print(
        "\nCLAIM ID\n"
    )

    print(
        result[
            "claim_id"
        ]
    )


    print(
        "\nMODEL FEATURES\n"
    )

    for key, value in (
        result[
            "model_features"
        ]
        .items()
    ):

        print(
            f"{key}: {value}"
        )


    print(
        "\nACTUAL FRAUD FLAG\n"
    )

    print(
        result[
            "actual_fraud_flag"
        ]
    )