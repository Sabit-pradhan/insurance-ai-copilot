# src/data/renewal_feature_service.py

from sqlalchemy import text

from src.data.db import engine


# ==========================================================
# MODEL FEATURES
# ==========================================================

RENEWAL_MODEL_FEATURES = [

    # Numeric
    "AGE",
    "ANNUAL_INCOME",
    "CREDIT_SCORE",
    "SUM_INSURED",
    "ANNUAL_PREMIUM",
    "RISK_SCORE",
    "PREMIUM_INCREASE_PCT",
    "CUSTOMER_TENURE_YEARS",
    "TOTAL_PAYMENTS",
    "AVG_PAYMENT_DELAY",
    "MAX_PAYMENT_DELAY",
    "TOTAL_CLAIMS",
    "TOTAL_CLAIM_AMOUNT",
    "AVG_CLAIM_AMOUNT",
    "MAX_CLAIM_AMOUNT",

    # Categorical
    "GENDER",
    "MARITAL_STATUS",
    "OCCUPATION",
    "STATE",
    "CUSTOMER_RISK_SEGMENT",
    "POLICY_TYPE",
    "SALES_CHANNEL",
    "PAYMENT_MODE",
    "RISK_BAND",
]


# ==========================================================
# GET RENEWAL CASE
# ==========================================================

def get_renewal_case_by_policy_id(
    policy_id: str,
):

    policy_id = (
        policy_id
        .strip()
        .upper()
    )


    query = text(
        """
        WITH payment_summary AS
        (
            SELECT

                policy_id,

                COUNT(
                    premium_txn_id
                ) AS total_payments,

                AVG(
                    payment_delay_days
                ) AS avg_payment_delay,

                MAX(
                    payment_delay_days
                ) AS max_payment_delay

            FROM public.premiums

            GROUP BY
                policy_id
        ),


        claim_summary AS
        (
            SELECT

                policy_id,

                COUNT(
                    claim_id
                ) AS total_claims,

                SUM(
                    claim_amount
                ) AS total_claim_amount,

                AVG(
                    claim_amount
                ) AS avg_claim_amount,

                MAX(
                    claim_amount
                ) AS max_claim_amount

            FROM public.claims

            GROUP BY
                policy_id
        )


        SELECT

            -- ==========================================
            -- IDENTIFIERS
            -- ==========================================

            r.renewal_id,

            r.policy_id,

            r.customer_id,


            -- ==========================================
            -- CUSTOMER FEATURES
            -- ==========================================

            c.age,

            c.annual_income,

            c.credit_score,

            c.gender,

            c.marital_status,

            c.occupation,

            c.state,

            c.customer_risk_segment,


            -- ==========================================
            -- POLICY FEATURES
            -- ==========================================

            p.sum_insured,

            p.annual_premium,

            p.risk_score,

            p.policy_type,

            p.sales_channel,

            p.payment_mode,

            p.risk_band,


            -- ==========================================
            -- RENEWAL FEATURES
            -- ==========================================

            r.premium_increase_pct,

            r.loyalty_years,

            r.renewal_status,


            -- ==========================================
            -- PAYMENT FEATURES
            -- ==========================================

            COALESCE(
                pay.total_payments,
                0
            ) AS total_payments,

            COALESCE(
                pay.avg_payment_delay,
                0
            ) AS avg_payment_delay,

            COALESCE(
                pay.max_payment_delay,
                0
            ) AS max_payment_delay,


            -- ==========================================
            -- CLAIM FEATURES
            -- ==========================================

            COALESCE(
                cl.total_claims,
                0
            ) AS total_claims,

            COALESCE(
                cl.total_claim_amount,
                0
            ) AS total_claim_amount,

            COALESCE(
                cl.avg_claim_amount,
                0
            ) AS avg_claim_amount,

            COALESCE(
                cl.max_claim_amount,
                0
            ) AS max_claim_amount


        FROM public.renewals r


        INNER JOIN public.policies p

            ON r.policy_id
            =
            p.policy_id


        INNER JOIN public.customers c

            ON p.customer_id
            =
            c.customer_id


        LEFT JOIN payment_summary pay

            ON r.policy_id
            =
            pay.policy_id


        LEFT JOIN claim_summary cl

            ON r.policy_id
            =
            cl.policy_id


        WHERE
            r.policy_id = :policy_id
        """
    )


    # ======================================================
    # RUN QUERY
    # ======================================================

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


    # ======================================================
    # POLICY NOT FOUND
    # ======================================================

    if row is None:

        raise ValueError(
            (
                "Renewal record not found "
                f"for policy: {policy_id}"
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
        # CUSTOMER
        # --------------------------------------------------

        "AGE":
            int(
                row[
                    "age"
                ]
            ),

        "ANNUAL_INCOME":
            float(
                row[
                    "annual_income"
                ]
            ),

        "CREDIT_SCORE":
            int(
                row[
                    "credit_score"
                ]
            ),


        # --------------------------------------------------
        # POLICY
        # --------------------------------------------------

        "SUM_INSURED":
            float(
                row[
                    "sum_insured"
                ]
            ),

        "ANNUAL_PREMIUM":
            float(
                row[
                    "annual_premium"
                ]
            ),

        "RISK_SCORE":
            float(
                row[
                    "risk_score"
                ]
            ),


        # --------------------------------------------------
        # RENEWAL
        # --------------------------------------------------

        "PREMIUM_INCREASE_PCT":
            float(
                row[
                    "premium_increase_pct"
                ]
            ),

        # Current model calls this:
        # CUSTOMER_TENURE_YEARS
        #
        # Renewal source stores this business concept as:
        # LOYALTY_YEARS

        "CUSTOMER_TENURE_YEARS":
            float(
                row[
                    "loyalty_years"
                ]
            ),


        # --------------------------------------------------
        # PAYMENT HISTORY
        # --------------------------------------------------

        "TOTAL_PAYMENTS":
            float(
                row[
                    "total_payments"
                ]
            ),

        "AVG_PAYMENT_DELAY":
            float(
                row[
                    "avg_payment_delay"
                ]
            ),

        "MAX_PAYMENT_DELAY":
            float(
                row[
                    "max_payment_delay"
                ]
            ),


        # --------------------------------------------------
        # CLAIM HISTORY
        # --------------------------------------------------

        "TOTAL_CLAIMS":
            float(
                row[
                    "total_claims"
                ]
            ),

        "TOTAL_CLAIM_AMOUNT":
            float(
                row[
                    "total_claim_amount"
                ]
            ),

        "AVG_CLAIM_AMOUNT":
            float(
                row[
                    "avg_claim_amount"
                ]
            ),

        "MAX_CLAIM_AMOUNT":
            float(
                row[
                    "max_claim_amount"
                ]
            ),


        # --------------------------------------------------
        # CATEGORICAL
        # --------------------------------------------------

        "GENDER":
            str(
                row[
                    "gender"
                ]
            ),

        "MARITAL_STATUS":
            str(
                row[
                    "marital_status"
                ]
            ),

        "OCCUPATION":
            str(
                row[
                    "occupation"
                ]
            ),

        "STATE":
            str(
                row[
                    "state"
                ]
            ),

        "CUSTOMER_RISK_SEGMENT":
            str(
                row[
                    "customer_risk_segment"
                ]
            ),

        "POLICY_TYPE":
            str(
                row[
                    "policy_type"
                ]
            ),

        "SALES_CHANNEL":
            str(
                row[
                    "sales_channel"
                ]
            ),

        "PAYMENT_MODE":
            str(
                row[
                    "payment_mode"
                ]
            ),

        "RISK_BAND":
            str(
                row[
                    "risk_band"
                ]
            ),
    }


    # ======================================================
    # RETURN
    # ======================================================

    return {

        "policy_id":
            row[
                "policy_id"
            ],

        "renewal_id":
            row[
                "renewal_id"
            ],

        "customer_id":
            row[
                "customer_id"
            ],

        "model_features":
            model_features,

        # Keep actual outcome separate.
        # NEVER send this into the model.

        "actual_renewal_status":
            row[
                "renewal_status"
            ],
    }


# ==========================================================
# SIMPLE FEATURE FUNCTION
# ==========================================================

def get_renewal_features_by_policy_id(
    policy_id: str,
):

    result = (
        get_renewal_case_by_policy_id(
            policy_id
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


    query = text(
        """
        SELECT policy_id
        FROM public.renewals
        ORDER BY policy_id
        LIMIT 1
        """
    )


    with engine.connect() as connection:

        policy_id = (
            connection.execute(
                query
            )
            .scalar()
        )


    print(
        "\nTesting Policy:",
        policy_id,
    )


    result = (
        get_renewal_case_by_policy_id(
            policy_id
        )
    )


    pprint(
        result
    )