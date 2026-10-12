from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# RENEWAL FEATURE SERVICE
# ============================================================

class RenewalFeatureService:

    def __init__(self):

        self.customers = pd.read_csv(
            RAW_DIR / "customers.csv"
        )

        self.policies = pd.read_csv(
            RAW_DIR / "policies.csv"
        )

        self.renewals = pd.read_csv(
            RAW_DIR / "renewals.csv"
        )

        self.premiums = pd.read_csv(
            RAW_DIR / "premiums.csv"
        )

        # ----------------------------------------------------
        # Date conversion
        # ----------------------------------------------------

        self.policies["POLICY_START_DATE"] = pd.to_datetime(
            self.policies["POLICY_START_DATE"],
            errors="coerce",
        )

        self.policies["POLICY_END_DATE"] = pd.to_datetime(
            self.policies["POLICY_END_DATE"],
            errors="coerce",
        )

        self.renewals["RENEWAL_DUE_DATE"] = pd.to_datetime(
            self.renewals["RENEWAL_DUE_DATE"],
            errors="coerce",
        )

        self.premiums["DUE_DATE"] = pd.to_datetime(
            self.premiums["DUE_DATE"],
            errors="coerce",
        )

        self.premiums["PAID_DATE"] = pd.to_datetime(
            self.premiums["PAID_DATE"],
            errors="coerce",
        )

    # ========================================================
    # FIND RENEWAL RECORD
    # ========================================================

    def get_renewal_record(self, policy_id):

        renewal = self.renewals[
            self.renewals["POLICY_ID"]
            == policy_id
        ]

        if renewal.empty:

            raise ValueError(
                f"No renewal record found for policy "
                f"{policy_id}"
            )

        return renewal.iloc[0]

    # ========================================================
    # FIND POLICY RECORD
    # ========================================================

    def get_policy_record(self, policy_id):

        policy = self.policies[
            self.policies["POLICY_ID"]
            == policy_id
        ]

        if policy.empty:

            raise ValueError(
                f"No policy record found for policy "
                f"{policy_id}"
            )

        return policy.iloc[0]

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    def build_features(self, policy_id):

        renewal = self.get_renewal_record(
            policy_id
        )

        policy = self.get_policy_record(
            policy_id
        )

        # ----------------------------------------------------
        # Dates
        # ----------------------------------------------------

        renewal_due_date = pd.to_datetime(
            renewal["RENEWAL_DUE_DATE"],
            errors="coerce",
        )

        policy_start_date = pd.to_datetime(
            policy["POLICY_START_DATE"],
            errors="coerce",
        )

        # ====================================================
        # PREMIUM FEATURES
        # ====================================================

        old_premium = float(
            renewal["OLD_PREMIUM"]
        )

        new_premium = float(
            renewal["NEW_PREMIUM"]
        )

        premium_increase_pct = float(
            renewal["PREMIUM_INCREASE_PCT"]
        )

        premium_increase_amount = (
            new_premium - old_premium
        )

        # ====================================================
        # BASIC POLICY FEATURES
        # ====================================================

        loyalty_years = float(
            renewal["LOYALTY_YEARS"]
        )

        # ----------------------------------------------------
        # FIX:
        # CLAIM_HISTORY_FLAG is stored as Yes/No in CSV.
        #
        # Yes -> 1
        # No  -> 0
        # ----------------------------------------------------

        claim_value = str(
            renewal["CLAIM_HISTORY_FLAG"]
        ).strip().lower()

        if claim_value in {
            "yes",
            "y",
            "true",
            "1",
        }:

            claim_history_flag = 1.0

        else:

            claim_history_flag = 0.0

        sum_insured = float(
            policy["SUM_INSURED"]
        )

        annual_premium = float(
            policy["ANNUAL_PREMIUM"]
        )

        risk_score = float(
            policy["RISK_SCORE"]
        )

        discount_pct = float(
            policy["DISCOUNT_PCT"]
        )

        commission_pct = float(
            policy["COMMISSION_PCT"]
        )

        # ====================================================
        # POLICY TENURE
        # ====================================================

        if (
            pd.notna(renewal_due_date)
            and pd.notna(policy_start_date)
        ):

            policy_tenure_days = (
                renewal_due_date
                - policy_start_date
            ).days

        else:

            policy_tenure_days = np.nan

        if pd.notna(policy_tenure_days):

            policy_tenure_years = (
                policy_tenure_days / 365.25
            )

        else:

            policy_tenure_years = np.nan

        # ====================================================
        # PREMIUM RATIOS
        # ====================================================

        if sum_insured > 0:

            premium_to_sum_insured = (
                annual_premium
                / sum_insured
            )

        else:

            premium_to_sum_insured = np.nan

        discounted_premium_ratio = (
            1 - discount_pct / 100
        )

        # ====================================================
        # PAYMENT HISTORY
        # ====================================================

        # Only transactions whose DUE_DATE is on or before
        # the renewal decision date are used.

        policy_premiums = self.premiums[
            (
                self.premiums["POLICY_ID"]
                == policy_id
            )
            &
            (
                self.premiums["DUE_DATE"]
                <= renewal_due_date
            )
        ].copy()

        total_payment_txns_prior = (
            len(policy_premiums)
        )

        # ====================================================
        # NO PAYMENT HISTORY
        # ====================================================

        if total_payment_txns_prior == 0:

            paid_txns_prior = 0

            failed_txns_prior = 0

            overdue_txns_prior = 0

            avg_payment_delay_prior = None

            max_payment_delay_prior = None

            total_premium_prior = 0.0

            avg_premium_prior = None

        # ====================================================
        # PAYMENT HISTORY EXISTS
        # ====================================================

        else:

            payment_status = (
                policy_premiums[
                    "PAYMENT_STATUS"
                ]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            # ------------------------------------------------
            # Payment status groups
            # ------------------------------------------------

            paid_statuses = {
                "PAID",
                "SUCCESS",
                "SUCCESSFUL",
                "COMPLETED",
            }

            failed_statuses = {
                "FAILED",
                "FAIL",
                "FAILURE",
                "REJECTED",
                "DECLINED",
            }

            overdue_statuses = {
                "OVERDUE",
                "LATE",
            }

            # ------------------------------------------------
            # Status masks
            # ------------------------------------------------

            paid_mask = (
                payment_status
                .isin(paid_statuses)
            )

            failed_mask = (
                payment_status
                .isin(failed_statuses)
            )

            overdue_mask = (
                payment_status
                .isin(overdue_statuses)
            )

            # ------------------------------------------------
            # Counts
            # ------------------------------------------------

            paid_txns_prior = int(
                paid_mask.sum()
            )

            failed_txns_prior = int(
                failed_mask.sum()
            )

            overdue_txns_prior = int(
                overdue_mask.sum()
            )

            # =================================================
            # PAYMENT DELAY
            # =================================================

            policy_premiums[
                "PAYMENT_DELAY_DAYS"
            ] = pd.to_numeric(
                policy_premiums[
                    "PAYMENT_DELAY_DAYS"
                ],
                errors="coerce",
            )

            # Only use payment delays when the payment
            # itself was known by the renewal date.

            known_delay_mask = (
                policy_premiums[
                    "PAID_DATE"
                ].notna()
                &
                (
                    policy_premiums[
                        "PAID_DATE"
                    ]
                    <= renewal_due_date
                )
            )

            known_delays = (
                policy_premiums.loc[
                    known_delay_mask,
                    "PAYMENT_DELAY_DAYS",
                ]
            )

            if known_delays.empty:

                avg_payment_delay_prior = None

                max_payment_delay_prior = None

            else:

                avg_payment_delay_prior = float(
                    known_delays.mean()
                )

                max_payment_delay_prior = float(
                    known_delays.max()
                )

            # =================================================
            # PREMIUM HISTORY
            # =================================================

            policy_premiums[
                "PREMIUM_AMOUNT"
            ] = pd.to_numeric(
                policy_premiums[
                    "PREMIUM_AMOUNT"
                ],
                errors="coerce",
            )

            total_premium_prior = float(
                policy_premiums[
                    "PREMIUM_AMOUNT"
                ].sum()
            )

            avg_premium_prior = float(
                policy_premiums[
                    "PREMIUM_AMOUNT"
                ].mean()
            )

        # ====================================================
        # PAYMENT RATES
        # ====================================================

        if total_payment_txns_prior > 0:

            paid_rate_prior = (
                paid_txns_prior
                / total_payment_txns_prior
            )

            failed_rate_prior = (
                failed_txns_prior
                / total_payment_txns_prior
            )

            overdue_rate_prior = (
                overdue_txns_prior
                / total_payment_txns_prior
            )

        else:

            paid_rate_prior = None

            failed_rate_prior = None

            overdue_rate_prior = None

        # ====================================================
        # PREMIUM PAYMENT RATIO
        # ====================================================

        if total_premium_prior > 0:

            premium_payment_ratio = (
                old_premium
                / total_premium_prior
            )

        else:

            premium_payment_ratio = None

        # ====================================================
        # FINAL 32 FEATURES
        # ====================================================

        features = {

            # ------------------------------------------------
            # Numeric features
            # ------------------------------------------------

            "OLD_PREMIUM":
                old_premium,

            "NEW_PREMIUM":
                new_premium,

            "PREMIUM_INCREASE_PCT":
                premium_increase_pct,

            "PREMIUM_INCREASE_AMOUNT":
                premium_increase_amount,

            "LOYALTY_YEARS":
                loyalty_years,

            "CLAIM_HISTORY_FLAG":
                claim_history_flag,

            "SUM_INSURED":
                sum_insured,

            "ANNUAL_PREMIUM":
                annual_premium,

            "RISK_SCORE":
                risk_score,

            "DISCOUNT_PCT":
                discount_pct,

            "COMMISSION_PCT":
                commission_pct,

            "POLICY_TENURE_DAYS":
                policy_tenure_days,

            "POLICY_TENURE_YEARS":
                policy_tenure_years,

            "PREMIUM_TO_SUM_INSURED":
                premium_to_sum_insured,

            "DISCOUNTED_PREMIUM_RATIO":
                discounted_premium_ratio,

            "TOTAL_PAYMENT_TXNS_PRIOR":
                total_payment_txns_prior,

            "PAID_TXNS_PRIOR":
                paid_txns_prior,

            "FAILED_TXNS_PRIOR":
                failed_txns_prior,

            "OVERDUE_TXNS_PRIOR":
                overdue_txns_prior,

            "AVG_PAYMENT_DELAY_PRIOR":
                avg_payment_delay_prior,

            "MAX_PAYMENT_DELAY_PRIOR":
                max_payment_delay_prior,

            "TOTAL_PREMIUM_PRIOR":
                total_premium_prior,

            "AVG_PREMIUM_PRIOR":
                avg_premium_prior,

            "PAID_RATE_PRIOR":
                paid_rate_prior,

            "FAILED_RATE_PRIOR":
                failed_rate_prior,

            "OVERDUE_RATE_PRIOR":
                overdue_rate_prior,

            "PREMIUM_PAYMENT_RATIO":
                premium_payment_ratio,

            # ------------------------------------------------
            # Categorical features
            # ------------------------------------------------

            "POLICY_TYPE":
                policy["POLICY_TYPE"],

            "PAYMENT_MODE":
                policy["PAYMENT_MODE"],

            "SALES_CHANNEL":
                policy["SALES_CHANNEL"],

            "CONTACT_CHANNEL":
                renewal["CONTACT_CHANNEL"],

            "RISK_BAND":
                policy["RISK_BAND"],
        }

        return features


# ============================================================
# SINGLETON
# ============================================================

_feature_service = None


def get_feature_service():

    global _feature_service

    if _feature_service is None:

        _feature_service = (
            RenewalFeatureService()
        )

    return _feature_service


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def build_renewal_features(policy_id):

    service = get_feature_service()

    return service.build_features(
        policy_id
    )