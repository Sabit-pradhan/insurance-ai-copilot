from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("RENEWAL CHURN MODEL V3")
print("=" * 70)

print("\n[1/9] Loading data...")

customers = pd.read_csv(
    RAW_DIR / "customers.csv"
)

policies = pd.read_csv(
    RAW_DIR / "policies.csv"
)

renewals = pd.read_csv(
    RAW_DIR / "renewals.csv"
)

premiums = pd.read_csv(
    RAW_DIR / "premiums.csv"
)

print(
    f"Customers : {customers.shape}"
)

print(
    f"Policies  : {policies.shape}"
)

print(
    f"Renewals  : {renewals.shape}"
)

print(
    f"Premiums  : {premiums.shape}"
)


# ============================================================
# 3. DATE CONVERSION
# ============================================================

print("\n[2/9] Converting dates...")

policies["POLICY_START_DATE"] = pd.to_datetime(
    policies["POLICY_START_DATE"],
    errors="coerce",
)

policies["POLICY_END_DATE"] = pd.to_datetime(
    policies["POLICY_END_DATE"],
    errors="coerce",
)

renewals["RENEWAL_DUE_DATE"] = pd.to_datetime(
    renewals["RENEWAL_DUE_DATE"],
    errors="coerce",
)

premiums["DUE_DATE"] = pd.to_datetime(
    premiums["DUE_DATE"],
    errors="coerce",
)

premiums["PAID_DATE"] = pd.to_datetime(
    premiums["PAID_DATE"],
    errors="coerce",
)


# ============================================================
# 4. CREATE TARGET
# ============================================================

print("\n[3/9] Creating target...")

# Pending is excluded because it is not a final outcome.
#
# Renewed   = 0
# Lapsed    = 1
# Cancelled = 1

valid_statuses = [
    "Renewed",
    "Lapsed",
    "Cancelled",
]

model_data = renewals[
    renewals["RENEWAL_STATUS"]
    .isin(valid_statuses)
].copy()

model_data["CHURN_FLAG"] = (
    model_data["RENEWAL_STATUS"]
    .isin(
        [
            "Lapsed",
            "Cancelled",
        ]
    )
    .astype(int)
)

print("\nRenewal status distribution:")

print(
    model_data[
        "RENEWAL_STATUS"
    ].value_counts()
)

print("\nTarget distribution:")

print(
    model_data[
        "CHURN_FLAG"
    ].value_counts()
)

print(
    f"\nChurn rate: "
    f"{model_data['CHURN_FLAG'].mean():.4f}"
)


# ============================================================
# 5. MERGE POLICY DATA
# ============================================================

print(
    "\n[4/9] Building policy-level features..."
)

policy_columns = [
    "POLICY_ID",
    "CUSTOMER_ID",
    "POLICY_TYPE",
    "SALES_CHANNEL",
    "POLICY_START_DATE",
    "POLICY_END_DATE",
    "SUM_INSURED",
    "ANNUAL_PREMIUM",
    "PAYMENT_MODE",
    "RISK_SCORE",
    "RISK_BAND",
    "DISCOUNT_PCT",
    "COMMISSION_PCT",
]

policy_data = policies[
    policy_columns
].copy()

model_data = model_data.merge(
    policy_data,
    on=[
        "POLICY_ID",
        "CUSTOMER_ID",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# 6. POLICY DERIVED FEATURES
# ============================================================

model_data[
    "PREMIUM_INCREASE_AMOUNT"
] = (
    model_data["NEW_PREMIUM"]
    - model_data["OLD_PREMIUM"]
)

model_data[
    "POLICY_TENURE_DAYS"
] = (
    model_data["RENEWAL_DUE_DATE"]
    - model_data["POLICY_START_DATE"]
).dt.days

model_data[
    "POLICY_TENURE_YEARS"
] = (
    model_data["POLICY_TENURE_DAYS"]
    / 365.25
)

model_data[
    "PREMIUM_TO_SUM_INSURED"
] = np.where(
    model_data["SUM_INSURED"] > 0,
    model_data["ANNUAL_PREMIUM"]
    / model_data["SUM_INSURED"],
    np.nan,
)

model_data[
    "DISCOUNTED_PREMIUM_RATIO"
] = (
    1
    - (
        model_data["DISCOUNT_PCT"]
        / 100
    )
)


# ============================================================
# 7. FIX CLAIM HISTORY
# ============================================================

print(
    "[5/9] Converting claim history..."
)

# Raw data contains Yes / No.
#
# Yes -> 1
# No  -> 0

claim_mapping = {
    "yes": 1.0,
    "y": 1.0,
    "true": 1.0,
    "1": 1.0,

    "no": 0.0,
    "n": 0.0,
    "false": 0.0,
    "0": 0.0,
}

model_data[
    "CLAIM_HISTORY_FLAG"
] = (
    model_data[
        "CLAIM_HISTORY_FLAG"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(claim_mapping)
)

print(
    "\nClaim history values after conversion:"
)

print(
    model_data[
        "CLAIM_HISTORY_FLAG"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# 8. TIME-SAFE PAYMENT HISTORY
# ============================================================

print(
    "\n[6/9] Building time-safe payment history..."
)

payment_base = premiums[
    [
        "POLICY_ID",
        "CUSTOMER_ID",
        "DUE_DATE",
        "PAID_DATE",
        "PREMIUM_AMOUNT",
        "PAYMENT_STATUS",
        "PAYMENT_DELAY_DAYS",
    ]
].copy()

payment_base = payment_base.merge(
    model_data[
        [
            "POLICY_ID",
            "CUSTOMER_ID",
            "RENEWAL_DUE_DATE",
        ]
    ],
    on=[
        "POLICY_ID",
        "CUSTOMER_ID",
    ],
    how="inner",
)

# Only payment transactions known by
# renewal due date are used.

payment_base = payment_base[
    payment_base["DUE_DATE"]
    <= payment_base["RENEWAL_DUE_DATE"]
].copy()


# ============================================================
# PAYMENT STATUS
# ============================================================

payment_base[
    "PAYMENT_STATUS"
] = (
    payment_base[
        "PAYMENT_STATUS"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)

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

payment_base[
    "IS_PAID"
] = (
    payment_base[
        "PAYMENT_STATUS"
    ]
    .isin(paid_statuses)
    .astype(int)
)

payment_base[
    "IS_FAILED"
] = (
    payment_base[
        "PAYMENT_STATUS"
    ]
    .isin(failed_statuses)
    .astype(int)
)

payment_base[
    "IS_OVERDUE"
] = (
    payment_base[
        "PAYMENT_STATUS"
    ]
    .isin(overdue_statuses)
    .astype(int)
)


# ============================================================
# PAYMENT DELAY
# ============================================================

payment_base[
    "PAYMENT_DELAY_DAYS"
] = pd.to_numeric(
    payment_base[
        "PAYMENT_DELAY_DAYS"
    ],
    errors="coerce",
)

known_delay_mask = (
    payment_base[
        "PAID_DATE"
    ].notna()
    &
    (
        payment_base[
            "PAID_DATE"
        ]
        <= payment_base[
            "RENEWAL_DUE_DATE"
        ]
    )
)

payment_base[
    "KNOWN_DELAY_DAYS"
] = np.where(
    known_delay_mask,
    payment_base[
        "PAYMENT_DELAY_DAYS"
    ],
    np.nan,
)


# ============================================================
# PAYMENT AGGREGATION
# ============================================================

payment_features = (
    payment_base
    .groupby(
        [
            "POLICY_ID",
            "CUSTOMER_ID",
            "RENEWAL_DUE_DATE",
        ],
        as_index=False,
    )
    .agg(
        TOTAL_PAYMENT_TXNS_PRIOR=(
            "POLICY_ID",
            "size",
        ),

        PAID_TXNS_PRIOR=(
            "IS_PAID",
            "sum",
        ),

        FAILED_TXNS_PRIOR=(
            "IS_FAILED",
            "sum",
        ),

        OVERDUE_TXNS_PRIOR=(
            "IS_OVERDUE",
            "sum",
        ),

        AVG_PAYMENT_DELAY_PRIOR=(
            "KNOWN_DELAY_DAYS",
            "mean",
        ),

        MAX_PAYMENT_DELAY_PRIOR=(
            "KNOWN_DELAY_DAYS",
            "max",
        ),

        TOTAL_PREMIUM_PRIOR=(
            "PREMIUM_AMOUNT",
            "sum",
        ),

        AVG_PREMIUM_PRIOR=(
            "PREMIUM_AMOUNT",
            "mean",
        ),
    )
)


# ============================================================
# MERGE PAYMENT FEATURES
# ============================================================

model_data = model_data.merge(
    payment_features,
    on=[
        "POLICY_ID",
        "CUSTOMER_ID",
        "RENEWAL_DUE_DATE",
    ],
    how="left",
)


# ============================================================
# MISSING PAYMENT COUNTS
# ============================================================

count_columns = [
    "TOTAL_PAYMENT_TXNS_PRIOR",
    "PAID_TXNS_PRIOR",
    "FAILED_TXNS_PRIOR",
    "OVERDUE_TXNS_PRIOR",
]

for col in count_columns:

    model_data[col] = (
        model_data[col]
        .fillna(0)
        .astype(float)
    )


# ============================================================
# PAYMENT RATES
# ============================================================

total_txns = (
    model_data[
        "TOTAL_PAYMENT_TXNS_PRIOR"
    ]
)

model_data[
    "PAID_RATE_PRIOR"
] = np.where(
    total_txns > 0,
    model_data[
        "PAID_TXNS_PRIOR"
    ]
    / total_txns,
    np.nan,
)

model_data[
    "FAILED_RATE_PRIOR"
] = np.where(
    total_txns > 0,
    model_data[
        "FAILED_TXNS_PRIOR"
    ]
    / total_txns,
    np.nan,
)

model_data[
    "OVERDUE_RATE_PRIOR"
] = np.where(
    total_txns > 0,
    model_data[
        "OVERDUE_TXNS_PRIOR"
    ]
    / total_txns,
    np.nan,
)


# ============================================================
# PREMIUM PAYMENT RATIO
# ============================================================

model_data[
    "PREMIUM_PAYMENT_RATIO"
] = np.where(
    model_data[
        "TOTAL_PREMIUM_PRIOR"
    ] > 0,
    model_data[
        "OLD_PREMIUM"
    ]
    / model_data[
        "TOTAL_PREMIUM_PRIOR"
    ],
    np.nan,
)


# ============================================================
# 9. FEATURE DEFINITIONS
# ============================================================

print(
    "\n[7/9] Preparing V3 features..."
)

numeric_features = [

    "OLD_PREMIUM",

    "NEW_PREMIUM",

    "PREMIUM_INCREASE_PCT",

    "PREMIUM_INCREASE_AMOUNT",

    "LOYALTY_YEARS",

    "CLAIM_HISTORY_FLAG",

    "SUM_INSURED",

    "ANNUAL_PREMIUM",

    "RISK_SCORE",

    "DISCOUNT_PCT",

    "COMMISSION_PCT",

    "POLICY_TENURE_DAYS",

    "POLICY_TENURE_YEARS",

    "PREMIUM_TO_SUM_INSURED",

    "DISCOUNTED_PREMIUM_RATIO",

    "TOTAL_PAYMENT_TXNS_PRIOR",

    "PAID_TXNS_PRIOR",

    "FAILED_TXNS_PRIOR",

    "OVERDUE_TXNS_PRIOR",

    "AVG_PAYMENT_DELAY_PRIOR",

    "MAX_PAYMENT_DELAY_PRIOR",

    "TOTAL_PREMIUM_PRIOR",

    "AVG_PREMIUM_PRIOR",

    "PAID_RATE_PRIOR",

    "FAILED_RATE_PRIOR",

    "OVERDUE_RATE_PRIOR",

    "PREMIUM_PAYMENT_RATIO",
]


categorical_features = [

    "POLICY_TYPE",

    "PAYMENT_MODE",

    "SALES_CHANNEL",

    "CONTACT_CHANNEL",

    "RISK_BAND",
]


feature_columns = (
    numeric_features
    + categorical_features
)


# ============================================================
# 10. CREATE X AND Y
# ============================================================

X = model_data[
    feature_columns
].copy()

y = model_data[
    "CHURN_FLAG"
].copy()


# ============================================================
# 11. CLEAN NUMERIC FEATURES
# ============================================================

for col in numeric_features:

    if col == "CLAIM_HISTORY_FLAG":

        # IMPORTANT:
        # This column has already been converted
        # to 0/1 above.

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    else:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )


# ============================================================
# 12. CLEAN CATEGORICAL FEATURES
# ============================================================

for col in categorical_features:

    X[col] = X[col].astype(
        "object"
    )


# ============================================================
# 13. VERIFY CLAIM FEATURE
# ============================================================

print(
    "\nCLAIM_HISTORY_FLAG verification:"
)

print(
    X[
        "CLAIM_HISTORY_FLAG"
    ].value_counts(
        dropna=False
    )
)

claim_missing = (
    X[
        "CLAIM_HISTORY_FLAG"
    ].isna().sum()
)

print(
    "Missing claim values:",
    claim_missing,
)

if (
    X[
        "CLAIM_HISTORY_FLAG"
    ].notna().sum()
    == 0
):

    raise ValueError(
        "CLAIM_HISTORY_FLAG contains "
        "no usable values. Training stopped."
    )


# ============================================================
# 14. SAVE AUDIT DATASET
# ============================================================

audit_columns = [
    "POLICY_ID",
    "CUSTOMER_ID",
    "RENEWAL_DUE_DATE",
    "RENEWAL_STATUS",
    "CHURN_FLAG",
] + feature_columns

audit_path = (
    PROCESSED_DIR
    / "renewal_v3_audit_dataset.csv"
)

model_data[
    audit_columns
].to_csv(
    audit_path,
    index=False,
)

print(
    "\nAudit dataset saved:"
)

print(audit_path)


# ============================================================
# 15. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print(
    "\n[8/9] Splitting dataset..."
)

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )
)

print(
    f"Train size: {len(X_train)}"
)

print(
    f"Val size  : {len(X_val)}"
)

print(
    f"Test size : {len(X_test)}"
)

print(
    f"Train churn rate: "
    f"{y_train.mean():.4f}"
)

print(
    f"Val churn rate: "
    f"{y_val.mean():.4f}"
)

print(
    f"Test churn rate: "
    f"{y_test.mean():.4f}"
)


# ============================================================
# 16. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            numeric_transformer,
            numeric_features,
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# ============================================================
# 17. CLASS BALANCE
# ============================================================

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count
    / positive_count
)

print(
    "\nClass distribution:"
)

print(
    f"Negative / Renewed : "
    f"{negative_count}"
)

print(
    f"Positive / Churn   : "
    f"{positive_count}"
)

print(
    f"scale_pos_weight   : "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 18. XGBOOST
# ============================================================

classifier = XGBClassifier(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=3,

    min_child_weight=5,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.30,

    reg_lambda=3.0,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1,

    tree_method="hist",
)


# ============================================================
# 19. PIPELINE
# ============================================================

model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor,
        ),

        (
            "classifier",
            classifier,
        ),
    ]
)


# ============================================================
# 20. TRAIN
# ============================================================

print(
    "\nTraining V3 model..."
)

model.fit(
    X_train,
    y_train,
)

print(
    "Training completed."
)


# ============================================================
# 21. VALIDATION PROBABILITIES
# ============================================================

print(
    "\n[9/9] Finding validation threshold..."
)

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 22. FIND F2 THRESHOLD
# ============================================================

thresholds = np.arange(
    0.05,
    0.951,
    0.005,
)

best_threshold = 0.50

best_f2 = -1.0

for threshold in thresholds:

    val_predictions = (
        val_probabilities
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    beta = 2.0

    if (
        precision == 0
        and recall == 0
    ):

        f2 = 0.0

    else:

        f2 = (
            (1 + beta ** 2)
            * precision
            * recall
            / (
                beta ** 2
                * precision
                + recall
            )
        )

    if f2 > best_f2:

        best_f2 = f2

        best_threshold = float(
            threshold
        )


print(
    f"\nBest validation threshold: "
    f"{best_threshold:.4f}"
)

print(
    f"Best validation F2: "
    f"{best_f2:.4f}"
)


# ============================================================
# 23. TEST PREDICTIONS
# ============================================================

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


# ============================================================
# 24. TEST METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)

accuracy = accuracy_score(
    y_test,
    test_predictions,
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

conf_matrix = confusion_matrix(
    y_test,
    test_predictions,
)


# ============================================================
# 25. RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("V3 MODEL RESULTS")
print("=" * 70)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"Threshold : {best_threshold:.4f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    conf_matrix
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Renewed",
            "Churn",
        ],
        zero_division=0,
    )
)


# ============================================================
# 26. FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating feature importance..."
)

try:

    fitted_preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    fitted_classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    transformed_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    importances = (
        fitted_classifier
        .feature_importances_
    )

    feature_importance = (
        pd.DataFrame(
            {
                "FEATURE":
                    transformed_names,

                "IMPORTANCE":
                    importances,
            }
        )
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "IMPORTANCE",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance_path = (
        PROCESSED_DIR
        / "renewal_v3_feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False,
    )

    print(
        "\nFeature importance saved:"
    )

    print(
        feature_importance_path
    )

    print(
        "\nTop 20 features:"
    )

    print(
        feature_importance
        .head(20)
        .to_string(index=False)
    )

except Exception as exc:

    print(
        "\nWarning: Could not save "
        "feature importance."
    )

    print(
        f"Reason: {exc}"
    )


# ============================================================
# 27. SAVE MODEL
# ============================================================

print(
    "\nSaving model artifacts..."
)

model_path = (
    MODEL_DIR
    / "renewal_v3_xgboost_model.joblib"
)

threshold_path = (
    MODEL_DIR
    / "renewal_v3_churn_threshold.joblib"
)

features_path = (
    MODEL_DIR
    / "renewal_v3_features.joblib"
)

joblib.dump(
    model,
    model_path,
)

joblib.dump(
    best_threshold,
    threshold_path,
)

joblib.dump(
    {
        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "feature_columns":
            feature_columns,

        "target":
            "CHURN_FLAG",

        "target_definition":
            {
                "0": "Renewed",
                "1": "Lapsed or Cancelled",
            },
    },
    features_path,
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("V3 TRAINING COMPLETE")
print("=" * 70)

print(
    "\nCreated files:"
)

print(
    f"1. {model_path}"
)

print(
    f"2. {threshold_path}"
)

print(
    f"3. {features_path}"
)

print(
    f"4. {audit_path}"
)

print(
    f"5. "
    f"{PROCESSED_DIR / 'renewal_v3_feature_importance.csv'}"
)

print(
    "\nFinal test metrics:"
)

print(
    f"ROC-AUC   = {roc_auc:.4f}"
)

print(
    f"PR-AUC    = {pr_auc:.4f}"
)

print(
    f"Threshold = {best_threshold:.4f}"
)

print(
    f"Precision = {precision:.4f}"
)

print(
    f"Recall    = {recall:.4f}"
)

print(
    f"F1        = {f1:.4f}"
)

print(
    "\nDone."
)