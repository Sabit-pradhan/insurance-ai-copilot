import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "renewal_v2_audit_dataset.csv"
)


MODELS_DIR = (
    PROJECT_DIR
    / "models"
)


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# LOAD AUDITED DATASET
# ==========================================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)


print("\n========================================")
print("RENEWAL MODEL V2")
print("========================================")


print(
    "Dataset Shape:",
    df.shape,
)


# ==========================================================
# TARGET CHECK
# ==========================================================

if (
    "RENEWED_FLAG"
    not in df.columns
):

    raise ValueError(
        "RENEWED_FLAG is missing."
    )


# ==========================================================
# CREATE BUSINESS TARGET
# ==========================================================
#
# OLD TARGET:
#
# 1 = Renewed
# 0 = Not Renewed
#
#
# NEW MODEL TARGET:
#
# CHURN_FLAG = 1
# means customer DID NOT renew.
#
# This makes class 1 the business-risk class.
# ==========================================================

df[
    "CHURN_FLAG"
] = (
    1
    -
    df[
        "RENEWED_FLAG"
    ]
)


print("\nTarget Definition:")

print(
    "0 = Renewed"
)

print(
    "1 = Not Renewed / Churn"
)


print("\nTarget Counts:")

print(
    df[
        "CHURN_FLAG"
    ].value_counts()
)


print("\nTarget Percentages:")

print(
    (
        df[
            "CHURN_FLAG"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )
)


# ==========================================================
# V2 FEATURE SELECTION
# ==========================================================
#
# We are intentionally removing:
#
# GENDER
# MARITAL_STATUS
# OCCUPATION
# STATE
# CREDIT_SCORE
# RISK_SCORE
# CUSTOMER_RISK_SEGMENT
# RISK_BAND
#
# because our audit did not show a useful
# renewal relationship for these variables.
# ==========================================================


numeric_features = [

    "OLD_PREMIUM",

    "NEW_PREMIUM",

    "PREMIUM_INCREASE_PCT",

    "LOYALTY_YEARS",

    "TOTAL_PAYMENT_TXNS_PRIOR",

    "PAID_TXNS_PRIOR",

    "FAILED_TXNS_PRIOR",

    "OVERDUE_TXNS_PRIOR",

    "AVG_PAYMENT_DELAY_PRIOR",

    "MAX_PAYMENT_DELAY_PRIOR",

    "HAS_PAYMENT_HISTORY_PRIOR",
]


categorical_features = [

    "POLICY_TYPE",

    "PAYMENT_MODE",

    "SALES_CHANNEL",

    "CONTACT_CHANNEL",
]


# ==========================================================
# CHECK FEATURES
# ==========================================================

all_features = (
    numeric_features
    +
    categorical_features
)


missing_features = [

    column

    for column
    in all_features

    if column
    not in df.columns
]


if missing_features:

    print(
        "\nMissing Features:"
    )


    for column in missing_features:

        print(
            column
        )


    raise ValueError(
        (
            "Some V2 features are missing "
            "from the dataset."
        )
    )


print("\n========================================")
print("V2 FEATURES")
print("========================================")


print(
    "\nNumeric Features:"
)


for column in numeric_features:

    print(
        "-",
        column,
    )


print(
    "\nCategorical Features:"
)


for column in categorical_features:

    print(
        "-",
        column,
    )


print(
    "\nTotal Features:",
    len(
        all_features
    ),
)


# ==========================================================
# X AND Y
# ==========================================================

X = df[
    all_features
].copy()


y = df[
    "CHURN_FLAG"
].copy()


# ==========================================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================================
#
# We create:
#
# 70% training
# 15% validation
# 15% test
#
# Validation:
# used for threshold selection.
#
# Test:
# untouched final evaluation.
# ==========================================================


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


print("\n========================================")
print("DATA SPLIT")
print("========================================")


print(
    "Train:",
    X_train.shape,
)


print(
    "Validation:",
    X_val.shape,
)


print(
    "Test:",
    X_test.shape,
)


print(
    "\nTrain Churn Rate:",
    round(
        y_train.mean()
        * 100,
        2,
    ),
    "%",
)


print(
    "Validation Churn Rate:",
    round(
        y_val.mean()
        * 100,
        2,
    ),
    "%",
)


print(
    "Test Churn Rate:",
    round(
        y_test.mean()
        * 100,
        2,
    ),
    "%",
)


# ==========================================================
# NUMERIC PREPROCESSING
# ==========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)


# ==========================================================
# CATEGORICAL PREPROCESSING
# ==========================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


# ==========================================================
# COLUMN TRANSFORMER
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ==========================================================
# CLASS IMBALANCE
# ==========================================================
#
# Target distribution:
#
# Renewed      ~90%
# Churn        ~10%
#
# Therefore simply optimizing accuracy
# would be misleading.
#
# scale_pos_weight gives more importance
# to churn cases.
# ==========================================================

negative_count = (
    y_train
    ==
    0
).sum()


positive_count = (
    y_train
    ==
    1
).sum()


scale_pos_weight = (
    negative_count
    /
    positive_count
)


print("\n========================================")
print("CLASS IMBALANCE")
print("========================================")


print(
    "Renewed Cases:",
    negative_count,
)


print(
    "Churn Cases:",
    positive_count,
)


print(
    "Scale Pos Weight:",
    round(
        scale_pos_weight,
        2,
    ),
)


# ==========================================================
# XGBOOST MODEL
# ==========================================================

xgb_model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=3,

    min_child_weight=5,

    subsample=0.80,

    colsample_bytree=0.80,

    reg_alpha=0.20,

    reg_lambda=2.0,

    objective=(
        "binary:logistic"
    ),

    eval_metric=(
        "logloss"
    ),

    scale_pos_weight=(
        scale_pos_weight
    ),

    random_state=42,

    n_jobs=-1,
)


# ==========================================================
# FULL PIPELINE
# ==========================================================

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),

        (
            "model",
            xgb_model,
        ),
    ]
)


# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n========================================")
print("TRAINING MODEL")
print("========================================")


model_pipeline.fit(
    X_train,
    y_train,
)


print(
    "Model Training Complete."
)


# ==========================================================
# VERIFY MODEL CLASSES
# ==========================================================

trained_model = (
    model_pipeline
    .named_steps[
        "model"
    ]
)


print("\n========================================")
print("MODEL CLASS ORIENTATION")
print("========================================")


print(
    "Classes:",
    trained_model.classes_,
)


print(
    """
IMPORTANT:

Class 0 = Renewed
Class 1 = Churn / Not Renewed

Therefore:

predict_proba()[:, 1]

is directly the churn probability.
"""
)


# ==========================================================
# VALIDATION PROBABILITIES
# ==========================================================

val_probability = (
    model_pipeline
    .predict_proba(
        X_val
    )[
        :,
        1
    ]
)


# ==========================================================
# FIND BUSINESS THRESHOLD
# ==========================================================
#
# We use F2 instead of F1.
#
# F2 gives more importance to recall.
#
# Business meaning:
#
# Missing an actual churn customer
# is more costly than contacting
# some extra customers.
# ==========================================================

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val,
        val_probability,
    )
)


# precision / recall contain one extra value
# compared with thresholds.

precision_for_threshold = (
    precision_values[:-1]
)


recall_for_threshold = (
    recall_values[:-1]
)


beta = 2


f2_scores = (

    (
        1
        +
        beta ** 2
    )

    *

    (
        precision_for_threshold
        *
        recall_for_threshold
    )

    /

    (
        (
            beta ** 2
            *
            precision_for_threshold
        )

        +

        recall_for_threshold

        +

        1e-9
    )
)


best_index = (
    np.argmax(
        f2_scores
    )
)


best_threshold = float(
    thresholds[
        best_index
    ]
)


print("\n========================================")
print("THRESHOLD TUNING")
print("========================================")


print(
    "Selected Threshold:",
    round(
        best_threshold,
        4,
    ),
)


print(
    "Validation Precision:",
    round(
        precision_for_threshold[
            best_index
        ],
        4,
    ),
)


print(
    "Validation Recall:",
    round(
        recall_for_threshold[
            best_index
        ],
        4,
    ),
)


print(
    "Validation F2:",
    round(
        f2_scores[
            best_index
        ],
        4,
    ),
)


# ==========================================================
# TEST SET PROBABILITIES
# ==========================================================

test_probability = (
    model_pipeline
    .predict_proba(
        X_test
    )[
        :,
        1
    ]
)


# ==========================================================
# DEFAULT 0.50 PREDICTION
# ==========================================================

test_prediction_050 = (

    test_probability
    >=
    0.50

).astype(int)


# ==========================================================
# TUNED THRESHOLD PREDICTION
# ==========================================================

test_prediction_tuned = (

    test_probability
    >=
    best_threshold

).astype(int)


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    y_true,
    predictions,
    probabilities,
    name,
):

    print(
        "\n========================================"
    )

    print(
        name
    )

    print(
        "========================================"
    )


    accuracy = accuracy_score(
        y_true,
        predictions,
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )


    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )


    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )


    print(
        "Accuracy :",
        round(
            accuracy,
            4,
        ),
    )


    print(
        "Precision:",
        round(
            precision,
            4,
        ),
    )


    print(
        "Recall   :",
        round(
            recall,
            4,
        ),
    )


    print(
        "F1       :",
        round(
            f1,
            4,
        ),
    )


    print(
        "ROC-AUC  :",
        round(
            roc_auc,
            4,
        ),
    )


    print(
        "PR-AUC   :",
        round(
            pr_auc,
            4,
        ),
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        confusion_matrix(
            y_true,
            predictions,
        )
    )


    print(
        "\nClassification Report:"
    )


    print(
        classification_report(

            y_true,

            predictions,

            target_names=[
                "Renewed",
                "Churn",
            ],

            zero_division=0,
        )
    )


# ==========================================================
# EVALUATE DEFAULT THRESHOLD
# ==========================================================

evaluate_model(

    y_test,

    test_prediction_050,

    test_probability,

    "TEST RESULTS - THRESHOLD 0.50",
)


# ==========================================================
# EVALUATE TUNED THRESHOLD
# ==========================================================

evaluate_model(

    y_test,

    test_prediction_tuned,

    test_probability,

    (
        "TEST RESULTS - "
        f"TUNED THRESHOLD {best_threshold:.4f}"
    ),
)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\n========================================")
print("TOP FEATURE IMPORTANCE")
print("========================================")


fitted_preprocessor = (
    model_pipeline
    .named_steps[
        "preprocessor"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


importance_values = (
    trained_model
    .feature_importances_
)


importance_df = pd.DataFrame(
    {
        "FEATURE":
            feature_names,

        "IMPORTANCE":
            importance_values,
    }
)


importance_df = (
    importance_df
    .sort_values(
        "IMPORTANCE",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print(
    importance_df.head(
        25
    )
)


# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

importance_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "renewal_v2_feature_importance.csv"
)


importance_df.to_csv(
    importance_path,
    index=False,
)


# ==========================================================
# SAVE NEW MODEL
# ==========================================================
#
# IMPORTANT:
#
# We DO NOT overwrite the old model yet.
#
# First validate V2.
# ==========================================================

model_path = (
    MODELS_DIR
    / "renewal_v2_xgboost_model.joblib"
)


threshold_path = (
    MODELS_DIR
    / "renewal_v2_churn_threshold.joblib"
)


features_path = (
    MODELS_DIR
    / "renewal_v2_features.joblib"
)


joblib.dump(
    model_pipeline,
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

        "all_features":
            all_features,

        "target":
            "CHURN_FLAG",

        "target_definition": {
            0:
                "Renewed",

            1:
                "Not Renewed / Churn",
        },
    },
    features_path,
)


# ==========================================================
# COMPLETE
# ==========================================================

print("\n========================================")
print("MODEL V2 SAVED")
print("========================================")


print(
    "Model:"
)

print(
    model_path
)


print(
    "\nThreshold:"
)

print(
    threshold_path
)


print(
    "\nFeature Metadata:"
)

print(
    features_path
)


print(
    "\nFeature Importance:"
)

print(
    importance_path
)


print(
    "\nIMPORTANT:"
)

print(
    (
        "Old production renewal model "
        "has NOT been overwritten."
    )
)


print(
    (
        "We will compare V2 results "
        "before replacing it."
    )
)