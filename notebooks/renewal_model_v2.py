import pandas as pd
from pathlib import Path


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# FILE PATHS
# ==========================================================

RENEWALS_PATH = (
    RAW_DIR
    / "renewals.csv"
)

POLICIES_PATH = (
    RAW_DIR
    / "policies.csv"
)

CUSTOMERS_PATH = (
    RAW_DIR
    / "customers.csv"
)

PREMIUMS_PATH = (
    RAW_DIR
    / "premiums.csv"
)


# ==========================================================
# CHECK FILES
# ==========================================================

required_files = [
    RENEWALS_PATH,
    POLICIES_PATH,
    CUSTOMERS_PATH,
    PREMIUMS_PATH,
]


print("\n========================================")
print("FILE CHECK")
print("========================================")


for file_path in required_files:

    print(
        file_path.name,
        "->",
        file_path.exists(),
    )


missing_files = [
    file_path
    for file_path
    in required_files
    if not file_path.exists()
]


if missing_files:

    raise FileNotFoundError(
        (
            "Missing files: "
            + ", ".join(
                file.name
                for file
                in missing_files
            )
        )
    )


# ==========================================================
# LOAD RAW DATA
# ==========================================================

renewals = pd.read_csv(
    RENEWALS_PATH,
    low_memory=False,
)

policies = pd.read_csv(
    POLICIES_PATH,
    low_memory=False,
)

customers = pd.read_csv(
    CUSTOMERS_PATH,
    low_memory=False,
)

premiums = pd.read_csv(
    PREMIUMS_PATH,
    low_memory=False,
)


print("\n========================================")
print("RAW DATA SHAPES")
print("========================================")


print(
    "Renewals :",
    renewals.shape,
)

print(
    "Policies :",
    policies.shape,
)

print(
    "Customers:",
    customers.shape,
)

print(
    "Premiums :",
    premiums.shape,
)


# ==========================================================
# RENEWAL TARGET
# ==========================================================

print("\n========================================")
print("ORIGINAL RENEWAL STATUS")
print("========================================")


print(
    renewals[
        "RENEWAL_STATUS"
    ].value_counts(
        dropna=False
    )
)


# ----------------------------------------------------------
# IMPORTANT:
#
# Renewed
#     -> 1
#
# Lapsed / Cancelled
#     -> 0
#
# Pending
#     -> remove because outcome is unknown
# ----------------------------------------------------------


finalized = renewals[
    renewals[
        "RENEWAL_STATUS"
    ].isin(
        [
            "Renewed",
            "Lapsed",
            "Cancelled",
        ]
    )
].copy()


finalized[
    "RENEWED_FLAG"
] = (
    finalized[
        "RENEWAL_STATUS"
    ]
    ==
    "Renewed"
).astype(int)


print("\n========================================")
print("FINAL TARGET")
print("========================================")


print(
    "1 = Renewed"
)

print(
    "0 = Not Renewed"
)


print(
    "\nFinalized rows:",
    len(finalized),
)


print(
    "\nTarget counts:"
)


print(
    finalized[
        "RENEWED_FLAG"
    ].value_counts()
)


print(
    "\nTarget percentages:"
)


print(
    finalized[
        "RENEWED_FLAG"
    ]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ==========================================================
# REMOVE LEAKAGE COLUMN
# ==========================================================

# RENEWAL_PROBABILITY already contains
# an outcome-related probability generated
# by the source system.
#
# We do NOT use it as a model feature.

if (
    "RENEWAL_PROBABILITY"
    in finalized.columns
):

    finalized = finalized.drop(
        columns=[
            "RENEWAL_PROBABILITY"
        ]
    )


# ==========================================================
# DATE CONVERSION
# ==========================================================

finalized[
    "RENEWAL_DUE_DATE"
] = pd.to_datetime(
    finalized[
        "RENEWAL_DUE_DATE"
    ],
    errors="coerce",
)


premiums[
    "DUE_DATE"
] = pd.to_datetime(
    premiums[
        "DUE_DATE"
    ],
    errors="coerce",
)


premiums[
    "PAID_DATE"
] = pd.to_datetime(
    premiums[
        "PAID_DATE"
    ],
    errors="coerce",
)


# ==========================================================
# MERGE POLICY INFORMATION
# ==========================================================

# CUSTOMER_ID already exists in renewals.
# Avoid duplicate CUSTOMER_ID from policies.

policy_columns = [
    column
    for column
    in policies.columns
    if column
    not in [
        "CUSTOMER_ID"
    ]
]


model_df = finalized.merge(

    policies[
        policy_columns
    ],

    on="POLICY_ID",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# MERGE CUSTOMER INFORMATION
# ==========================================================

customer_columns = [
    column
    for column
    in customers.columns
    if column
    not in [
        "CUSTOMER_NAME"
    ]
]


model_df = model_df.merge(

    customers[
        customer_columns
    ],

    on="CUSTOMER_ID",

    how="left",

    validate="many_to_one",
)


# ==========================================================
# TIME-SAFE PAYMENT FEATURES
# ==========================================================

# Critical point:
#
# We only use premium transactions that occurred
# BEFORE or ON the renewal due date.
#
# Future transactions must not be used because
# that would create leakage.


renewal_dates = finalized[
    [
        "POLICY_ID",
        "RENEWAL_DUE_DATE",
    ]
].copy()


premium_history = premiums.merge(

    renewal_dates,

    on="POLICY_ID",

    how="inner",
)


premium_history = premium_history[
    premium_history[
        "DUE_DATE"
    ]
    <=
    premium_history[
        "RENEWAL_DUE_DATE"
    ]
].copy()


print("\n========================================")
print("TIME-SAFE PREMIUM HISTORY")
print("========================================")


print(
    "Prior premium transactions:",
    len(premium_history),
)


# ==========================================================
# PAYMENT AGGREGATION
# ==========================================================

payment_features = (
    premium_history

    .groupby(
        "POLICY_ID"
    )

    .agg(

        TOTAL_PAYMENT_TXNS_PRIOR=(
            "PREMIUM_TXN_ID",
            "count",
        ),

        TOTAL_PREMIUM_AMOUNT_PRIOR=(
            "PREMIUM_AMOUNT",
            "sum",
        ),

        PAID_TXNS_PRIOR=(
            "PAYMENT_STATUS",
            lambda x:
                (
                    x
                    ==
                    "Paid"
                ).sum(),
        ),

        FAILED_TXNS_PRIOR=(
            "PAYMENT_STATUS",
            lambda x:
                (
                    x
                    ==
                    "Failed"
                ).sum(),
        ),

        OVERDUE_TXNS_PRIOR=(
            "PAYMENT_STATUS",
            lambda x:
                (
                    x
                    ==
                    "Overdue"
                ).sum(),
        ),

        PENDING_TXNS_PRIOR=(
            "PAYMENT_STATUS",
            lambda x:
                (
                    x
                    ==
                    "Pending"
                ).sum(),
        ),

        AVG_PAYMENT_DELAY_PRIOR=(
            "PAYMENT_DELAY_DAYS",
            "mean",
        ),

        MAX_PAYMENT_DELAY_PRIOR=(
            "PAYMENT_DELAY_DAYS",
            "max",
        ),
    )

    .reset_index()
)


# ==========================================================
# PAYMENT SUCCESS RATE
# ==========================================================

payment_features[
    "PAYMENT_SUCCESS_RATE_PRIOR"
] = (
    payment_features[
        "PAID_TXNS_PRIOR"
    ]
    /
    payment_features[
        "TOTAL_PAYMENT_TXNS_PRIOR"
    ]
)


# ==========================================================
# UNPAID TRANSACTIONS
# ==========================================================

payment_features[
    "UNPAID_TXNS_AT_RENEWAL"
] = (

    payment_features[
        "FAILED_TXNS_PRIOR"
    ]

    +

    payment_features[
        "OVERDUE_TXNS_PRIOR"
    ]

    +

    payment_features[
        "PENDING_TXNS_PRIOR"
    ]
)


payment_features[
    "HAS_UNPAID_PAYMENT_AT_RENEWAL"
] = (
    payment_features[
        "UNPAID_TXNS_AT_RENEWAL"
    ]
    >
    0
).astype(int)


# ==========================================================
# MERGE PAYMENT FEATURES
# ==========================================================

model_df = model_df.merge(

    payment_features,

    on="POLICY_ID",

    how="left",

    validate="one_to_one",
)


# ==========================================================
# PAYMENT HISTORY FLAG
# ==========================================================

model_df[
    "HAS_PAYMENT_HISTORY_PRIOR"
] = (
    model_df[
        "TOTAL_PAYMENT_TXNS_PRIOR"
    ]
    .notna()
    .astype(int)
)


# ==========================================================
# SAFE ZERO FILLS
# ==========================================================

payment_count_columns = [

    "TOTAL_PAYMENT_TXNS_PRIOR",

    "PAID_TXNS_PRIOR",

    "FAILED_TXNS_PRIOR",

    "OVERDUE_TXNS_PRIOR",

    "PENDING_TXNS_PRIOR",

    "UNPAID_TXNS_AT_RENEWAL",

    "HAS_UNPAID_PAYMENT_AT_RENEWAL",
]


for column in payment_count_columns:

    if column in model_df.columns:

        model_df[
            column
        ] = (
            model_df[
                column
            ]
            .fillna(0)
        )


# ==========================================================
# DATASET CHECK
# ==========================================================

print("\n========================================")
print("MODEL DATASET")
print("========================================")


print(
    "Rows:",
    len(model_df),
)


print(
    "Unique Policies:",
    model_df[
        "POLICY_ID"
    ].nunique(),
)


print(
    "Duplicate Policies:",
    model_df[
        "POLICY_ID"
    ]
    .duplicated()
    .sum(),
)


print(
    "\nColumns:",
    len(
        model_df.columns
    ),
)


# ==========================================================
# PREMIUM INCREASE ANALYSIS
# ==========================================================

print("\n========================================")
print("PREMIUM INCREASE VS RENEWAL")
print("========================================")


premium_analysis = model_df[
    [
        "PREMIUM_INCREASE_PCT",
        "RENEWED_FLAG",
    ]
].copy()


premium_analysis[
    "PREMIUM_BAND"
] = pd.cut(

    premium_analysis[
        "PREMIUM_INCREASE_PCT"
    ],

    bins=[
        -float("inf"),
        5,
        10,
        15,
        float("inf"),
    ],

    labels=[
        "Below 5%",
        "5-10%",
        "10-15%",
        "15%+",
    ],

    right=False,
)


premium_result = (
    premium_analysis

    .groupby(
        "PREMIUM_BAND",
        observed=False,
    )

    .agg(

        POLICIES=(
            "RENEWED_FLAG",
            "count",
        ),

        RENEWAL_RATE=(
            "RENEWED_FLAG",
            "mean",
        ),
    )
)


premium_result[
    "RENEWAL_RATE"
] = (
    premium_result[
        "RENEWAL_RATE"
    ]
    * 100
).round(2)


print(
    premium_result
)


# ==========================================================
# LOYALTY ANALYSIS
# ==========================================================

if (
    "LOYALTY_YEARS"
    in model_df.columns
):

    print("\n========================================")
    print("LOYALTY VS RENEWAL")
    print("========================================")


    loyalty_df = model_df[
        [
            "LOYALTY_YEARS",
            "RENEWED_FLAG",
        ]
    ].copy()


    loyalty_df[
        "LOYALTY_BAND"
    ] = pd.cut(

        loyalty_df[
            "LOYALTY_YEARS"
        ],

        bins=[
            -1,
            1,
            3,
            5,
            10,
            float("inf"),
        ],

        labels=[
            "0-1 Year",
            "2-3 Years",
            "4-5 Years",
            "6-10 Years",
            "10+ Years",
        ],
    )


    loyalty_result = (
        loyalty_df

        .groupby(
            "LOYALTY_BAND",
            observed=False,
        )

        .agg(

            POLICIES=(
                "RENEWED_FLAG",
                "count",
            ),

            RENEWAL_RATE=(
                "RENEWED_FLAG",
                "mean",
            ),
        )
    )


    loyalty_result[
        "RENEWAL_RATE"
    ] = (
        loyalty_result[
            "RENEWAL_RATE"
        ]
        * 100
    ).round(2)


    print(
        loyalty_result
    )


# ==========================================================
# PAYMENT DELAY ANALYSIS
# ==========================================================

print("\n========================================")
print("PAYMENT DELAY VS RENEWAL")
print("========================================")


payment_delay_df = model_df[
    model_df[
        "AVG_PAYMENT_DELAY_PRIOR"
    ].notna()
][
    [
        "AVG_PAYMENT_DELAY_PRIOR",
        "RENEWED_FLAG",
    ]
].copy()


payment_delay_df[
    "DELAY_BAND"
] = pd.cut(

    payment_delay_df[
        "AVG_PAYMENT_DELAY_PRIOR"
    ],

    bins=[
        -float("inf"),
        0,
        7,
        15,
        30,
        float("inf"),
    ],

    labels=[
        "On Time",
        "1-7 Days",
        "8-15 Days",
        "16-30 Days",
        "30+ Days",
    ],
)


delay_result = (
    payment_delay_df

    .groupby(
        "DELAY_BAND",
        observed=False,
    )

    .agg(

        POLICIES=(
            "RENEWED_FLAG",
            "count",
        ),

        RENEWAL_RATE=(
            "RENEWED_FLAG",
            "mean",
        ),
    )
)


delay_result[
    "RENEWAL_RATE"
] = (
    delay_result[
        "RENEWAL_RATE"
    ]
    * 100
).round(2)


print(
    delay_result
)


# ==========================================================
# UNPAID PAYMENT ANALYSIS
# ==========================================================

print("\n========================================")
print("UNPAID PAYMENT VS RENEWAL")
print("========================================")


unpaid_result = (
    model_df

    .groupby(
        "HAS_UNPAID_PAYMENT_AT_RENEWAL"
    )

    .agg(

        POLICIES=(
            "RENEWED_FLAG",
            "count",
        ),

        RENEWAL_RATE=(
            "RENEWED_FLAG",
            "mean",
        ),
    )
)


unpaid_result[
    "RENEWAL_RATE"
] = (
    unpaid_result[
        "RENEWAL_RATE"
    ]
    * 100
).round(2)


print(
    unpaid_result
)


# ==========================================================
# PAYMENT SUCCESS RATE
# ==========================================================

print("\n========================================")
print("PAYMENT SUCCESS VS RENEWAL")
print("========================================")


success_df = model_df[
    model_df[
        "PAYMENT_SUCCESS_RATE_PRIOR"
    ].notna()
][
    [
        "PAYMENT_SUCCESS_RATE_PRIOR",
        "RENEWED_FLAG",
    ]
].copy()


success_df[
    "SUCCESS_BAND"
] = pd.cut(

    success_df[
        "PAYMENT_SUCCESS_RATE_PRIOR"
    ],

    bins=[
        -0.01,
        0.5,
        0.8,
        0.95,
        1.01,
    ],

    labels=[
        "0-50%",
        "50-80%",
        "80-95%",
        "95-100%",
    ],
)


success_result = (
    success_df

    .groupby(
        "SUCCESS_BAND",
        observed=False,
    )

    .agg(

        POLICIES=(
            "RENEWED_FLAG",
            "count",
        ),

        RENEWAL_RATE=(
            "RENEWED_FLAG",
            "mean",
        ),
    )
)


success_result[
    "RENEWAL_RATE"
] = (
    success_result[
        "RENEWAL_RATE"
    ]
    * 100
).round(2)


print(
    success_result
)


# ==========================================================
# SUSPICIOUS CURRENT FEATURES
# ==========================================================

print("\n========================================")
print("CURRENT FEATURES TO REVIEW")
print("========================================")


review_features = [

    "CREDIT_SCORE",

    "RISK_SCORE",
]


for column in review_features:

    if column not in model_df.columns:

        continue


    print(
        f"\n--- {column} ---"
    )


    renewed_mean = (
        model_df.loc[
            model_df[
                "RENEWED_FLAG"
            ]
            ==
            1,
            column,
        ]
        .mean()
    )


    nonrenewed_mean = (
        model_df.loc[
            model_df[
                "RENEWED_FLAG"
            ]
            ==
            0,
            column,
        ]
        .mean()
    )


    print(
        "Renewed Mean:",
        round(
            renewed_mean,
            2,
        ),
    )


    print(
        "Not Renewed Mean:",
        round(
            nonrenewed_mean,
            2,
        ),
    )


# ==========================================================
# CUSTOMER RISK SEGMENT
# ==========================================================

if (
    "CUSTOMER_RISK_SEGMENT"
    in model_df.columns
):

    print("\n========================================")
    print("CUSTOMER RISK SEGMENT VS RENEWAL")
    print("========================================")


    result = (
        model_df

        .groupby(
            "CUSTOMER_RISK_SEGMENT"
        )

        .agg(

            POLICIES=(
                "RENEWED_FLAG",
                "count",
            ),

            RENEWAL_RATE=(
                "RENEWED_FLAG",
                "mean",
            ),
        )
    )


    result[
        "RENEWAL_RATE"
    ] = (
        result[
            "RENEWAL_RATE"
        ]
        * 100
    ).round(2)


    print(
        result
    )


# ==========================================================
# CONTACT CHANNEL
# ==========================================================

if (
    "CONTACT_CHANNEL"
    in model_df.columns
):

    print("\n========================================")
    print("CONTACT CHANNEL VS RENEWAL")
    print("========================================")


    result = (
        model_df

        .groupby(
            "CONTACT_CHANNEL"
        )

        .agg(

            POLICIES=(
                "RENEWED_FLAG",
                "count",
            ),

            RENEWAL_RATE=(
                "RENEWED_FLAG",
                "mean",
            ),
        )
    )


    result[
        "RENEWAL_RATE"
    ] = (
        result[
            "RENEWAL_RATE"
        ]
        * 100
    ).round(2)


    print(
        result.sort_values(
            "RENEWAL_RATE",
            ascending=False,
        )
    )


# ==========================================================
# POLICY TYPE
# ==========================================================

if (
    "POLICY_TYPE"
    in model_df.columns
):

    print("\n========================================")
    print("POLICY TYPE VS RENEWAL")
    print("========================================")


    result = (
        model_df

        .groupby(
            "POLICY_TYPE"
        )

        .agg(

            POLICIES=(
                "RENEWED_FLAG",
                "count",
            ),

            RENEWAL_RATE=(
                "RENEWED_FLAG",
                "mean",
            ),
        )
    )


    result[
        "RENEWAL_RATE"
    ] = (
        result[
            "RENEWAL_RATE"
        ]
        * 100
    ).round(2)


    print(
        result.sort_values(
            "RENEWAL_RATE",
            ascending=False,
        )
    )


# ==========================================================
# PROPOSED V2 FEATURES
# ==========================================================

candidate_features = [

    "PREMIUM_INCREASE_PCT",

    "LOYALTY_YEARS",

    "OLD_PREMIUM",

    "NEW_PREMIUM",

    "TOTAL_PAYMENT_TXNS_PRIOR",

    "PAID_TXNS_PRIOR",

    "FAILED_TXNS_PRIOR",

    "OVERDUE_TXNS_PRIOR",

    "UNPAID_TXNS_AT_RENEWAL",

    "AVG_PAYMENT_DELAY_PRIOR",

    "MAX_PAYMENT_DELAY_PRIOR",

    "PAYMENT_SUCCESS_RATE_PRIOR",

    "HAS_PAYMENT_HISTORY_PRIOR",

    "HAS_UNPAID_PAYMENT_AT_RENEWAL",

    "POLICY_TYPE",

    "PAYMENT_MODE",

    "SALES_CHANNEL",

    "CONTACT_CHANNEL",
]


available_features = [
    column
    for column
    in candidate_features
    if column
    in model_df.columns
]


print("\n========================================")
print("V2 CANDIDATE FEATURES")
print("========================================")


for column in available_features:

    print(
        "KEEP FOR NEXT TEST:",
        column,
    )


# ==========================================================
# FEATURES WE SHOULD NOT AUTOMATICALLY USE
# ==========================================================

review_or_remove = [

    "GENDER",

    "MARITAL_STATUS",

    "OCCUPATION",

    "STATE",

    "CREDIT_SCORE",

    "RISK_SCORE",

    "CUSTOMER_RISK_SEGMENT",

    "RISK_BAND",

    "RENEWAL_PROBABILITY",
]


print("\n========================================")
print("FEATURES TO REMOVE / REVIEW")
print("========================================")


for column in review_or_remove:

    if column in model_df.columns:

        print(
            "REVIEW:",
            column,
        )


# ==========================================================
# SAVE AUDIT DATASET
# ==========================================================

OUTPUT_PATH = (
    PROCESSED_DIR
    / "renewal_v2_audit_dataset.csv"
)


model_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n========================================")
print("AUDIT COMPLETE")
print("========================================")


print(
    "Saved:",
    OUTPUT_PATH,
)


print(
    "\nNext step:"
)


print(
    (
        "Use this audited dataset to train "
        "Renewal Model V2."
    )
)