# src/explainability/fraud_explainer.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_xgboost_model.joblib"
)


# ==========================================================
# LOAD MODEL PIPELINE
# ==========================================================

fraud_pipeline = (
    joblib.load(
        MODEL_PATH
    )
)


# ==========================================================
# VALIDATE PIPELINE
# ==========================================================

if not hasattr(
    fraud_pipeline,
    "named_steps",
):

    raise ValueError(
        "Fraud model must be a scikit-learn Pipeline."
    )


if (
    "preprocessor"
    not in fraud_pipeline.named_steps
):

    raise ValueError(
        "Fraud pipeline does not contain "
        "'preprocessor' step."
    )


if (
    "model"
    not in fraud_pipeline.named_steps
):

    raise ValueError(
        "Fraud pipeline does not contain "
        "'model' step."
    )


# ==========================================================
# PIPELINE COMPONENTS
# ==========================================================

preprocessor = (
    fraud_pipeline
    .named_steps[
        "preprocessor"
    ]
)

xgb_model = (
    fraud_pipeline
    .named_steps[
        "model"
    ]
)


# ==========================================================
# SHAP EXPLAINER
# ==========================================================

shap_explainer = (
    shap.TreeExplainer(
        xgb_model
    )
)


# ==========================================================
# RAW MODEL FEATURES
# ==========================================================

def get_expected_features():

    if not hasattr(
        fraud_pipeline,
        "feature_names_in_",
    ):

        raise ValueError(
            "Fraud pipeline does not expose "
            "feature_names_in_."
        )

    return list(
        fraud_pipeline
        .feature_names_in_
    )


# ==========================================================
# TRANSFORMED FEATURES
# ==========================================================

def get_transformed_feature_names():

    if not hasattr(
        preprocessor,
        "get_feature_names_out",
    ):

        raise ValueError(
            "Preprocessor does not support "
            "get_feature_names_out()."
        )

    return list(
        preprocessor
        .get_feature_names_out()
    )


# ==========================================================
# CLEAN TRANSFORMED FEATURE NAME
# ==========================================================

def clean_feature_name(
    transformed_name: str,
    raw_features: list[str],
):

    """
    Example:

    num__CLAIM_AMOUNT
    ->
    CLAIM_AMOUNT

    cat__STATE_Karnataka
    ->
    STATE
    """

    feature_name = (
        transformed_name
        .replace(
            "num__",
            "",
        )
        .replace(
            "cat__",
            "",
        )
    )


    # Longest raw feature first.
    # Important for features containing underscores.

    sorted_features = sorted(
        raw_features,
        key=len,
        reverse=True,
    )


    for raw_feature in sorted_features:

        if (
            feature_name
            ==
            raw_feature
        ):

            return raw_feature


        if feature_name.startswith(
            raw_feature + "_"
        ):

            return raw_feature


    return feature_name


# ==========================================================
# GET SHAP VALUES
# ==========================================================

def get_shap_values(
    transformed_data,
):

    """
    Return SHAP values for one observation.

    Handles different SHAP output shapes.
    """

    # Convert sparse matrix to dense

    if hasattr(
        transformed_data,
        "toarray",
    ):

        transformed_data = (
            transformed_data
            .toarray()
        )


    shap_values = (
        shap_explainer
        .shap_values(
            transformed_data
        )
    )


    # ------------------------------------------------------
    # OLD SHAP FORMAT
    # list[class_0, class_1]
    # ------------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):

        if len(
            shap_values
        ) > 1:

            shap_values = (
                shap_values[1]
            )

        else:

            shap_values = (
                shap_values[0]
            )


    shap_values = (
        np.asarray(
            shap_values
        )
    )


    # ------------------------------------------------------
    # 3D:
    # rows x features x classes
    # ------------------------------------------------------

    if (
        shap_values.ndim
        ==
        3
    ):

        if (
            shap_values.shape[2]
            >
            1
        ):

            shap_values = (
                shap_values[
                    0,
                    :,
                    1,
                ]
            )

        else:

            shap_values = (
                shap_values[
                    0,
                    :,
                    0,
                ]
            )


    # ------------------------------------------------------
    # 2D:
    # rows x features
    # ------------------------------------------------------

    elif (
        shap_values.ndim
        ==
        2
    ):

        shap_values = (
            shap_values[0]
        )


    # ------------------------------------------------------
    # 1D
    # ------------------------------------------------------

    elif (
        shap_values.ndim
        ==
        1
    ):

        pass


    else:

        raise ValueError(
            "Unexpected SHAP value shape: "
            f"{shap_values.shape}"
        )


    return (
        np.asarray(
            shap_values,
            dtype=float,
        )
    )


# ==========================================================
# GET BASE VALUE
# ==========================================================

def get_base_value():

    expected_value = (
        shap_explainer
        .expected_value
    )


    expected_value = (
        np.asarray(
            expected_value
        )
    )


    # ------------------------------------------------------
    # SCALAR
    # ------------------------------------------------------

    if (
        expected_value.ndim
        ==
        0
    ):

        return float(
            expected_value
        )


    # ------------------------------------------------------
    # ARRAY
    # ------------------------------------------------------

    flattened = (
        expected_value
        .reshape(-1)
    )


    # Binary model with separate values
    if len(
        flattened
    ) > 1:

        return float(
            flattened[1]
        )


    return float(
        flattened[0]
    )


# ==========================================================
# EXPLAIN FRAUD DATAFRAME
# ==========================================================

def explain_fraud_dataframe(
    model_input: pd.DataFrame,
    top_n: int = 8,
):

    """
    Explain one fraud prediction using SHAP.
    """

    # ------------------------------------------------------
    # VALIDATE INPUT
    # ------------------------------------------------------

    if not isinstance(
        model_input,
        pd.DataFrame,
    ):

        raise TypeError(
            "model_input must be a pandas DataFrame."
        )


    if len(
        model_input
    ) != 1:

        raise ValueError(
            "Fraud explanation currently supports "
            "one row at a time."
        )


    # ------------------------------------------------------
    # EXPECTED RAW FEATURES
    # ------------------------------------------------------

    expected_features = (
        get_expected_features()
    )


    missing_features = [

        feature

        for feature
        in expected_features

        if feature
        not in model_input.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing fraud model features: "
            f"{missing_features}"
        )


    # Keep exact training order

    model_input = (
        model_input[
            expected_features
        ]
        .copy()
    )


    # ------------------------------------------------------
    # MODEL PROBABILITY
    # ------------------------------------------------------

    probabilities = (
        fraud_pipeline
        .predict_proba(
            model_input
        )[0]
    )


    classes = list(
        fraud_pipeline
        .classes_
    )


    # Positive class = fraud risk

    if (
        1
        in classes
    ):

        positive_index = (
            classes.index(
                1
            )
        )

    else:

        positive_index = (
            len(classes)
            -
            1
        )


    fraud_probability = float(
        probabilities[
            positive_index
        ]
    )


    non_fraud_probability = (
        1.0
        -
        fraud_probability
    )


    # ------------------------------------------------------
    # TRANSFORM INPUT
    # ------------------------------------------------------

    transformed_data = (
        preprocessor
        .transform(
            model_input
        )
    )


    transformed_features = (
        get_transformed_feature_names()
    )


    # ------------------------------------------------------
    # SHAP VALUES
    # ------------------------------------------------------

    shap_values = (
        get_shap_values(
            transformed_data
        )
    )


    if (
        len(
            shap_values
        )
        !=
        len(
            transformed_features
        )
    ):

        raise ValueError(
            "SHAP feature count does not match "
            "transformed feature count. "
            f"SHAP={len(shap_values)}, "
            f"Features={len(transformed_features)}"
        )


    # ======================================================
    # AGGREGATE TRANSFORMED FEATURES
    # BACK TO RAW FEATURES
    # ======================================================

    aggregated = {}


    for (
        transformed_feature,
        shap_value,
    ) in zip(
        transformed_features,
        shap_values,
    ):

        raw_feature = (
            clean_feature_name(
                transformed_feature,
                expected_features,
            )
        )


        aggregated[
            raw_feature
        ] = (
            aggregated.get(
                raw_feature,
                0.0,
            )
            +
            float(
                shap_value
            )
        )


    # ======================================================
    # BUILD EXPLANATION
    # ======================================================

    factors = []


    for (
        feature,
        shap_value,
    ) in aggregated.items():

        if (
            feature
            in model_input.columns
        ):

            feature_value = (
                model_input.iloc[
                    0
                ][
                    feature
                ]
            )

        else:

            feature_value = (
                None
            )


        # Convert NumPy values to normal Python values

        if isinstance(
            feature_value,
            np.generic,
        ):

            feature_value = (
                feature_value.item()
            )


        # Positive SHAP:
        # pushes model toward fraud-risk class

        if (
            shap_value
            >
            0
        ):

            effect = (
                "increases_fraud_risk"
            )


        elif (
            shap_value
            <
            0
        ):

            effect = (
                "reduces_fraud_risk"
            )


        else:

            effect = (
                "neutral"
            )


        factors.append(
            {

                "feature":
                    feature,

                "value":
                    feature_value,

                "shap_value":
                    round(
                        float(
                            shap_value
                        ),
                        6,
                    ),

                "absolute_impact":
                    round(
                        abs(
                            float(
                                shap_value
                            )
                        ),
                        6,
                    ),

                "effect":
                    effect,
            }
        )


    # ------------------------------------------------------
    # SORT BY ABSOLUTE IMPACT
    # ------------------------------------------------------

    factors = sorted(
        factors,
        key=lambda item:
            item[
                "absolute_impact"
            ],
        reverse=True,
    )


    # ------------------------------------------------------
    # TOP N
    # ------------------------------------------------------

    top_factors = (
        factors[
            :top_n
        ]
    )


    # ======================================================
    # FINAL RESPONSE
    # ======================================================

    return {

        "fraud_probability":
            round(
                fraud_probability,
                6,
            ),

        "non_fraud_probability":
            round(
                non_fraud_probability,
                6,
            ),

        "base_value":
            round(
                get_base_value(),
                6,
            ),

        "top_factors":
            top_factors,

        "explanation_method":
            "SHAP TreeExplainer",

        "note":
            (
                "SHAP values represent model contribution "
                "toward or away from the fraud-risk class. "
                "They are not direct probability "
                "percentage-point changes."
            ),
    }


# ==========================================================
# SIMPLE DICTIONARY INTERFACE
# ==========================================================

def explain_fraud(
    input_data: dict,
    top_n: int = 8,
):

    model_input = (
        pd.DataFrame(
            [
                input_data
            ]
        )
    )


    return (
        explain_fraud_dataframe(
            model_input=
                model_input,

            top_n=
                top_n,
        )
    )


# ==========================================================
# DIRECT TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "MODEL PATH:"
    )

    print(
        MODEL_PATH
    )


    print(
        "\nEXPECTED RAW FEATURES:"
    )


    for feature in (
        get_expected_features()
    ):

        print(
            "-",
            feature,
        )


    print(
        "\nTRANSFORMED FEATURE COUNT:"
    )


    print(
        len(
            get_transformed_feature_names()
        )
    )