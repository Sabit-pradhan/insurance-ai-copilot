# src/explainability/renewal_explainer.py

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
    / "renewal_xgboost_model.joblib"
)


# ==========================================================
# LOAD RENEWAL PIPELINE
# ==========================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Renewal model not found: {MODEL_PATH}"
    )


renewal_pipeline = joblib.load(
    MODEL_PATH
)


# ==========================================================
# GET PIPELINE COMPONENTS
# ==========================================================

if not hasattr(
    renewal_pipeline,
    "named_steps",
):

    raise TypeError(
        "Renewal model is expected to be "
        "a scikit-learn Pipeline."
    )


if "preprocessor" not in renewal_pipeline.named_steps:

    raise KeyError(
        "Pipeline does not contain "
        "'preprocessor' step."
    )


if "model" not in renewal_pipeline.named_steps:

    raise KeyError(
        "Pipeline does not contain "
        "'model' step."
    )


preprocessor = (
    renewal_pipeline
    .named_steps[
        "preprocessor"
    ]
)

xgb_model = (
    renewal_pipeline
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
# EXPECTED RAW FEATURES
# ==========================================================

def get_expected_features():
    """
    Return raw input columns expected
    by the trained renewal pipeline.
    """

    if hasattr(
        renewal_pipeline,
        "feature_names_in_",
    ):

        return list(
            renewal_pipeline
            .feature_names_in_
        )

    return []


# ==========================================================
# TRANSFORMED FEATURE NAMES
# ==========================================================

def get_transformed_feature_names():
    """
    Return feature names after preprocessing.

    Example:

    AGE
        ->
    num__AGE

    STATE
        ->
    cat__STATE_Karnataka
    """

    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):

        return list(
            preprocessor
            .get_feature_names_out()
        )

    return []


# ==========================================================
# CLEAN FEATURE NAME
# ==========================================================

def clean_feature_name(
    transformed_name,
    raw_columns,
):
    """
    Convert transformed OneHot feature names
    back to the original business feature.

    Example:

    num__AGE
        ->
    AGE

    cat__STATE_Karnataka
        ->
    STATE
    """

    name = str(
        transformed_name
    )

    # Remove ColumnTransformer prefix
    if "__" in name:

        name = name.split(
            "__",
            1,
        )[1]

    # Exact raw feature
    if name in raw_columns:

        return name

    # One-hot encoded feature
    for column in sorted(
        raw_columns,
        key=len,
        reverse=True,
    ):

        prefix = (
            f"{column}_"
        )

        if name.startswith(
            prefix
        ):

            return column

    return name


# ==========================================================
# GET SHAP VALUES
# ==========================================================

def get_shap_values(
    transformed_data,
):
    """
    Calculate SHAP values for
    the positive / non-renewal class.
    """

    # One row only, so converting to dense
    # is safe and keeps SHAP simple.
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

    # Older SHAP versions can return:
    #
    # [
    #   class_0_values,
    #   class_1_values
    # ]

    if isinstance(
        shap_values,
        list,
    ):

        if len(
            shap_values
        ) > 1:

            values = (
                shap_values[1][0]
            )

        else:

            values = (
                shap_values[0][0]
            )

        return np.asarray(
            values,
            dtype=float,
        )

    values = np.asarray(
        shap_values
    )

    # ------------------------------------------------------
    # Binary SHAP:
    # (rows, features)
    # ------------------------------------------------------

    if values.ndim == 2:

        return (
            values[0]
            .astype(float)
        )

    # ------------------------------------------------------
    # Some versions:
    # (rows, features, classes)
    # ------------------------------------------------------

    if values.ndim == 3:

        class_index = (
            1
            if values.shape[2] > 1
            else 0
        )

        return (
            values[
                0,
                :,
                class_index,
            ]
            .astype(float)
        )

    if values.ndim == 1:

        return values.astype(
            float
        )

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{values.shape}"
    )


# ==========================================================
# BASE VALUE
# ==========================================================

def get_base_value():
    """
    Return SHAP model baseline.
    """

    value = (
        shap_explainer
        .expected_value
    )

    if isinstance(
        value,
        list,
    ):

        if len(value) > 1:

            return float(
                value[1]
            )

        return float(
            value[0]
        )

    value = np.asarray(
        value
    )

    if value.ndim == 0:

        return float(
            value
        )

    if len(value) > 1:

        return float(
            value[1]
        )

    return float(
        value[0]
    )


# ==========================================================
# EXPLAIN DATAFRAME
# ==========================================================

def explain_renewal_dataframe(
    model_input: pd.DataFrame,
    top_n: int = 8,
):
    """
    Explain one renewal prediction.

    model_input must contain the same raw
    features used by the trained pipeline.
    """

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
            "Explainability currently supports "
            "one customer at a time."
        )


    # ======================================================
    # EXPECTED FEATURES
    # ======================================================

    expected_features = (
        get_expected_features()
    )


    if expected_features:

        missing_features = [

            column

            for column
            in expected_features

            if column
            not in model_input.columns
        ]


        if missing_features:

            raise ValueError(
                "Missing model features: "
                f"{missing_features}"
            )


        # Keep exact training order
        model_input = (
            model_input[
                expected_features
            ]
            .copy()
        )


    raw_columns = list(
        model_input.columns
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    probabilities = (
        renewal_pipeline
        .predict_proba(
            model_input
        )[0]
    )


    classes = getattr(
        renewal_pipeline,
        "classes_",
        None,
    )


    positive_class_index = (
        len(probabilities) - 1
    )


    if classes is not None:

        classes_list = list(
            classes
        )

        if 1 in classes_list:

            positive_class_index = (
                classes_list.index(
                    1
                )
            )


    churn_probability = float(
        probabilities[
            positive_class_index
        ]
    )


    renewal_probability = float(
        1
        -
        churn_probability
    )


    # ======================================================
    # PREPROCESS INPUT
    # ======================================================

    transformed_data = (
        preprocessor
        .transform(
            model_input
        )
    )


    transformed_feature_names = (
        get_transformed_feature_names()
    )


    if not transformed_feature_names:

        number_of_features = (
            transformed_data.shape[1]
        )

        transformed_feature_names = [

            f"feature_{i}"

            for i in range(
                number_of_features
            )
        ]


    # ======================================================
    # SHAP VALUES
    # ======================================================

    shap_values = (
        get_shap_values(
            transformed_data
        )
    )


    if (
        len(shap_values)
        !=
        len(
            transformed_feature_names
        )
    ):

        raise ValueError(
            "SHAP feature count does not match "
            "preprocessor feature count."
        )


    # ======================================================
    # AGGREGATE ONE-HOT FEATURES
    # ======================================================

    aggregated = {}


    for (
        transformed_feature,
        shap_value,
    ) in zip(
        transformed_feature_names,
        shap_values,
    ):

        raw_feature = (
            clean_feature_name(
                transformed_feature,
                raw_columns,
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
    # BUILD BUSINESS EXPLANATION
    # ======================================================

    factors = []


    for (
        feature,
        shap_value,
    ) in aggregated.items():


        feature_value = None


        if feature in model_input.columns:

            feature_value = (
                model_input
                .iloc[0][
                    feature
                ]
            )


            if pd.isna(
                feature_value
            ):

                feature_value = None


            elif isinstance(
                feature_value,
                np.generic,
            ):

                feature_value = (
                    feature_value
                    .item()
                )


        # Positive SHAP contribution means
        # movement toward positive model class.
        #
        # Renewal model positive class is treated
        # as non-renewal / churn risk.

        if shap_value > 0:

            effect = (
                "increases_churn_risk"
            )

        elif shap_value < 0:

            effect = (
                "reduces_churn_risk"
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


    # ======================================================
    # SORT MOST IMPORTANT FIRST
    # ======================================================

    factors = sorted(
        factors,
        key=lambda item:
            item[
                "absolute_impact"
            ],
        reverse=True,
    )


    top_factors = (
        factors[
            :top_n
        ]
    )


    # ======================================================
    # FINAL RESPONSE
    # ======================================================

    return {

        "renewal_probability":
            round(
                renewal_probability,
                6,
            ),

        "churn_probability":
            round(
                churn_probability,
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

        "note": (
            "SHAP values represent model contribution, "
            "not direct percentage-point changes."
        ),
    }


# ==========================================================
# EXPLAIN DICTIONARY
# ==========================================================

def explain_renewal(
    input_data: dict,
    top_n: int = 8,
):
    """
    Convenience wrapper for dictionary input.
    """

    if not isinstance(
        input_data,
        dict,
    ):

        raise TypeError(
            "input_data must be a dictionary."
        )


    model_input = pd.DataFrame(
        [
            input_data
        ]
    )


    return explain_renewal_dataframe(
        model_input=
            model_input,

        top_n=
            top_n,
    )


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "\nRenewal SHAP Explainer Loaded"
    )

    print(
        "\nMODEL:"
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