# src/explainability/underwriting_explainer.py

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
    / "underwriting_xgboost_model.joblib"
)

LABEL_ENCODER_PATH = (
    PROJECT_ROOT
    / "models"
    / "underwriting_label_encoder.joblib"
)


# ==========================================================
# LOAD MODEL + LABEL ENCODER
# ==========================================================

underwriting_pipeline = (
    joblib.load(
        MODEL_PATH
    )
)

label_encoder = (
    joblib.load(
        LABEL_ENCODER_PATH
    )
)


# ==========================================================
# VALIDATE PIPELINE
# ==========================================================

if not hasattr(
    underwriting_pipeline,
    "named_steps",
):

    raise ValueError(
        "Underwriting model must be "
        "a scikit-learn Pipeline."
    )


if (
    "preprocessor"
    not in underwriting_pipeline.named_steps
):

    raise ValueError(
        "Underwriting pipeline does not contain "
        "'preprocessor' step."
    )


if (
    "model"
    not in underwriting_pipeline.named_steps
):

    raise ValueError(
        "Underwriting pipeline does not contain "
        "'model' step."
    )


# ==========================================================
# PIPELINE COMPONENTS
# ==========================================================

preprocessor = (
    underwriting_pipeline
    .named_steps[
        "preprocessor"
    ]
)

xgb_model = (
    underwriting_pipeline
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

    if not hasattr(
        underwriting_pipeline,
        "feature_names_in_",
    ):

        raise ValueError(
            "Underwriting pipeline does not expose "
            "feature_names_in_."
        )

    return list(
        underwriting_pipeline
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
# CLEAN FEATURE NAME
# ==========================================================

def clean_feature_name(
    transformed_name: str,
    raw_features: list[str],
):

    """
    Examples:

    num__AGE
    ->
    AGE

    cat__LIFESTYLE_Good
    ->
    LIFESTYLE
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


    # Longest first because raw names
    # themselves may contain underscores.

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
# GET PREDICTED CLASS INFORMATION
# ==========================================================

def get_prediction_details(
    model_input: pd.DataFrame,
):

    probabilities = (
        underwriting_pipeline
        .predict_proba(
            model_input
        )[0]
    )


    model_classes = list(
        underwriting_pipeline
        .classes_
    )


    predicted_encoded = (
        underwriting_pipeline
        .predict(
            model_input
        )[0]
    )


    predicted_index = (
        model_classes.index(
            predicted_encoded
        )
    )


    predicted_label = (
        label_encoder
        .inverse_transform(
            [
                int(
                    predicted_encoded
                )
            ]
        )[0]
    )


    probability_dict = {}


    for (
        encoded_class,
        probability,
    ) in zip(
        model_classes,
        probabilities,
    ):

        class_label = (
            label_encoder
            .inverse_transform(
                [
                    int(
                        encoded_class
                    )
                ]
            )[0]
        )


        probability_dict[
            str(
                class_label
            )
        ] = round(
            float(
                probability
            ),
            6,
        )


    return {

        "predicted_encoded":
            int(
                predicted_encoded
            ),

        "predicted_index":
            predicted_index,

        "predicted_label":
            str(
                predicted_label
            ),

        "predicted_probability":
            float(
                probabilities[
                    predicted_index
                ]
            ),

        "probabilities":
            probability_dict,

        "number_of_classes":
            len(
                model_classes
            ),
    }


# ==========================================================
# GET SHAP VALUES FOR PREDICTED CLASS
# ==========================================================

def get_shap_values_for_class(
    transformed_data,
    class_index: int,
    number_of_classes: int,
):

    """
    Return SHAP values for one row
    and one selected underwriting class.
    """

    if hasattr(
        transformed_data,
        "toarray",
    ):

        transformed_data = (
            transformed_data
            .toarray()
        )


    number_of_features = (
        transformed_data
        .shape[1]
    )


    shap_values = (
        shap_explainer
        .shap_values(
            transformed_data
        )
    )


    # ======================================================
    # FORMAT 1
    # list[class]
    # ======================================================

    if isinstance(
        shap_values,
        list,
    ):

        selected = (
            np.asarray(
                shap_values[
                    class_index
                ]
            )
        )


        if (
            selected.ndim
            ==
            2
        ):

            selected = (
                selected[0]
            )


        return np.asarray(
            selected,
            dtype=float,
        )


    shap_values = (
        np.asarray(
            shap_values
        )
    )


    # ======================================================
    # FORMAT 2
    # rows x features x classes
    # ======================================================

    if (
        shap_values.ndim
        ==
        3
    ):

        if (
            shap_values.shape[0]
            ==
            1
            and
            shap_values.shape[1]
            ==
            number_of_features
            and
            shap_values.shape[2]
            ==
            number_of_classes
        ):

            return np.asarray(
                shap_values[
                    0,
                    :,
                    class_index,
                ],
                dtype=float,
            )


        # --------------------------------------------------
        # rows x classes x features
        # --------------------------------------------------

        if (
            shap_values.shape[0]
            ==
            1
            and
            shap_values.shape[1]
            ==
            number_of_classes
            and
            shap_values.shape[2]
            ==
            number_of_features
        ):

            return np.asarray(
                shap_values[
                    0,
                    class_index,
                    :,
                ],
                dtype=float,
            )


        # --------------------------------------------------
        # classes x rows x features
        # --------------------------------------------------

        if (
            shap_values.shape[0]
            ==
            number_of_classes
            and
            shap_values.shape[1]
            ==
            1
            and
            shap_values.shape[2]
            ==
            number_of_features
        ):

            return np.asarray(
                shap_values[
                    class_index,
                    0,
                    :,
                ],
                dtype=float,
            )


    # ======================================================
    # FORMAT 3
    # 2D
    # ======================================================

    if (
        shap_values.ndim
        ==
        2
    ):

        # One row x features

        if (
            shap_values.shape
            ==
            (
                1,
                number_of_features,
            )
        ):

            return np.asarray(
                shap_values[
                    0
                ],
                dtype=float,
            )


        # Features x classes

        if (
            shap_values.shape
            ==
            (
                number_of_features,
                number_of_classes,
            )
        ):

            return np.asarray(
                shap_values[
                    :,
                    class_index,
                ],
                dtype=float,
            )


        # Classes x features

        if (
            shap_values.shape
            ==
            (
                number_of_classes,
                number_of_features,
            )
        ):

            return np.asarray(
                shap_values[
                    class_index,
                    :,
                ],
                dtype=float,
            )


    # ======================================================
    # FORMAT 4
    # 1D
    # ======================================================

    if (
        shap_values.ndim
        ==
        1
        and
        len(
            shap_values
        )
        ==
        number_of_features
    ):

        return np.asarray(
            shap_values,
            dtype=float,
        )


    raise ValueError(
        "Unexpected multiclass SHAP shape: "
        f"{shap_values.shape}"
    )


# ==========================================================
# GET BASE VALUE FOR CLASS
# ==========================================================

def get_base_value_for_class(
    class_index: int,
):

    expected_value = (
        shap_explainer
        .expected_value
    )


    expected_value = (
        np.asarray(
            expected_value
        )
    )


    # Scalar

    if (
        expected_value.ndim
        ==
        0
    ):

        return float(
            expected_value
        )


    flattened = (
        expected_value
        .reshape(-1)
    )


    if (
        class_index
        <
        len(
            flattened
        )
    ):

        return float(
            flattened[
                class_index
            ]
        )


    return float(
        flattened[
            0
        ]
    )


# ==========================================================
# EXPLAIN UNDERWRITING DATAFRAME
# ==========================================================

def explain_underwriting_dataframe(
    model_input: pd.DataFrame,
    top_n: int = 8,
):

    # ------------------------------------------------------
    # VALIDATE
    # ------------------------------------------------------

    if not isinstance(
        model_input,
        pd.DataFrame,
    ):

        raise TypeError(
            "model_input must be a pandas DataFrame."
        )


    if (
        len(
            model_input
        )
        !=
        1
    ):

        raise ValueError(
            "Underwriting explanation currently "
            "supports one applicant at a time."
        )


    # ------------------------------------------------------
    # EXPECTED FEATURES
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
            "Missing underwriting model features: "
            f"{missing_features}"
        )


    # Keep training order

    model_input = (
        model_input[
            expected_features
        ]
        .copy()
    )


    # ======================================================
    # PREDICTION
    # ======================================================

    prediction_details = (
        get_prediction_details(
            model_input
        )
    )


    predicted_index = (
        prediction_details[
            "predicted_index"
        ]
    )


    number_of_classes = (
        prediction_details[
            "number_of_classes"
        ]
    )


    # ======================================================
    # TRANSFORM INPUT
    # ======================================================

    transformed_data = (
        preprocessor
        .transform(
            model_input
        )
    )


    transformed_features = (
        get_transformed_feature_names()
    )


    # ======================================================
    # SHAP FOR PREDICTED CLASS
    # ======================================================

    shap_values = (
        get_shap_values_for_class(
            transformed_data=
                transformed_data,

            class_index=
                predicted_index,

            number_of_classes=
                number_of_classes,
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
    # AGGREGATE BACK TO RAW FEATURES
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
                transformed_name=
                    transformed_feature,

                raw_features=
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
    # CREATE FACTORS
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


        if isinstance(
            feature_value,
            np.generic,
        ):

            feature_value = (
                feature_value.item()
            )


        # --------------------------------------------------
        # MULTICLASS INTERPRETATION
        # --------------------------------------------------
        #
        # Positive:
        # pushes model toward predicted class
        #
        # Negative:
        # pushes model away from predicted class
        # --------------------------------------------------

        if (
            shap_value
            >
            0
        ):

            effect = (
                "supports_predicted_class"
            )


        elif (
            shap_value
            <
            0
        ):

            effect = (
                "opposes_predicted_class"
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
    # SORT
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

        "predicted_class":
            prediction_details[
                "predicted_label"
            ],

        "predicted_probability":
            round(
                prediction_details[
                    "predicted_probability"
                ],
                6,
            ),

        "probabilities":
            prediction_details[
                "probabilities"
            ],

        "base_value":
            round(
                get_base_value_for_class(
                    predicted_index
                ),
                6,
            ),

        "top_factors":
            top_factors,

        "explanation_method":
            "SHAP TreeExplainer",

        "explained_class":
            prediction_details[
                "predicted_label"
            ],

        "note":
            (
                "SHAP values explain the model score "
                "for the predicted underwriting class. "
                "Positive values support the predicted "
                "class and negative values push away "
                "from it. They are not direct probability "
                "percentage-point changes."
            ),
    }


# ==========================================================
# DICTIONARY INTERFACE
# ==========================================================

def explain_underwriting(
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
        explain_underwriting_dataframe(
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
        "\nLABEL ENCODER PATH:"
    )

    print(
        LABEL_ENCODER_PATH
    )


    print(
        "\nLABEL CLASSES:"
    )

    print(
        list(
            label_encoder
            .classes_
        )
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