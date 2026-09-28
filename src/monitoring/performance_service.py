from collections import Counter

from sqlalchemy import text

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.monitoring.prediction_logger import (
    monitoring_engine,
)


# ==========================================================
# MODEL LABELS
# ==========================================================

def get_model_labels(
    model_name: str,
):

    model_name = (
        model_name
        .strip()
        .lower()
    )


    if model_name == "fraud":

        return [
            "Fraud",
            "Not Fraud",
        ]


    if model_name == "renewal":

        return [
            "Not Renewed",
            "Renewed",
        ]


    if model_name == "underwriting":

        return [
            "Approved",
            "Approved with Loading",
            "Declined",
        ]


    raise ValueError(
        f"Unsupported model: {model_name}"
    )


# ==========================================================
# NORMALIZE ACTUAL LABEL
# ==========================================================

def normalize_actual_label(
    model_name: str,
    actual_class,
):

    if actual_class is None:

        return None


    value = (
        str(
            actual_class
        )
        .strip()
        .lower()
    )


    # ======================================================
    # FRAUD
    # ======================================================

    if model_name == "fraud":

        if value in {
            "fraud",
            "fraud confirmed",
            "confirmed",
            "1",
            "yes",
        }:

            return "Fraud"


        if value in {
            "not fraud",
            "fraud not confirmed",
            "no fraud",
            "normal",
            "normal claim",
            "no",
            "0",
        }:

            return "Not Fraud"


        return None


    # ======================================================
    # RENEWAL
    # ======================================================

    if model_name == "renewal":

        if value in {
            "renewed",
            "renew",
            "1",
        }:

            return "Renewed"


        if value in {
            "not renewed",
            "non-renewed",
            "non renewed",
            "lapsed",
            "cancelled",
            "canceled",
            "0",
        }:

            return "Not Renewed"


        return None


    # ======================================================
    # UNDERWRITING
    # ======================================================

    if model_name == "underwriting":

        if value == "approved":

            return "Approved"


        if value in {
            "approved with loading",
            "loading",
        }:

            return (
                "Approved with Loading"
            )


        if value in {
            "declined",
            "rejected",
        }:

            return "Declined"


        return None


    return None


# ==========================================================
# DERIVE PREDICTED LABEL
# ==========================================================

def derive_predicted_label(
    model_name: str,
    predicted_class,
    probability,
    threshold,
):

    # ======================================================
    # FRAUD
    # ======================================================

    if model_name == "fraud":

        if (
            probability is not None
            and
            threshold is not None
        ):

            if (
                float(
                    probability
                )
                >=
                float(
                    threshold
                )
            ):

                return "Fraud"


            return "Not Fraud"


        value = (
            str(
                predicted_class
            )
            .strip()
            .lower()
        )


        if (
            "elevated"
            in value
            or
            "fraud risk"
            in value
        ):

            return "Fraud"


        if (
            "normal"
            in value
            or
            "no elevated"
            in value
        ):

            return "Not Fraud"


        return None


    # ======================================================
    # RENEWAL
    # ======================================================

    if model_name == "renewal":

        # Monitoring stores churn / non-renewal probability.

        if (
            probability is not None
            and
            threshold is not None
        ):

            if (
                float(
                    probability
                )
                >=
                float(
                    threshold
                )
            ):

                return (
                    "Not Renewed"
                )


            return "Renewed"


        value = (
            str(
                predicted_class
            )
            .strip()
            .lower()
        )


        if (
            "non-renewal"
            in value
            or
            "non renewal"
            in value
            or
            "churn"
            in value
            or
            "high risk"
            in value
            or
            "elevated"
            in value
        ):

            return (
                "Not Renewed"
            )


        if (
            "renew"
            in value
            or
            "lower"
            in value
        ):

            return "Renewed"


        return None


    # ======================================================
    # UNDERWRITING
    # ======================================================

    if model_name == "underwriting":

        if predicted_class is None:

            return None


        value = (
            str(
                predicted_class
            )
            .strip()
            .lower()
        )


        if value == "approved":

            return "Approved"


        if (
            "loading"
            in value
        ):

            return (
                "Approved with Loading"
            )


        if value in {
            "declined",
            "rejected",
        }:

            return "Declined"


        return None


    return None


# ==========================================================
# GET LABELED PREDICTIONS
# ==========================================================

def get_labeled_predictions(
    model_name: str,
):

    model_name = (
        model_name
        .strip()
        .lower()
    )


    # ======================================================
    # FRAUD
    # ======================================================

    if model_name == "fraud":

        query = text(
            """
            WITH ranked_predictions AS
            (
                SELECT

                    p.prediction_id,
                    p.model_name,
                    p.endpoint,
                    p.predicted_class,
                    p.probability,
                    p.threshold,
                    p.feature_snapshot,
                    p.created_at,

                    ROW_NUMBER() OVER
                    (
                        PARTITION BY
                            p.feature_snapshot
                            ->> 'CLAIM_ID'

                        ORDER BY
                            p.prediction_id DESC
                    ) AS row_number

                FROM
                    monitoring.prediction_logs p

                WHERE
                    p.model_name = 'fraud'

                    AND
                    p.status = 'success'

                    AND
                    p.endpoint =
                    '/predict/fraud/by-claim'

                    AND
                    p.feature_snapshot
                    ->> 'CLAIM_ID'
                    IS NOT NULL
            )

            SELECT

                r.prediction_id,
                r.predicted_class,
                r.probability,
                r.threshold,
                r.feature_snapshot,
                o.actual_class

            FROM ranked_predictions r

            INNER JOIN
                monitoring.prediction_outcomes o

                ON r.prediction_id
                =
                o.prediction_id

            WHERE
                r.row_number = 1

            ORDER BY
                r.prediction_id
            """
        )


    # ======================================================
    # RENEWAL
    # ======================================================

    elif model_name == "renewal":

        query = text(
            """
            WITH ranked_predictions AS
            (
                SELECT

                    p.prediction_id,
                    p.model_name,
                    p.endpoint,
                    p.predicted_class,
                    p.probability,
                    p.threshold,
                    p.feature_snapshot,
                    p.created_at,

                    ROW_NUMBER() OVER
                    (
                        PARTITION BY
                            p.feature_snapshot
                            ->> 'POLICY_ID'

                        ORDER BY
                            p.prediction_id DESC
                    ) AS row_number

                FROM
                    monitoring.prediction_logs p

                WHERE
                    p.model_name = 'renewal'

                    AND
                    p.status = 'success'

                    AND
                    p.endpoint =
                    '/predict/renewal/by-policy'

                    AND
                    p.feature_snapshot
                    ->> 'POLICY_ID'
                    IS NOT NULL
            )

            SELECT

                r.prediction_id,
                r.predicted_class,
                r.probability,
                r.threshold,
                r.feature_snapshot,
                o.actual_class

            FROM ranked_predictions r

            INNER JOIN
                monitoring.prediction_outcomes o

                ON r.prediction_id
                =
                o.prediction_id

            WHERE
                r.row_number = 1

            ORDER BY
                r.prediction_id
            """
        )


    # ======================================================
    # UNDERWRITING
    # ======================================================

    elif model_name == "underwriting":

        query = text(
            """
            WITH ranked_predictions AS
            (
                SELECT

                    p.prediction_id,
                    p.model_name,
                    p.endpoint,
                    p.predicted_class,
                    p.probability,
                    p.threshold,
                    p.feature_snapshot,
                    p.created_at,

                    ROW_NUMBER() OVER
                    (
                        PARTITION BY
                            p.feature_snapshot
                            ->> 'UNDERWRITING_ID'

                        ORDER BY
                            p.prediction_id DESC
                    ) AS row_number

                FROM
                    monitoring.prediction_logs p

                WHERE
                    p.model_name = 'underwriting'

                    AND
                    p.status = 'success'

                    AND
                    p.endpoint =
                    '/predict/underwriting/by-id'

                    AND
                    p.feature_snapshot
                    ->> 'UNDERWRITING_ID'
                    IS NOT NULL
            )

            SELECT

                r.prediction_id,
                r.predicted_class,
                r.probability,
                r.threshold,
                r.feature_snapshot,
                o.actual_class

            FROM ranked_predictions r

            INNER JOIN
                monitoring.prediction_outcomes o

                ON r.prediction_id
                =
                o.prediction_id

            WHERE
                r.row_number = 1

            ORDER BY
                r.prediction_id
            """
        )


    # ======================================================
    # OTHER MODELS
    # ======================================================

    else:

        query = text(
            """
            SELECT

                p.prediction_id,
                p.predicted_class,
                p.probability,
                p.threshold,
                p.feature_snapshot,
                o.actual_class

            FROM
                monitoring.prediction_logs p

            INNER JOIN
                monitoring.prediction_outcomes o

                ON p.prediction_id
                =
                o.prediction_id

            WHERE
                p.model_name = :model_name

                AND
                p.status = 'success'

            ORDER BY
                p.prediction_id
            """
        )


    # ======================================================
    # EXECUTE
    # ======================================================

    with monitoring_engine.connect() as connection:

        if model_name in {
            "fraud",
            "renewal",
            "underwriting",
        }:

            rows = (
                connection.execute(
                    query
                )
                .mappings()
                .all()
            )

        else:

            rows = (
                connection.execute(
                    query,
                    {
                        "model_name":
                            model_name
                    },
                )
                .mappings()
                .all()
            )


    return [
        dict(
            row
        )
        for row in rows
    ]


# ==========================================================
# PREPARE PERFORMANCE DATA
# ==========================================================

def prepare_performance_data(
    model_name: str,
):

    rows = (
        get_labeled_predictions(
            model_name
        )
    )


    actual_labels = []

    predicted_labels = []

    skipped_details = []


    for row in rows:

        actual_label = (
            normalize_actual_label(

                model_name=
                    model_name,

                actual_class=
                    row.get(
                        "actual_class"
                    ),
            )
        )


        predicted_label = (
            derive_predicted_label(

                model_name=
                    model_name,

                predicted_class=
                    row.get(
                        "predicted_class"
                    ),

                probability=
                    row.get(
                        "probability"
                    ),

                threshold=
                    row.get(
                        "threshold"
                    ),
            )
        )


        if (
            actual_label is None
            or
            predicted_label is None
        ):

            skipped_details.append(
                {
                    "prediction_id":
                        row.get(
                            "prediction_id"
                        ),

                    "actual_class":
                        row.get(
                            "actual_class"
                        ),

                    "predicted_class":
                        row.get(
                            "predicted_class"
                        ),

                    "reason":
                        (
                            "Could not normalize "
                            "actual/predicted label"
                        ),
                }
            )


            continue


        actual_labels.append(
            actual_label
        )


        predicted_labels.append(
            predicted_label
        )


    return {

        "rows":
            rows,

        "actual_labels":
            actual_labels,

        "predicted_labels":
            predicted_labels,

        "skipped_details":
            skipped_details,
    }


# ==========================================================
# ADD EVALUATION METADATA
# ==========================================================

def add_evaluation_metadata(
    result: dict,
    model_name: str,
):

    if model_name == "fraud":

        result[
            "evaluation_source"
        ] = (
            "database_claim_predictions"
        )

        result[
            "endpoint_filter"
        ] = (
            "/predict/fraud/by-claim"
        )

        result[
            "deduplication"
        ] = (
            "latest_prediction_per_claim_id"
        )


    elif model_name == "renewal":

        result[
            "evaluation_source"
        ] = (
            "database_policy_predictions"
        )

        result[
            "endpoint_filter"
        ] = (
            "/predict/renewal/by-policy"
        )

        result[
            "deduplication"
        ] = (
            "latest_prediction_per_policy_id"
        )


    elif model_name == "underwriting":

        result[
            "evaluation_source"
        ] = (
            "database_underwriting_predictions"
        )

        result[
            "endpoint_filter"
        ] = (
            "/predict/underwriting/by-id"
        )

        result[
            "deduplication"
        ] = (
            "latest_prediction_per_underwriting_id"
        )


    return result


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

def get_model_performance(
    model_name: str,
):

    model_name = (
        model_name
        .strip()
        .lower()
    )


    labels = (
        get_model_labels(
            model_name
        )
    )


    prepared = (
        prepare_performance_data(
            model_name
        )
    )


    actual_labels = (
        prepared[
            "actual_labels"
        ]
    )


    predicted_labels = (
        prepared[
            "predicted_labels"
        ]
    )


    skipped_details = (
        prepared[
            "skipped_details"
        ]
    )


    # ======================================================
    # NO LABELED DATA
    # ======================================================

    if not actual_labels:

        result = {

            "model_name":
                model_name,

            "status":
                "insufficient_data",

            "evaluated_predictions":
                0,

            "skipped_predictions":
                len(
                    skipped_details
                ),

            "skipped_details":
                skipped_details,

            "message":
                (
                    "No usable labeled predictions "
                    "are available for evaluation."
                ),
        }


        return (
            add_evaluation_metadata(
                result,
                model_name,
            )
        )


    # ======================================================
    # OVERALL METRICS
    # ======================================================

    accuracy = (
        accuracy_score(
            actual_labels,
            predicted_labels,
        )
    )


    precision = (
        precision_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


    recall = (
        recall_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


    f1 = (
        f1_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


    # ======================================================
    # CLASS-WISE METRICS
    # ======================================================

    class_precision = (
        precision_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )


    class_recall = (
        recall_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )


    class_f1 = (
        f1_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )


    actual_counter = (
        Counter(
            actual_labels
        )
    )


    class_metrics = []


    for index, label in enumerate(
        labels
    ):

        class_metrics.append(
            {
                "class":
                    label,

                "precision":
                    round(
                        float(
                            class_precision[
                                index
                            ]
                        ),
                        4,
                    ),

                "recall":
                    round(
                        float(
                            class_recall[
                                index
                            ]
                        ),
                        4,
                    ),

                "f1_score":
                    round(
                        float(
                            class_f1[
                                index
                            ]
                        ),
                        4,
                    ),

                "support":
                    int(
                        actual_counter.get(
                            label,
                            0,
                        )
                    ),
            }
        )


    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    matrix = (
        confusion_matrix(
            actual_labels,
            predicted_labels,
            labels=labels,
        )
    )


    # ======================================================
    # RESULT
    # ======================================================

    result = {

        "model_name":
            model_name,

        "status":
            "success",

        "evaluated_predictions":
            len(
                actual_labels
            ),

        "skipped_predictions":
            len(
                skipped_details
            ),

        "skipped_details":
            skipped_details,

        "accuracy":
            round(
                float(
                    accuracy
                ),
                4,
            ),

        "precision":
            round(
                float(
                    precision
                ),
                4,
            ),

        "recall":
            round(
                float(
                    recall
                ),
                4,
            ),

        "f1_score":
            round(
                float(
                    f1
                ),
                4,
            ),

        "class_metrics":
            class_metrics,

        "confusion_matrix": {

            "labels":
                labels,

            "matrix":
                matrix.tolist(),
        },

        "actual_distribution":
            dict(
                Counter(
                    actual_labels
                )
            ),

        "predicted_distribution":
            dict(
                Counter(
                    predicted_labels
                )
            ),
    }


    return (
        add_evaluation_metadata(
            result,
            model_name,
        )
    )


# ==========================================================
# ALL MODEL PERFORMANCE
# ==========================================================

def get_all_model_performance():

    models = [
        "renewal",
        "fraud",
        "underwriting",
    ]


    results = {}


    for model_name in models:

        try:

            results[
                model_name
            ] = (
                get_model_performance(
                    model_name
                )
            )


        except Exception as error:

            results[
                model_name
            ] = {

                "model_name":
                    model_name,

                "status":
                    "error",

                "error":
                    str(
                        error
                    ),
            }


    return results


# ==========================================================
# MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    from pprint import pprint


    print(
        "\n=============================="
    )

    print(
        "FRAUD PERFORMANCE"
    )

    print(
        "=============================="
    )


    pprint(
        get_model_performance(
            "fraud"
        )
    )


    print(
        "\n=============================="
    )

    print(
        "RENEWAL PERFORMANCE"
    )

    print(
        "=============================="
    )


    pprint(
        get_model_performance(
            "renewal"
        )
    )


    print(
        "\n=============================="
    )

    print(
        "UNDERWRITING PERFORMANCE"
    )

    print(
        "=============================="
    )


    pprint(
        get_model_performance(
            "underwriting"
        )
    )