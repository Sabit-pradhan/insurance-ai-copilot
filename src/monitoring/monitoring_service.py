# src/monitoring/monitoring_service.py

from sqlalchemy import text

from src.monitoring.prediction_logger import (
    monitoring_engine,
)


# ==========================================================
# ALLOWED PRODUCTION MODELS
# ==========================================================

PRODUCTION_MODELS = (
    "renewal",
    "fraud",
    "underwriting",
)


# ==========================================================
# SAFE NUMBER HELPERS
# ==========================================================

def safe_int(
    value,
):

    if value is None:
        return 0

    return int(
        value
    )


def safe_float(
    value,
    digits=3,
):

    if value is None:
        return 0.0

    return round(
        float(
            value
        ),
        digits,
    )


# ==========================================================
# OVERALL MONITORING SUMMARY
# ==========================================================

def get_monitoring_summary():

    """
    Return overall operational monitoring
    statistics for production ML models.
    """

    query = text(
        """
        SELECT

            COUNT(*) AS total_predictions,

            COUNT(*) FILTER (
                WHERE status = 'success'
            ) AS successful_predictions,

            COUNT(*) FILTER (
                WHERE status = 'error'
            ) AS failed_predictions,

            AVG(
                latency_ms
            ) FILTER (
                WHERE latency_ms IS NOT NULL
            ) AS average_latency_ms,

            PERCENTILE_CONT(
                0.95
            )
            WITHIN GROUP (
                ORDER BY latency_ms
            )
            FILTER (
                WHERE latency_ms IS NOT NULL
            ) AS p95_latency_ms,

            AVG(
                probability
            ) FILTER (
                WHERE probability IS NOT NULL
                AND status = 'success'
            ) AS average_probability

        FROM monitoring.prediction_logs

        WHERE model_name IN (
            'renewal',
            'fraud',
            'underwriting'
        );
        """
    )


    with monitoring_engine.connect() as connection:

        row = (
            connection.execute(
                query
            )
            .mappings()
            .first()
        )


    if row is None:

        return {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "average_probability": 0.0,
        }


    total_predictions = (
        safe_int(
            row[
                "total_predictions"
            ]
        )
    )


    successful_predictions = (
        safe_int(
            row[
                "successful_predictions"
            ]
        )
    )


    failed_predictions = (
        safe_int(
            row[
                "failed_predictions"
            ]
        )
    )


    if total_predictions > 0:

        success_rate = (
            successful_predictions
            /
            total_predictions
            *
            100
        )


        failure_rate = (
            failed_predictions
            /
            total_predictions
            *
            100
        )

    else:

        success_rate = 0.0

        failure_rate = 0.0


    return {

        "total_predictions":
            total_predictions,

        "successful_predictions":
            successful_predictions,

        "failed_predictions":
            failed_predictions,

        "success_rate":
            round(
                success_rate,
                2,
            ),

        "failure_rate":
            round(
                failure_rate,
                2,
            ),

        "average_latency_ms":
            safe_float(
                row[
                    "average_latency_ms"
                ]
            ),

        "p95_latency_ms":
            safe_float(
                row[
                    "p95_latency_ms"
                ]
            ),

        "average_probability":
            safe_float(
                row[
                    "average_probability"
                ],
                digits=6,
            ),
    }


# ==========================================================
# MODEL-WISE MONITORING SUMMARY
# ==========================================================

def get_model_summary():

    """
    Return monitoring metrics grouped
    by model and model version.
    """

    query = text(
        """
        SELECT

            model_name,

            model_version,

            COUNT(*) AS total_predictions,

            COUNT(*) FILTER (
                WHERE status = 'success'
            ) AS successful_predictions,

            COUNT(*) FILTER (
                WHERE status = 'error'
            ) AS failed_predictions,

            AVG(
                latency_ms
            ) FILTER (
                WHERE latency_ms IS NOT NULL
            ) AS average_latency_ms,

            PERCENTILE_CONT(
                0.95
            )
            WITHIN GROUP (
                ORDER BY latency_ms
            )
            FILTER (
                WHERE latency_ms IS NOT NULL
            ) AS p95_latency_ms,

            AVG(
                probability
            ) FILTER (
                WHERE probability IS NOT NULL
                AND status = 'success'
            ) AS average_probability,

            MIN(
                created_at
            ) AS first_prediction_at,

            MAX(
                created_at
            ) AS latest_prediction_at

        FROM monitoring.prediction_logs

        WHERE model_name IN (
            'renewal',
            'fraud',
            'underwriting'
        )

        GROUP BY
            model_name,
            model_version

        ORDER BY
            model_name,
            model_version;
        """
    )


    with monitoring_engine.connect() as connection:

        rows = (
            connection.execute(
                query
            )
            .mappings()
            .all()
        )


    results = []


    for row in rows:

        total_predictions = (
            safe_int(
                row[
                    "total_predictions"
                ]
            )
        )


        successful_predictions = (
            safe_int(
                row[
                    "successful_predictions"
                ]
            )
        )


        failed_predictions = (
            safe_int(
                row[
                    "failed_predictions"
                ]
            )
        )


        if total_predictions > 0:

            success_rate = (
                successful_predictions
                /
                total_predictions
                *
                100
            )

        else:

            success_rate = 0.0


        results.append(
            {

                "model_name":
                    row[
                        "model_name"
                    ],

                "model_version":
                    row[
                        "model_version"
                    ],

                "total_predictions":
                    total_predictions,

                "successful_predictions":
                    successful_predictions,

                "failed_predictions":
                    failed_predictions,

                "success_rate":
                    round(
                        success_rate,
                        2,
                    ),

                "average_latency_ms":
                    safe_float(
                        row[
                            "average_latency_ms"
                        ]
                    ),

                "p95_latency_ms":
                    safe_float(
                        row[
                            "p95_latency_ms"
                        ]
                    ),

                "average_probability":
                    safe_float(
                        row[
                            "average_probability"
                        ],
                        digits=6,
                    ),

                "first_prediction_at":
                    row[
                        "first_prediction_at"
                    ],

                "latest_prediction_at":
                    row[
                        "latest_prediction_at"
                    ],
            }
        )


    return results


# ==========================================================
# RECENT PREDICTIONS
# ==========================================================

def get_recent_predictions(
    limit=20,
):

    """
    Return the latest production
    model predictions.
    """

    try:

        limit = int(
            limit
        )

    except Exception:

        limit = 20


    if limit < 1:

        limit = 1


    if limit > 100:

        limit = 100


    query = text(
        """
        SELECT

            prediction_id,

            model_name,

            model_version,

            endpoint,

            predicted_class,

            probability,

            threshold,

            latency_ms,

            status,

            error_message,

            feature_snapshot,

            explainability,

            created_at

        FROM monitoring.prediction_logs

        WHERE model_name IN (
            'renewal',
            'fraud',
            'underwriting'
        )

        ORDER BY
            created_at DESC,
            prediction_id DESC

        LIMIT :limit;
        """
    )


    with monitoring_engine.connect() as connection:

        rows = (
            connection.execute(
                query,
                {
                    "limit":
                        limit,
                },
            )
            .mappings()
            .all()
        )


    results = []


    for row in rows:

        results.append(
            {

                "prediction_id":
                    safe_int(
                        row[
                            "prediction_id"
                        ]
                    ),

                "model_name":
                    row[
                        "model_name"
                    ],

                "model_version":
                    row[
                        "model_version"
                    ],

                "endpoint":
                    row[
                        "endpoint"
                    ],

                "predicted_class":
                    row[
                        "predicted_class"
                    ],

                "probability": (
                    None

                    if row[
                        "probability"
                    ] is None

                    else

                    safe_float(
                        row[
                            "probability"
                        ],
                        digits=6,
                    )
                ),

                "threshold": (
                    None

                    if row[
                        "threshold"
                    ] is None

                    else

                    safe_float(
                        row[
                            "threshold"
                        ],
                        digits=6,
                    )
                ),

                "latency_ms": (
                    None

                    if row[
                        "latency_ms"
                    ] is None

                    else

                    safe_float(
                        row[
                            "latency_ms"
                        ]
                    )
                ),

                "status":
                    row[
                        "status"
                    ],

                "error_message":
                    row[
                        "error_message"
                    ],

                "feature_snapshot":
                    row[
                        "feature_snapshot"
                    ],

                "explainability":
                    row[
                        "explainability"
                    ],

                "created_at":
                    row[
                        "created_at"
                    ],
            }
        )


    return results


# ==========================================================
# MANUAL TEST
# ==========================================================

if __name__ == "__main__":

    import pprint


    print(
        "\nMONITORING SUMMARY\n"
    )

    pprint.pp(
        get_monitoring_summary()
    )


    print(
        "\nMODEL SUMMARY\n"
    )

    pprint.pp(
        get_model_summary()
    )


    print(
        "\nRECENT PREDICTIONS\n"
    )

    pprint.pp(
        get_recent_predictions(
            limit=5
        )
    )