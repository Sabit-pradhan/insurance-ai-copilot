# src/rag/evaluation.py

import json
import os

from datetime import datetime
from pathlib import Path

from src.rag.retriever import retrieve_context
from src.rag.rag_service import answer_rag_question
from src.core.logger import get_logger


# ==========================================================
# LOGGER
# ==========================================================

logger = get_logger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

RAG_EVAL_RUN_GENERATION = (
    os.getenv(
        "RAG_EVAL_RUN_GENERATION",
        "false",
    )
    .strip()
    .lower()
    == "true"
)

RAG_EVAL_REPORT_DIR = Path(
    os.getenv(
        "RAG_EVAL_REPORT_DIR",
        "artifacts/evaluation",
    )
)


# ==========================================================
# EVALUATION DATASET
# ==========================================================

EVALUATION_CASES = [

    # ------------------------------------------------------
    # CLAIM
    # ------------------------------------------------------

    {
        "id": "claim_documents",
        "question": (
            "What documents are required "
            "for an insurance claim?"
        ),
        "expected_sources": [
            "claim_guide.txt",
        ],
    },

    # ------------------------------------------------------
    # RENEWAL
    # ------------------------------------------------------

    {
        "id": "renewal_review",
        "question": (
            "What information may be reviewed "
            "before policy renewal?"
        ),
        "expected_sources": [
            "renewal_guide.txt",
        ],
    },

    # ------------------------------------------------------
    # UNDERWRITING
    # ------------------------------------------------------

    {
        "id": "underwriting_factors",
        "question": (
            "What factors may be considered "
            "during insurance underwriting?"
        ),
        "expected_sources": [
            "underwriting_guide.txt",
        ],
    },

    # ------------------------------------------------------
    # FRAUD / CLAIM RISK
    # ------------------------------------------------------

    {
        "id": "fraud_signals",
        "question": (
            "What are fraud risk "
            "investigation signals?"
        ),
        "expected_sources": [
            "fraud_investigation_guide.txt",
        ],
    },

    # ------------------------------------------------------
    # POLICY SERVICING
    # ------------------------------------------------------

    {
        "id": "policy_servicing",
        "question": (
            "What services can a "
            "policyholder request?"
        ),
        "expected_sources": [
            "policy_servicing_guide.txt",
        ],
    },

    # ------------------------------------------------------
    # CROSS-DOCUMENT
    # ------------------------------------------------------

    {
        "id": "claim_vs_fraud",
        "question": (
            "Compare claim processing "
            "with fraud risk review."
        ),
        "expected_sources": [
            "claim_guide.txt",
            "fraud_investigation_guide.txt",
        ],
    },
]


# ==========================================================
# UNIQUE SOURCES
# ==========================================================

def get_unique_sources(
    chunks: list[dict],
) -> list[str]:
    """
    Return retrieved source names in ranking order,
    while removing duplicate source names.

    Example:

    claim_guide.txt
    claim_guide.txt
    renewal_guide.txt

    becomes:

    claim_guide.txt
    renewal_guide.txt
    """

    sources = []

    for chunk in chunks:

        source = chunk.get(
            "source"
        )

        if (
            source
            and source not in sources
        ):

            sources.append(
                source
            )

    return sources


# ==========================================================
# TOP-1 CHECK
# ==========================================================

def calculate_top1_hit(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> bool:
    """
    Check whether the top-ranked document
    is one of the expected documents.
    """

    if not retrieved_sources:
        return False

    return (
        retrieved_sources[0]
        in expected_sources
    )


# ==========================================================
# ANY EXPECTED SOURCE HIT
# ==========================================================

def calculate_any_hit(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> bool:
    """
    True if at least one expected source
    was retrieved.
    """

    return any(
        source in retrieved_sources
        for source in expected_sources
    )


# ==========================================================
# ALL EXPECTED SOURCES HIT
# ==========================================================

def calculate_all_hit(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> bool:
    """
    True only when every expected source
    was retrieved.

    Important for cross-document questions.
    """

    return all(
        source in retrieved_sources
        for source in expected_sources
    )


# ==========================================================
# SOURCE PRECISION
# ==========================================================

def calculate_source_precision(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Of all unique documents retrieved,
    how many were expected?

    Higher is better.

    1.0 = no irrelevant document retrieved.
    """

    if not retrieved_sources:
        return 0.0

    relevant_count = sum(
        1
        for source in retrieved_sources
        if source in expected_sources
    )

    precision = (
        relevant_count
        / len(retrieved_sources)
    )

    return round(
        precision,
        4,
    )


# ==========================================================
# SOURCE RECALL
# ==========================================================

def calculate_source_recall(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Of all expected documents,
    how many were retrieved?

    Especially useful for cross-document questions.
    """

    if not expected_sources:
        return 0.0

    found_count = sum(
        1
        for source in expected_sources
        if source in retrieved_sources
    )

    recall = (
        found_count
        / len(expected_sources)
    )

    return round(
        recall,
        4,
    )


# ==========================================================
# IRRELEVANT SOURCE RATE
# ==========================================================

def calculate_irrelevant_source_rate(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Fraction of retrieved unique sources that
    were not expected.

    Lower is better.

    0.0 = ideal.
    """

    if not retrieved_sources:
        return 0.0

    irrelevant_count = sum(
        1
        for source in retrieved_sources
        if source not in expected_sources
    )

    rate = (
        irrelevant_count
        / len(retrieved_sources)
    )

    return round(
        rate,
        4,
    )


# ==========================================================
# RECIPROCAL RANK
# ==========================================================

def calculate_reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    """
    Measure how early the first relevant source
    appears in the ranking.

    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.3333
    """

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):

        if source in expected_sources:

            return round(
                1 / rank,
                4,
            )

    return 0.0


# ==========================================================
# EVALUATE ONE QUESTION
# ==========================================================

def evaluate_case(
    case: dict,
    run_generation: bool = False,
) -> dict:
    """
    Evaluate one RAG test case.
    """

    case_id = case[
        "id"
    ]

    question = case[
        "question"
    ]

    expected_sources = case[
        "expected_sources"
    ]

    logger.info(
        "Evaluating RAG case: %s",
        case_id,
    )

    # ======================================================
    # RETRIEVAL
    # ======================================================

    chunks = retrieve_context(
        question
    )

    retrieved_sources = get_unique_sources(
        chunks
    )

    # ======================================================
    # RETRIEVAL METRICS
    # ======================================================

    top1_hit = calculate_top1_hit(
        retrieved_sources,
        expected_sources,
    )

    any_hit = calculate_any_hit(
        retrieved_sources,
        expected_sources,
    )

    all_hit = calculate_all_hit(
        retrieved_sources,
        expected_sources,
    )

    source_precision = (
        calculate_source_precision(
            retrieved_sources,
            expected_sources,
        )
    )

    source_recall = (
        calculate_source_recall(
            retrieved_sources,
            expected_sources,
        )
    )

    irrelevant_source_rate = (
        calculate_irrelevant_source_rate(
            retrieved_sources,
            expected_sources,
        )
    )

    reciprocal_rank = (
        calculate_reciprocal_rank(
            retrieved_sources,
            expected_sources,
        )
    )

    # ======================================================
    # CHUNK DETAILS
    # ======================================================

    retrieved_chunks = []

    for chunk in chunks:

        retrieved_chunks.append(
            {
                "source": chunk.get(
                    "source"
                ),
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "semantic_score": chunk.get(
                    "semantic_score",
                    chunk.get(
                        "score"
                    ),
                ),
                "lexical_score": chunk.get(
                    "lexical_score"
                ),
                "rerank_score": chunk.get(
                    "rerank_score"
                ),
            }
        )

    result = {
        "id": case_id,
        "question": question,
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "retrieved_chunk_count": len(
            chunks
        ),
        "top1_hit": top1_hit,
        "any_expected_hit": any_hit,
        "all_expected_hit": all_hit,
        "source_precision": source_precision,
        "source_recall": source_recall,
        "irrelevant_source_rate": (
            irrelevant_source_rate
        ),
        "reciprocal_rank": (
            reciprocal_rank
        ),
        "retrieved_chunks": (
            retrieved_chunks
        ),
    }

    # ======================================================
    # OPTIONAL FULL GENERATION EVALUATION
    # ======================================================

    if run_generation:

        rag_result = answer_rag_question(
            question
        )

        grounding_validation = (
            rag_result.get(
                "grounding_validation"
            )
        )

        result[
            "generation"
        ] = {
            "answer": rag_result.get(
                "answer"
            ),
            "grounded": rag_result.get(
                "grounded",
                False,
            ),
            "sources": rag_result.get(
                "sources",
                [],
            ),
            "grounding_validation": (
                grounding_validation
            ),
        }

    return result


# ==========================================================
# CALCULATE SUMMARY
# ==========================================================

def calculate_summary(
    results: list[dict],
) -> dict:
    """
    Build overall evaluation metrics.
    """

    total_cases = len(
        results
    )

    if total_cases == 0:

        return {
            "total_cases": 0,
        }

    top1_hits = sum(
        1
        for item in results
        if item[
            "top1_hit"
        ]
    )

    any_hits = sum(
        1
        for item in results
        if item[
            "any_expected_hit"
        ]
    )

    all_hits = sum(
        1
        for item in results
        if item[
            "all_expected_hit"
        ]
    )

    average_precision = (
        sum(
            item[
                "source_precision"
            ]
            for item in results
        )
        / total_cases
    )

    average_recall = (
        sum(
            item[
                "source_recall"
            ]
            for item in results
        )
        / total_cases
    )

    average_irrelevant_rate = (
        sum(
            item[
                "irrelevant_source_rate"
            ]
            for item in results
        )
        / total_cases
    )

    mean_reciprocal_rank = (
        sum(
            item[
                "reciprocal_rank"
            ]
            for item in results
        )
        / total_cases
    )

    average_chunks = (
        sum(
            item[
                "retrieved_chunk_count"
            ]
            for item in results
        )
        / total_cases
    )

    summary = {
        "total_cases": total_cases,

        "top1_accuracy": round(
            top1_hits
            / total_cases,
            4,
        ),

        "any_expected_hit_rate": round(
            any_hits
            / total_cases,
            4,
        ),

        "all_expected_hit_rate": round(
            all_hits
            / total_cases,
            4,
        ),

        "average_source_precision": round(
            average_precision,
            4,
        ),

        "average_source_recall": round(
            average_recall,
            4,
        ),

        "average_irrelevant_source_rate": round(
            average_irrelevant_rate,
            4,
        ),

        "mean_reciprocal_rank": round(
            mean_reciprocal_rank,
            4,
        ),

        "average_retrieved_chunks": round(
            average_chunks,
            2,
        ),
    }

    # ======================================================
    # GENERATION / GROUNDING METRIC
    # ======================================================

    generated_results = [
        item
        for item in results
        if "generation" in item
    ]

    if generated_results:

        grounded_count = sum(
            1
            for item in generated_results
            if item[
                "generation"
            ].get(
                "grounded",
                False,
            )
        )

        summary[
            "grounding_pass_rate"
        ] = round(
            grounded_count
            / len(
                generated_results
            ),
            4,
        )

    return summary


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(
    report: dict,
) -> Path:
    """
    Save evaluation output to JSON.
    """

    RAG_EVAL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        RAG_EVAL_REPORT_DIR
        / "rag_evaluation_latest.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "RAG evaluation report saved: %s",
        report_path,
    )

    return report_path


# ==========================================================
# RUN FULL EVALUATION
# ==========================================================

def run_evaluation(
    run_generation: bool | None = None,
) -> dict:
    """
    Execute the complete RAG evaluation suite.
    """

    if run_generation is None:

        run_generation = (
            RAG_EVAL_RUN_GENERATION
        )

    logger.info(
        "Starting RAG evaluation. "
        "Generation evaluation=%s",
        run_generation,
    )

    results = []

    for case in EVALUATION_CASES:

        result = evaluate_case(
            case=case,
            run_generation=run_generation,
        )

        results.append(
            result
        )

    summary = calculate_summary(
        results
    )

    report = {
        "generated_at": (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        ),

        "generation_evaluation_enabled": (
            run_generation
        ),

        "summary": summary,

        "cases": results,
    }

    save_report(
        report
    )

    return report


# ==========================================================
# PRINT REPORT
# ==========================================================

def print_report(
    report: dict,
) -> None:
    """
    Print readable terminal results.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG EVALUATION REPORT"
    )

    print(
        "=" * 70
    )

    for case in report[
        "cases"
    ]:

        print(
            "\nQUESTION:"
        )

        print(
            case[
                "question"
            ]
        )

        print(
            "Expected:",
            case[
                "expected_sources"
            ],
        )

        print(
            "Retrieved:",
            case[
                "retrieved_sources"
            ],
        )

        print(
            "Top-1:",
            case[
                "top1_hit"
            ],
        )

        print(
            "All Expected:",
            case[
                "all_expected_hit"
            ],
        )

        print(
            "Precision:",
            case[
                "source_precision"
            ],
        )

        print(
            "Recall:",
            case[
                "source_recall"
            ],
        )

        print(
            "Irrelevant Rate:",
            case[
                "irrelevant_source_rate"
            ],
        )

        print(
            "Reciprocal Rank:",
            case[
                "reciprocal_rank"
            ],
        )

        if "generation" in case:

            print(
                "Grounded:",
                case[
                    "generation"
                ][
                    "grounded"
                ],
            )

    # ======================================================
    # SUMMARY
    # ======================================================

    summary = report[
        "summary"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 70
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    evaluation_report = (
        run_evaluation()
    )

    print_report(
        evaluation_report
    )