# src/agents/router.py

import re

from typing import Any, Callable, TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from src.utils.llm import llm

from src.agents.sql_agent import (
    ask_sql_agent,
)

from src.agents.ml_agent import (
    ask_ml_agent,
)

from src.agents.rag_agent import (
    ask_rag_agent,
)

from src.memory.session_memory import (
    session_memory,
)

from src.memory.context_manager import (
    resolve_question,
)

from src.memory.conversation_store import (
    get_session_route as db_get_session_route,
    set_session_route as db_set_session_route,
)


# ==========================================================
# PERSISTENT SESSION ROUTE MEMORY
# ==========================================================

VALID_ROUTES = {
    "sql",
    "ml",
    "rag",
    "general",
}


def get_session_route(
    session_id: str | None,
) -> str | None:
    """
    Read the previous successful route
    from PostgreSQL.
    """

    if not session_id:
        return None

    return db_get_session_route(
        session_id
    )


def set_session_route(
    session_id: str | None,
    route: str | None,
) -> None:
    """
    Save the latest successful route
    in PostgreSQL.
    """

    if (
        not session_id
        or route not in VALID_ROUTES
    ):
        return

    db_set_session_route(
        session_id=session_id,
        route=route,
    )


def clear_copilot_session(
    session_id: str,
) -> bool:
    """
    Delete one persistent Copilot session.

    Messages are automatically removed
    because of ON DELETE CASCADE.
    """

    if not session_id:
        return False

    return session_memory.clear_session(
        session_id
    )


# ==========================================================
# GRAPH STATE
# ==========================================================

class CopilotState(
    TypedDict,
    total=False,
):

    # Question actually sent to router/agent
    question: str

    # Original user question
    original_question: str

    # Context resolved standalone question
    resolved_question: str

    # ML input features
    input_data: dict[str, Any]

    # Conversation
    session_id: str

    follow_up_detected: bool

    was_rewritten: bool

    context_topic: str

    previous_route: str

    route_inherited: bool

    # sql / ml / rag / general
    route: str

    # Agent output
    result: dict[str, Any]


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize_question(
    question: str,
) -> str:
    """
    Normalize user question for
    deterministic routing.
    """

    if not question:
        return ""

    text = (
        question
        .lower()
        .strip()
    )

    text = re.sub(
        r"[^\w\s₹%/-]",
        " ",
        text,
    )

    replacements = {

        "policies":
            "policy",

        "types":
            "type",

        "kinds":
            "kind",

        "categories":
            "category",

        "customers":
            "customer",

        "claims":
            "claim",

        "agents":
            "agent",

        "premiums":
            "premium",

        "renewals":
            "renewal",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_any(
    text: str,
    phrases: list[str],
) -> bool:
    """
    True when at least one phrase exists.
    """

    return any(
        phrase in text
        for phrase in phrases
    )


# ==========================================================
# ML ROUTING PHRASES
# ==========================================================

ML_PHRASES = [

    # Generic ML
    "predict",
    "prediction",
    "probability",
    "likelihood",
    "risk assessment",
    "run model",
    "run the model",
    "model prediction",
    "classify this",

    # Renewal
    "will this customer renew",
    "will customer renew",
    "customer renew karega",
    "renew karega kya",
    "renewal probability",
    "renewal prediction",
    "predict renewal",
    "non renewal risk",
    "churn probability",
    "likely to churn",
    "predict churn",

    # Fraud
    "predict fraud",
    "fraud probability",
    "fraud prediction",
    "fraud risk for this claim",
    "fraud risk hai",
    "is this claim suspicious",
    "claim suspicious hai",
    "is this claim fraudulent",
    "should this claim be investigated",
    "calculate fraud risk",

    # Underwriting
    "assess this applicant",
    "evaluate this applicant",
    "underwriting assessment",
    "underwriting prediction",
    "predicted underwriting",
    "run underwriting",
    "applicant risk",
    "underwriting assessment karo",
]


# ==========================================================
# RAG ROUTING PHRASES
# ==========================================================

RAG_PHRASES = [

    "according to our guide",
    "according to the guide",

    "according to our document",
    "according to the document",

    "according to our manual",
    "according to company guideline",

    "claim guide",
    "claim manual",

    "renewal guide",
    "underwriting guide",
    "fraud investigation guide",

    "policy document",
    "policy manual",

    "company document",
    "company manual",

    "uploaded document",
    "uploaded file",

    "knowledge base",

    "internal guide",
    "internal guideline",

    "what does the document say",
    "what does our document say",

    "what does the guide say",
    "what does our guide say",

    "what does the manual say",

    # Hinglish
    "guide me kya",
    "document me kya",
    "manual me kya",
    "claim guide ke according",
]


CLAIM_DOCUMENT_PHRASES = [

    "documents required for a claim",

    "documents required for insurance claim",

    "documents required for an insurance claim",

    "documents needed for a claim",

    "claim document required",

    "supporting documents for claim",

    "what documents are required for claim",

    "what documents are required for an insurance claim",

    "what documents are required for insurance claim",

    "required documents for an insurance claim",

    "required documents for insurance claim",

    "required claim documents",

    "insurance claim documents",

    "claim submission procedure",

    "claim verification procedure",

    "claim processing procedure",

    "claim processing",
]


# ==========================================================
# SQL ROUTING PHRASES
# ==========================================================

DATA_CONTEXT_PHRASES = [

    "our data",

    "our dataset",

    "our database",

    "in our database",

    "in our data",

    "in the dataset",

    "company data",

    "company record",

    "database record",

    "our customer",

    "our policy",

    "our claim",

    "our premium",

    "our renewal",

    "our agent",

    "our business",

    "our portfolio",
]


ANALYTICS_PHRASES = [

    "how many",

    "total",

    "sum of",

    "average",

    "avg ",

    "mean ",

    "minimum",

    "maximum",

    "highest",

    "lowest",

    "top 3",

    "top 5",

    "top 10",

    "bottom 5",

    "bottom 10",

    "count",

    "percentage",

    "distribution",

    "breakdown",

    "show me",

    "show customer",

    "show policy",

    "show claim",

    "show agent",

    "show premium",

    "show renewal",

    "list customer",

    "list policy",

    "list claim",

    "list agent",

    "compare",

    "ranking",

    "rank ",

    "trend",

    "which state",

    "which customer",

    "which policy",

    "which agent",

    "renewal rate",

    "claim amount",

    "premium by",

    "claim by",

    "policy by",

    "customer by",

    # Hinglish
    "kitne customer",

    "kitni policy",

    "kitne claim",

    "sabse jyada",

    "sabse zyada",

    "sabse kam",

    "dikhao",
]


DATABASE_ENTITIES = [

    "customer",

    "policy",

    "claim",

    "premium",

    "renewal",

    "agent",

    "underwriting",

    "sales channel",

    "state",

    "credit score",

    "risk segment",

    "risk band",

    "payment delay",

    "annual income",

    "income",

    "claim amount",

    "sum insured",

    "payment",
]


SHORT_SQL_PHRASES = [

    "policy type in database",

    "policy type exist in database",

    "policy type do we have",

    "state with most customer",

    "highest premium customer",

    "highest claim",

    "best performing agent",

    "customer with multiple claim",

    "customer with delayed payment",

    "customer who did not renew",

    "top agent",

    "top customer",

    "top policy",
]


# ==========================================================
# LOCAL ROUTER
# ==========================================================

def detect_route(
    question: str,
) -> str:
    """
    Deterministic routing.

    Priority:

    1. ML
    2. RAG
    3. SQL
    4. GENERAL
    """

    q = normalize_question(
        question
    )

    # ======================================================
    # ML
    # ======================================================

    if contains_any(
        q,
        ML_PHRASES,
    ):

        return "ml"

    # ======================================================
    # RAG - EXPLICIT DOCUMENT QUESTIONS
    # ======================================================

    if contains_any(
        q,
        RAG_PHRASES,
    ):

        return "rag"

    # ======================================================
    # RAG - CLAIM DOCUMENT QUESTIONS
    # ======================================================

    if contains_any(
        q,
        CLAIM_DOCUMENT_PHRASES,
    ):

        return "rag"

    # ======================================================
    # RAG - FLEXIBLE CLAIM PROCESS QUESTIONS
    # ======================================================

    has_claim_context = (
        "claim" in q
    )

    has_claim_knowledge_intent = (
        contains_any(
            q,
            [
                "document",
                "documents",

                "claim form",

                "proof of loss",

                "supporting evidence",

                "submit",

                "submitted",

                "submission",

                "verification",

                "verify",

                "claim process",

                "claim processing",

                "claim procedure",

                "settlement process",

                "after i submit",

                "after submitting",

                "required insurance claim documents",
            ],
        )
    )

    if (
        has_claim_context
        and has_claim_knowledge_intent
    ):

        return "rag"

    # ======================================================
    # RAG - OTHER INTERNAL GUIDE DOMAINS
    # ======================================================

    has_renewal_knowledge_intent = (
        "renewal" in q
        and
        contains_any(
            q,
            [
                "review",
                "information",
                "factors",
                "process",
                "before renewal",
                "premium increase",
            ],
        )
    )

    if has_renewal_knowledge_intent:

        return "rag"

    has_underwriting_knowledge_intent = (
        "underwriting" in q
        and
        contains_any(
            q,
            [
                "factor",
                "factors",
                "considered",
                "review",
                "process",
                "decision",
                "loading",
            ],
        )
    )

    if has_underwriting_knowledge_intent:

        return "rag"

    has_fraud_knowledge_intent = (
        (
            "fraud" in q
            or "investigation" in q
        )
        and
        contains_any(
            q,
            [
                "signal",
                "signals",
                "indicator",
                "indicators",
                "review",
                "investigation",
                "process",
            ],
        )
    )

    if has_fraud_knowledge_intent:

        return "rag"

    has_servicing_knowledge_intent = (
        (
            "policyholder" in q
            or "servicing" in q
        )
        and
        contains_any(
            q,
            [
                "service",
                "services",
                "request",
                "requests",
                "change",
                "update",
            ],
        )
    )

    if has_servicing_knowledge_intent:

        return "rag"

    # ======================================================
    # SQL
    # ======================================================

    has_data_context = (
        contains_any(
            q,
            DATA_CONTEXT_PHRASES,
        )
    )

    has_analytics_intent = (
        contains_any(
            q,
            ANALYTICS_PHRASES,
        )
    )

    has_database_entity = (
        contains_any(
            q,
            DATABASE_ENTITIES,
        )
    )

    if (
        has_data_context
        and has_database_entity
    ):

        return "sql"

    if (
        has_analytics_intent
        and has_database_entity
    ):

        return "sql"

    if contains_any(
        q,
        SHORT_SQL_PHRASES,
    ):

        return "sql"

    # ======================================================
    # GENERAL
    # ======================================================

    return "general"


# ==========================================================
# ROUTER NODE
# ==========================================================

def router_node(
    state: CopilotState,
):
    """
    Choose destination agent.

    If a follow-up becomes too generic,
    preserve previous SQL/RAG/ML route.
    """

    route = detect_route(
        state[
            "question"
        ]
    )

    route_inherited = False

    # ======================================================
    # FOLLOW-UP ROUTE CONTINUITY
    # ======================================================

    if (
        state.get(
            "follow_up_detected",
            False,
        )
        and route == "general"
    ):

        previous_route = (
            state.get(
                "previous_route"
            )
        )

        if previous_route in {
            "sql",
            "rag",
            "ml",
        }:

            route = previous_route

            route_inherited = True

    return {

        "route":
            route,

        "route_inherited":
            route_inherited,
    }


# ==========================================================
# RATE LIMIT DETECTION
# ==========================================================

def is_rate_limit_error(
    error: Exception,
) -> bool:

    error_text = (
        str(
            error
        )
        .lower()
    )

    markers = [

        "429",

        "rate limit",

        "rate_limit",

        "free-models-per-day",

        "free models per day",

        "too many requests",
    ]

    return any(
        marker in error_text
        for marker in markers
    )


# ==========================================================
# SAFE AGENT EXECUTION
# ==========================================================

def safe_agent_call(
    agent_name: str,
    function: Callable,
    *args,
    **kwargs,
) -> dict[str, Any]:
    """
    Prevent individual agents from
    crashing the Copilot.
    """

    try:

        result = function(
            *args,
            **kwargs,
        )

        if isinstance(
            result,
            dict,
        ):

            return result

        return {

            "answer":
                str(
                    result
                )
        }

    except Exception as error:

        if is_rate_limit_error(
            error
        ):

            return {

                "answer": (
                    f"{agent_name} cannot currently use "
                    "the AI service because the request "
                    "limit has been reached. "
                    "Please try again later."
                ),

                "error_type":
                    "rate_limit",
            }

        return {

            "answer": (
                f"{agent_name} could not complete "
                "the request."
            ),

            "error":
                str(
                    error
                ),
        }


# ==========================================================
# SQL NODE
# ==========================================================

def sql_node(
    state: CopilotState,
):

    result = safe_agent_call(

        "SQL Agent",

        ask_sql_agent,

        state[
            "question"
        ],
    )

    return {

        "result":
            result
    }


# ==========================================================
# RAG NODE
# ==========================================================

def rag_node(
    state: CopilotState,
):

    result = safe_agent_call(

        "RAG Agent",

        ask_rag_agent,

        question=state[
            "question"
        ],
    )

    return {

        "result":
            result
    }


# ==========================================================
# ML NODE
# ==========================================================

def ml_node(
    state: CopilotState,
):
    """
    Run ML model when input exists.

    Otherwise provide local guidance.
    """

    question = state[
        "question"
    ]

    input_data = state.get(
        "input_data",
        {},
    )

    q = normalize_question(
        question
    )

    # ======================================================
    # NO INPUT DATA
    # ======================================================

    if not input_data:

        # --------------------------------------------------
        # RENEWAL
        # --------------------------------------------------

        if contains_any(
            q,
            [
                "renew",
                "renewal",
                "churn",
                "non renewal",
            ],
        ):

            return {

                "result": {

                    "answer": (
                        "### Renewal Prediction\n\n"

                        "I can predict whether a customer is likely "
                        "to renew, but the model needs the customer's "
                        "profile, policy, payment and claim information "
                        "before it can make a prediction.\n\n"

                        "Please open **Renewal Intelligence** from the "
                        "sidebar, enter the required customer details, "
                        "and click **Analyze Renewal Risk**.\n\n"

                        "The model will provide:\n\n"

                        "- **Renewal Probability**\n"

                        "- **Churn Probability**\n"

                        "- **Risk Classification**\n"

                        "- **Decision Threshold**"
                    ),

                    "model_type":
                        "renewal",

                    "requires_input":
                        True,

                    "source":
                        "local_ml_guidance",
                }
            }

        # --------------------------------------------------
        # FRAUD
        # --------------------------------------------------

        if contains_any(
            q,
            [
                "fraud",
                "suspicious claim",
                "claim suspicious",
                "fraudulent",
            ],
        ):

            return {

                "result": {

                    "answer": (
                        "### Fraud Risk Assessment\n\n"

                        "I can calculate a fraud-risk signal, but the "
                        "model requires claim and customer information "
                        "before it can perform the assessment.\n\n"

                        "Please open **Fraud Intelligence** from the "
                        "sidebar, enter the claim details, and click "
                        "**Analyze Claim Risk**.\n\n"

                        "The model will provide:\n\n"

                        "- **Fraud Risk Probability**\n"

                        "- **Investigation Threshold**\n"

                        "- **Risk Signal**\n\n"

                        "A fraud-risk flag is an investigation signal "
                        "and does not establish that fraud occurred."
                    ),

                    "model_type":
                        "fraud",

                    "requires_input":
                        True,

                    "source":
                        "local_ml_guidance",
                }
            }

        # --------------------------------------------------
        # UNDERWRITING
        # --------------------------------------------------

        if contains_any(
            q,
            [
                "underwriting",
                "applicant",
                "applicant risk",
            ],
        ):

            return {

                "result": {

                    "answer": (
                        "### Underwriting Assessment\n\n"

                        "I can run the underwriting model, but the "
                        "applicant's information is required first.\n\n"

                        "Please open **Underwriting** from the sidebar, "
                        "enter the applicant details, and click "
                        "**Run Assessment**.\n\n"

                        "The model will provide:\n\n"

                        "- **Predicted Underwriting Category**\n"

                        "- **Class Probabilities**\n\n"

                        "This model is decision-support only. "
                        "Final underwriting decisions require "
                        "qualified human review."
                    ),

                    "model_type":
                        "underwriting",

                    "requires_input":
                        True,

                    "source":
                        "local_ml_guidance",
                }
            }

        # --------------------------------------------------
        # GENERIC
        # --------------------------------------------------

        return {

            "result": {

                "answer": (
                    "INSURE AI currently supports three "
                    "predictive workflows:\n\n"

                    "- **Renewal Prediction**\n"

                    "- **Fraud Risk Assessment**\n"

                    "- **Underwriting Assessment**\n\n"

                    "Please provide the required case data "
                    "or open the corresponding model "
                    "workspace from the sidebar."
                ),

                "requires_input":
                    True,

                "source":
                    "local_ml_guidance",
            }
        }

    # ======================================================
    # INPUT DATA EXISTS
    # ======================================================

    result = safe_agent_call(

        "ML Agent",

        ask_ml_agent,

        question=question,

        input_data=input_data,
    )

    return {

        "result":
            result
    }


# ==========================================================
# LOCAL GENERAL INSURANCE KNOWLEDGE
# ==========================================================

GENERAL_KB = [

    # ------------------------------------------------------
    # INSURANCE
    # ------------------------------------------------------

    (
        [
            "what is insurance",
            "insurance meaning",
            "insurance kya hai",
            "insurance kya hota hai",
            "explain insurance",
        ],

        """
### What is Insurance?

Insurance is a financial arrangement that helps protect a person or business against specified financial losses.

The customer pays a **premium** to an insurer. In return, the insurer provides protection against risks covered by the insurance policy.

The exact coverage depends on the policy terms.
""",
    ),

    # ------------------------------------------------------
    # WHY INSURANCE
    # ------------------------------------------------------

    (
        [
            "why insurance",
            "why do we need insurance",
            "importance of insurance",
            "insurance kyu",
            "insurance kyun",
        ],

        """
### Why is Insurance Important?

Insurance helps reduce the financial impact of unexpected events.

Main benefits include:

- Financial protection
- Risk transfer
- Emergency support
- Family financial security
- Business continuity
- Protection against large unexpected losses

Insurance does not remove risk. It helps manage its financial consequences.
""",
    ),

    # ------------------------------------------------------
    # POLICY
    # ------------------------------------------------------

    (
        [
            "what is insurance policy",
            "what is an insurance policy",
            "policy meaning",
            "policy kya hai",
            "policy kya hoti hai",
        ],

        """
### What is an Insurance Policy?

An insurance policy is the contract between the insurer and the policyholder.

It usually describes:

- Coverage
- Exclusions
- Premium
- Sum insured
- Policy period
- Deductibles or co-payments
- Claim conditions
- Terms and conditions
""",
    ),

    # ------------------------------------------------------
    # PREMIUM
    # ------------------------------------------------------

    (
        [
            "what is premium",
            "what is insurance premium",
            "premium meaning",
            "premium kya hai",
            "premium kya hota hai",
            "explain premium",
        ],

        """
### Insurance Premium

An insurance premium is the amount a customer pays to an insurer for insurance coverage.

Premiums may be paid:

- Monthly
- Quarterly
- Half-yearly
- Annually
""",
    ),

    # ------------------------------------------------------
    # PREMIUM FACTORS
    # ------------------------------------------------------

    (
        [
            "how premium calculated",
            "how is premium calculated",
            "premium calculation",
            "factor affect premium",
            "what affects premium",
            "why premium increase",
        ],

        """
### What Factors Affect Premium?

Depending on the insurance product, premium may be influenced by:

- Age
- Health
- Coverage amount
- Policy type
- Claim history
- Occupation
- Location
- Lifestyle
- Risk profile
- Deductible
- Previous losses
""",
    ),

    # ------------------------------------------------------
    # SUM INSURED
    # ------------------------------------------------------

    (
        [
            "what is sum insured",
            "sum insured meaning",
            "sum insured kya",
        ],

        """
### Sum Insured

The **sum insured** is the maximum applicable coverage amount specified under an insurance policy, subject to policy terms and limits.
""",
    ),

    # ------------------------------------------------------
    # PREMIUM VS SUM INSURED
    # ------------------------------------------------------

    (
        [
            "premium vs sum insured",
            "difference between premium and sum insured",
        ],

        """
### Premium vs Sum Insured

**Premium**

The amount the customer pays for insurance.

**Sum Insured**

The maximum applicable coverage amount under the policy.
""",
    ),

    # ------------------------------------------------------
    # COVERAGE
    # ------------------------------------------------------

    (
        [
            "what is coverage",
            "coverage meaning",
            "insurance coverage meaning",
        ],

        """
### Insurance Coverage

Coverage refers to the risks, events, expenses or losses that an insurance policy protects against.

The exact coverage depends on the policy wording.
""",
    ),

    # ------------------------------------------------------
    # EXCLUSION
    # ------------------------------------------------------

    (
        [
            "what is exclusion",
            "what are exclusion",
            "exclusion meaning",
            "insurance exclusion",
        ],

        """
### Insurance Exclusion

An exclusion is an event, condition or loss that an insurance policy specifically does **not** cover.
""",
    ),

    # ------------------------------------------------------
    # DEDUCTIBLE
    # ------------------------------------------------------

    (
        [
            "what is deductible",
            "deductible meaning",
            "deductible kya",
        ],

        """
### Deductible

A deductible is the amount the policyholder bears before the insurer pays the remaining eligible portion of a covered claim.
""",
    ),

    # ------------------------------------------------------
    # COPAY
    # ------------------------------------------------------

    (
        [
            "what is copay",
            "what is co pay",
            "what is copayment",
            "what is co-payment",
        ],

        """
### Co-payment

A co-payment is a percentage or fixed portion of an eligible claim that the policyholder must pay.
""",
    ),

    # ------------------------------------------------------
    # WAITING PERIOD
    # ------------------------------------------------------

    (
        [
            "what is waiting period",
            "waiting period meaning",
            "waiting period kya",
        ],

        """
### Waiting Period

A waiting period is a specified period after a policy starts during which certain benefits or conditions may not yet be covered.
""",
    ),

    # ------------------------------------------------------
    # GRACE PERIOD
    # ------------------------------------------------------

    (
        [
            "what is grace period",
            "grace period meaning",
            "grace period kya",
        ],

        """
### Grace Period

A grace period is an additional period after a premium due date during which the policyholder may still be allowed to make payment.

Exact conditions depend on the insurance product.
""",
    ),

    # ------------------------------------------------------
    # CLAIM
    # ------------------------------------------------------

    (
        [
            "what is insurance claim",
            "what is claim",
            "claim meaning",
            "claim kya",
        ],

        """
### Insurance Claim

An insurance claim is a formal request made to an insurer for payment or service following a covered event.

A simplified process is:

1. Insured event occurs
2. Customer informs insurer
3. Claim is registered
4. Documents are submitted
5. Coverage is verified
6. Investigation may occur
7. Claim is assessed
8. Eligible claim is settled
""",
    ),

    # ------------------------------------------------------
    # CLAIM SETTLEMENT
    # ------------------------------------------------------

    (
        [
            "what is claim settlement",
            "claim settlement meaning",
            "how claim settlement work",
        ],

        """
### Claim Settlement

Claim settlement is the process through which an insurer evaluates a claim and determines the eligible payment or covered service.

It may include:

- Claim registration
- Document verification
- Coverage verification
- Loss assessment
- Investigation
- Decision
- Settlement
""",
    ),

    # ------------------------------------------------------
    # CASHLESS CLAIM
    # ------------------------------------------------------

    (
        [
            "what is cashless claim",
            "cashless hospitalization",
            "cashless insurance",
        ],

        """
### Cashless Claim

In a cashless claim, eligible expenses may be settled directly between the insurer or service administrator and an approved provider.

The customer may still need to pay non-covered expenses, deductibles or co-payments.
""",
    ),

    # ------------------------------------------------------
    # REIMBURSEMENT CLAIM
    # ------------------------------------------------------

    (
        [
            "what is reimbursement claim",
            "reimbursement claim meaning",
        ],

        """
### Reimbursement Claim

In a reimbursement claim, the customer generally pays the expense first and then submits eligible documents to the insurer.

Eligible expenses may then be reimbursed according to policy conditions.
""",
    ),

    # ------------------------------------------------------
    # CLAIM REJECTION
    # ------------------------------------------------------

    (
        [
            "why claim rejected",
            "why claim are rejected",
            "claim rejection reason",
            "reason for claim rejection",
        ],

        """
### Why Can a Claim Be Rejected?

Possible reasons include:

- Event is not covered
- Policy was inactive
- An exclusion applies
- Required documents are missing
- Incorrect information was provided
- Policy conditions were not satisfied
- Material information was misrepresented
- Coverage limits were exceeded
""",
    ),

    # ------------------------------------------------------
    # UNDERWRITING
    # ------------------------------------------------------

    (
        [
            "what is underwriting",
            "underwriting meaning",
            "underwriting kya",
            "explain underwriting",
        ],

        """
### Underwriting

Underwriting is the process through which an insurer evaluates the risk associated with providing insurance.

Factors may include:

- Age
- Health
- Lifestyle
- Occupation
- Claim history
- Coverage requested
- Other risk characteristics

In INSURE AI, underwriting model output is decision-support only and requires qualified human review.
""",
    ),

    # ------------------------------------------------------
    # UNDERWRITER
    # ------------------------------------------------------

    (
        [
            "what does underwriter do",
            "role of underwriter",
            "what is underwriter",
        ],

        """
### Role of an Underwriter

An underwriter evaluates insurance risk.

Typical responsibilities include:

- Reviewing applicant information
- Evaluating risk exposure
- Applying underwriting guidelines
- Determining suitable policy terms
- Supporting pricing decisions
- Escalating complex cases
""",
    ),

    # ------------------------------------------------------
    # RENEWAL
    # ------------------------------------------------------

    (
        [
            "what is policy renewal",
            "what is renewal",
            "renewal meaning",
            "renewal kya",
        ],

        """
### Policy Renewal

Policy renewal is the process of continuing insurance coverage for another policy period.

At renewal an insurer may review:

- Premium
- Coverage
- Risk profile
- Claim experience
- Policy conditions
""",
    ),

    # ------------------------------------------------------
    # POLICY LAPSE
    # ------------------------------------------------------

    (
        [
            "what is policy lapse",
            "policy lapse meaning",
            "lapsed policy",
        ],

        """
### Policy Lapse

A policy lapse occurs when insurance coverage stops because required conditions, commonly premium payment requirements, were not satisfied.
""",
    ),

    # ------------------------------------------------------
    # FRAUD
    # ------------------------------------------------------

    (
        [
            "what is insurance fraud",
            "what is claim fraud",
            "what is fraud detection",
            "fraud detection meaning",
        ],

        """
### Insurance Fraud

Insurance fraud involves intentionally providing false, misleading or manipulated information to obtain an insurance benefit improperly.

Fraud-detection systems identify **risk signals** for investigation.

A model flag is not proof that fraud occurred.
""",
    ),

    # ------------------------------------------------------
    # FRAUD INDICATORS
    # ------------------------------------------------------

    (
        [
            "fraud indicator",
            "fraud red flag",
            "claim red flag",
            "sign of insurance fraud",
        ],

        """
### Potential Fraud-Risk Indicators

Examples may include:

- Unusually high claim amount
- Inconsistent incident details
- Repeated claims
- Unusual reporting delay
- Suspicious documentation
- Multiple related claims
- Abnormal claim patterns

These are investigation signals only and do not prove fraud.
""",
    ),

    # ------------------------------------------------------
    # POLICYHOLDER
    # ------------------------------------------------------

    (
        [
            "what is policyholder",
            "who is policyholder",
            "policyholder meaning",
        ],

        """
A **policyholder** is the person or organization that owns the insurance policy and enters into the insurance contract with the insurer.
""",
    ),

    # ------------------------------------------------------
    # BENEFICIARY
    # ------------------------------------------------------

    (
        [
            "what is beneficiary",
            "who is beneficiary",
            "beneficiary meaning",
        ],

        """
A **beneficiary** is a person or entity designated to receive insurance benefits when the required conditions for payment are met.
""",
    ),

    # ------------------------------------------------------
    # NOMINEE
    # ------------------------------------------------------

    (
        [
            "what is nominee",
            "who is nominee",
            "nominee meaning",
        ],

        """
A **nominee** is a person named by the policyholder in relation to receipt of policy benefits after the policyholder's death, subject to applicable terms and legal requirements.
""",
    ),

    # ------------------------------------------------------
    # RIDER
    # ------------------------------------------------------

    (
        [
            "what is rider",
            "insurance rider meaning",
            "rider meaning",
        ],

        """
### Insurance Rider

A rider is an additional benefit or coverage attached to a base insurance policy, often for an additional premium.
""",
    ),

    # ------------------------------------------------------
    # ENDORSEMENT
    # ------------------------------------------------------

    (
        [
            "what is endorsement",
            "insurance endorsement",
            "endorsement meaning",
        ],

        """
### Insurance Endorsement

An endorsement is an official amendment or change made to an existing insurance policy.
""",
    ),

    # ------------------------------------------------------
    # NCB
    # ------------------------------------------------------

    (
        [
            "what is no claim bonus",
            "what is ncb",
            "no claim bonus meaning",
        ],

        """
### No Claim Bonus

A No Claim Bonus is a benefit that may be offered when no eligible claim is made during a specified policy period.

Depending on the product, it may provide a premium discount or another renewal benefit.
""",
    ),

    # ------------------------------------------------------
    # REINSURANCE
    # ------------------------------------------------------

    (
        [
            "what is reinsurance",
            "reinsurance meaning",
        ],

        """
### Reinsurance

Reinsurance is insurance purchased by an insurance company to transfer part of its own risk to another insurer or reinsurer.
""",
    ),

    # ------------------------------------------------------
    # INDEMNITY
    # ------------------------------------------------------

    (
        [
            "what is indemnity",
            "indemnity insurance",
            "principle of indemnity",
        ],

        """
### Principle of Indemnity

The principle of indemnity generally aims to restore the insured to approximately the financial position they were in before a covered loss rather than allowing profit from the loss.
""",
    ),

    # ------------------------------------------------------
    # INSURABLE INTEREST
    # ------------------------------------------------------

    (
        [
            "what is insurable interest",
            "insurable interest meaning",
        ],

        """
### Insurable Interest

Insurable interest means the insured has a legitimate interest in the person, property or risk being insured.
""",
    ),

    # ------------------------------------------------------
    # UTMOST GOOD FAITH
    # ------------------------------------------------------

    (
        [
            "utmost good faith",
            "uberrimae fidei",
        ],

        """
### Utmost Good Faith

Utmost good faith is the insurance principle that material information should be disclosed truthfully and accurately.
""",
    ),

    # ------------------------------------------------------
    # SUBROGATION
    # ------------------------------------------------------

    (
        [
            "what is subrogation",
            "subrogation meaning",
        ],

        """
### Subrogation

Subrogation can allow an insurer, after paying an eligible claim, to pursue recovery from a responsible third party where legally permitted.
""",
    ),

    # ------------------------------------------------------
    # ADVERSE SELECTION
    # ------------------------------------------------------

    (
        [
            "what is adverse selection",
            "adverse selection meaning",
        ],

        """
### Adverse Selection

Adverse selection occurs when higher-risk customers are more likely to seek insurance while the insurer has incomplete information about their risk.

Underwriting helps insurers manage this issue.
""",
    ),

    # ------------------------------------------------------
    # MORAL HAZARD
    # ------------------------------------------------------

    (
        [
            "what is moral hazard",
            "moral hazard insurance",
        ],

        """
### Moral Hazard

Moral hazard refers to behavior changes that may occur because a person is protected from some financial consequences of risk.
""",
    ),

    # ------------------------------------------------------
    # RISK POOLING
    # ------------------------------------------------------

    (
        [
            "what is risk pooling",
            "risk pooling meaning",
        ],

        """
### Risk Pooling

Risk pooling combines many insured risks so losses experienced by some members can be funded from premiums collected across the larger group.
""",
    ),

    # ------------------------------------------------------
    # CHURN
    # ------------------------------------------------------

    (
        [
            "what is customer churn",
            "what is churn",
            "churn in insurance",
        ],

        """
### Customer Churn

Customer churn in insurance generally refers to customers leaving, cancelling or not renewing their policies.
""",
    ),

    # ------------------------------------------------------
    # RETENTION
    # ------------------------------------------------------

    (
        [
            "what is customer retention",
            "customer retention meaning",
            "retention in insurance",
        ],

        """
### Customer Retention

Customer retention refers to an insurer's ability to keep existing customers and policies over time.
""",
    ),

    # ------------------------------------------------------
    # INSURANCE LIFECYCLE
    # ------------------------------------------------------

    (
        [
            "insurance lifecycle",
            "insurance life cycle",
            "insurance process end to end",
        ],

        """
### Simplified Insurance Lifecycle

1. Customer need identified
2. Product selection
3. Application
4. Underwriting
5. Policy issuance
6. Premium collection
7. Policy servicing
8. Claim handling
9. Renewal or lapse
10. Ongoing customer relationship
""",
    ),

    # ------------------------------------------------------
    # CLAIM FREQUENCY
    # ------------------------------------------------------

    (
        [
            "what is claim frequency",
            "claim frequency meaning",
        ],

        """
### Claim Frequency

Claim frequency refers to how often claims occur within a defined period or insurance portfolio.
""",
    ),

    # ------------------------------------------------------
    # CLAIM SEVERITY
    # ------------------------------------------------------

    (
        [
            "what is claim severity",
            "claim severity meaning",
        ],

        """
### Claim Severity

Claim severity refers to the size or financial impact of an insurance claim.
""",
    ),
]


# ==========================================================
# POLICY TYPES
# ==========================================================

POLICY_TYPES_ANSWER = """
### Common Types of Insurance Policies

- **Health Insurance**
- **Motor Insurance**
- **Term Life Insurance**
- **Whole Life Insurance**
- **Home Insurance**
- **Travel Insurance**
- **Personal Accident Insurance**
- **Commercial Insurance**

Exact coverage, exclusions and conditions depend on the individual policy.
"""


# ==========================================================
# LOCAL GENERAL ANSWER
# ==========================================================

def local_general_answer(
    question: str,
) -> str | None:
    """
    Use local FAQ knowledge before LLM.

    Longest matched phrase wins.

    This prevents:

    "What is insurance fraud?"

    from accidentally matching:

    "What is insurance?"
    """

    q = normalize_question(
        question
    )

    best_answer = None

    best_match_length = -1

    # ======================================================
    # FIND MOST SPECIFIC MATCH
    # ======================================================

    for phrases, answer in GENERAL_KB:

        for phrase in phrases:

            if (
                phrase in q
                and
                len(
                    phrase
                )
                >
                best_match_length
            ):

                best_answer = answer

                best_match_length = (
                    len(
                        phrase
                    )
                )

    if best_answer:

        return (
            best_answer
            .strip()
        )

    # ======================================================
    # FLEXIBLE POLICY TYPE QUESTION
    # ======================================================

    policy_type_words = [

        "type",

        "kind",

        "category",

        "different",
    ]

    policy_words = [

        "policy",

        "insurance",
    ]

    if (
        any(
            word in q
            for word in policy_type_words
        )
        and
        any(
            word in q
            for word in policy_words
        )
    ):

        return (
            POLICY_TYPES_ANSWER
            .strip()
        )

    return None


# ==========================================================
# GENERAL NODE
# ==========================================================

def general_node(
    state: CopilotState,
):

    question = state[
        "question"
    ]

    # ======================================================
    # LOCAL KNOWLEDGE FIRST
    # ======================================================

    local_answer = (
        local_general_answer(
            question
        )
    )

    if local_answer:

        return {

            "result": {

                "answer":
                    local_answer,

                "source":
                    "local_insurance_knowledge",
            }
        }

    # ======================================================
    # LOCAL LLM FALLBACK
    # ======================================================

    prompt = f"""
You are INSURE AI, a professional Insurance AI Copilot.

Answer the user's general insurance question clearly,
accurately and practically.

ANSWERING STYLE:

- Start with a direct answer.
- Use simple language.
- Use bullets where useful.
- Keep the answer focused.
- Give practical explanations where useful.

IMPORTANT RULES:

1. Do not invent company-specific numbers.

2. Do not invent database values.

3. Do not pretend to know internal documents.

4. Do not invent ML probabilities.

5. Fraud-risk output is an investigation signal,
   not proof that fraud occurred.

6. Underwriting-model output requires qualified
   human review.

7. Do not invent operational timelines.

8. Do not promise claim approval, payment,
   underwriting approval or policy issuance.

USER QUESTION:

{question}
"""

    try:

        response = llm.invoke(
            prompt
        )

        answer = (
            response.content
            .strip()
        )

        return {

            "result": {

                "answer":
                    answer,

                "source":
                    "general_ai",
            }
        }

    except Exception as error:

        if is_rate_limit_error(
            error
        ):

            return {

                "result": {

                    "answer": (
                        "This question is not yet available "
                        "in my local insurance knowledge base, "
                        "and the AI service has reached its "
                        "request limit. Please try again later."
                    ),

                    "source":
                        "rate_limit_fallback",
                }
            }

        return {

            "result": {

                "answer": (
                    "I could not complete this question "
                    "because the AI service returned an error."
                ),

                "error":
                    str(
                        error
                    ),
            }
        }


# ==========================================================
# ROUTE SELECTOR
# ==========================================================

def choose_route(
    state: CopilotState,
):

    return state[
        "route"
    ]


# ==========================================================
# BUILD LANGGRAPH
# ==========================================================

builder = StateGraph(
    CopilotState
)


# ==========================================================
# ADD NODES
# ==========================================================

builder.add_node(
    "router",
    router_node,
)

builder.add_node(
    "sql_agent",
    sql_node,
)

builder.add_node(
    "ml_agent",
    ml_node,
)

builder.add_node(
    "rag_agent",
    rag_node,
)

builder.add_node(
    "general_agent",
    general_node,
)


# ==========================================================
# START -> ROUTER
# ==========================================================

builder.add_edge(
    START,
    "router",
)


# ==========================================================
# CONDITIONAL ROUTING
# ==========================================================

builder.add_conditional_edges(

    "router",

    choose_route,

    {

        "sql":
            "sql_agent",

        "ml":
            "ml_agent",

        "rag":
            "rag_agent",

        "general":
            "general_agent",
    },
)


# ==========================================================
# AGENTS -> END
# ==========================================================

builder.add_edge(
    "sql_agent",
    END,
)

builder.add_edge(
    "ml_agent",
    END,
)

builder.add_edge(
    "rag_agent",
    END,
)

builder.add_edge(
    "general_agent",
    END,
)


# ==========================================================
# COMPILE GRAPH
# ==========================================================

copilot_graph = (
    builder.compile()
)


# ==========================================================
# MAIN COPILOT FUNCTION
# ==========================================================

def ask_copilot(
    question: str,
    input_data: dict | None = None,
    session_id: str | None = None,
):
    """
    Main Copilot entry point.

    Backward compatible:

        ask_copilot(
            question
        )

        ask_copilot(
            question,
            input_data
        )

    Conversational:

        ask_copilot(
            question,
            session_id="session_123"
        )
    """

    # ======================================================
    # VALIDATE QUESTION
    # ======================================================

    if (
        not question
        or
        not question.strip()
    ):

        raise ValueError(
            "question cannot be empty."
        )

    original_question = (
        question.strip()
    )

    # ======================================================
    # NORMALIZE SESSION
    # ======================================================

    if isinstance(
        session_id,
        str,
    ):

        session_id = (
            session_id.strip()
            or None
        )

    # ======================================================
    # DEFAULT VALUES
    # ======================================================

    resolved_question = (
        original_question
    )

    follow_up_detected = False

    was_rewritten = False

    context_topic = None

    previous_route = None

    # ======================================================
    # CONVERSATIONAL CONTEXT
    # ======================================================

    if session_id:

        # IMPORTANT:
        #
        # Resolve BEFORE storing the latest message.
        #
        # This ensures the context manager sees
        # only previous conversation.

        previous_route = (
            get_session_route(
                session_id
            )
        )

        resolution = (
            resolve_question(

                session_id=
                    session_id,

                question=
                    original_question,
            )
        )

        resolved_question = (

            resolution.get(
                "resolved_question",
                original_question,
            )

            or

            original_question
        )

        follow_up_detected = (
            bool(
                resolution.get(
                    "follow_up_detected",
                    False,
                )
            )
        )

        was_rewritten = (
            bool(
                resolution.get(
                    "was_rewritten",
                    False,
                )
            )
        )

        context_topic = (
            resolution.get(
                "context_topic"
            )
        )

    # ======================================================
    # BUILD GRAPH STATE
    # ======================================================

    initial_state: CopilotState = {

        "question":
            resolved_question,

        "original_question":
            original_question,

        "resolved_question":
            resolved_question,

        "input_data":
            input_data or {},

        "follow_up_detected":
            follow_up_detected,

        "was_rewritten":
            was_rewritten,
    }

    if session_id:

        initial_state[
            "session_id"
        ] = session_id

    if context_topic:

        initial_state[
            "context_topic"
        ] = context_topic

    if previous_route:

        initial_state[
            "previous_route"
        ] = previous_route

    # ======================================================
    # RUN LANGGRAPH
    # ======================================================

    final_state = (
        copilot_graph.invoke(
            initial_state
        )
    )

    route = (
        final_state.get(
            "route"
        )
    )

    result = (
        final_state.get(
            "result"
        )
    )

    route_inherited = (
        bool(
            final_state.get(
                "route_inherited",
                False,
            )
        )
    )

    # ======================================================
    # STORE CONVERSATION
    # ======================================================

    if session_id:

        # --------------------------------------------------
        # STORE ORIGINAL USER MESSAGE
        # --------------------------------------------------

        session_memory.add_user_message(

            session_id=
                session_id,

            content=
                original_question,
        )

        # --------------------------------------------------
        # STORE ASSISTANT RESPONSE
        # --------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            assistant_answer = (
                result.get(
                    "answer"
                )
            )

            if (
                isinstance(
                    assistant_answer,
                    str,
                )
                and
                assistant_answer.strip()
            ):

                session_memory.add_assistant_message(

                    session_id=
                        session_id,

                    content=
                        assistant_answer.strip(),

                    route=
                        route,
                )

        # --------------------------------------------------
        # REMEMBER ROUTE IN POSTGRESQL
        # --------------------------------------------------

        set_session_route(

            session_id=
                session_id,

            route=
                route,
        )

    # ======================================================
    # RESPONSE
    # ======================================================

    return {

        "route":
            route,

        "result":
            result,

        "session_id":
            session_id,

        "original_question":
            original_question,

        "resolved_question":
            resolved_question,

        "follow_up_detected":
            follow_up_detected,

        "was_rewritten":
            was_rewritten,

        "context_topic":
            context_topic,

        "previous_route":
            previous_route,

        "route_inherited":
            route_inherited,
    }


# ==========================================================
# QUICK ROUTER TEST
# ==========================================================

if __name__ == "__main__":

    test_questions = [

        # GENERAL
        "What is insurance?",

        "What is premium?",

        "What is underwriting?",

        "What is claim?",

        "What is insurance fraud?",

        # SQL
        "How many customers do we have?",

        "Which state has the most customers?",

        "Which policy type has the highest premium?",

        "How many claims are there?",

        "Show top 5 agents.",

        # RAG
        (
            "What documents are required "
            "for an insurance claim?"
        ),

        (
            "According to our claim guide, "
            "what documents are required?"
        ),

        (
            "What information may be reviewed "
            "before policy renewal?"
        ),

        (
            "What factors may be considered "
            "during insurance underwriting?"
        ),

        (
            "What are fraud risk "
            "investigation signals?"
        ),

        (
            "What services can a "
            "policyholder request?"
        ),

        # ML
        "Will this customer renew?",

        "Is this claim suspicious?",

        (
            "Assess this applicant "
            "for underwriting."
        ),
    ]

    print(
        "\n"
        "INSURE AI ROUTER TEST"
        "\n"
    )

    for number, question in enumerate(
        test_questions,
        start=1,
    ):

        route = detect_route(
            question
        )

        print(
            f"{number:02d}. "
            f"[{route.upper():7}] "
            f"{question}"
        )