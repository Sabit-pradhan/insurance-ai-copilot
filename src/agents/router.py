# src/agents/router.py

from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

# Shared NVIDIA/OpenRouter LLM
from src.utils.llm import llm

# Import Agents
from src.agents.sql_agent import ask_sql_agent
from src.agents.ml_agent import ask_ml_agent
from src.agents.rag_agent import ask_rag_agent


# --------------------------------------------------
# Graph State
# --------------------------------------------------

class CopilotState(TypedDict, total=False):

    # User question
    question: str

    # Optional ML input data
    input_data: dict[str, Any]

    # sql / ml / rag / unknown
    route: str

    # Final agent result
    result: dict


# --------------------------------------------------
# Router Node
# --------------------------------------------------

def router_node(state: CopilotState):
    """
    Decide which agent should handle the question.
    """

    question = state["question"]

    prompt = f"""
You are the routing agent for an Insurance AI Copilot.

Choose exactly ONE route:

sql
ml
rag
unknown


ROUTING RULES:

sql:
Use when the user asks questions about existing
database records or analytics.

Examples:
- Show top 5 policy types by premium
- Which state has the most customers?
- How many claims are there?
- Show highest premium customers
- Calculate average claim amount


ml:
Use when the user asks for prediction or risk assessment.

Examples:
- Will this customer renew?
- Is this claim fraudulent?
- Predict churn risk
- Evaluate this applicant for underwriting


rag:
Use when the user asks questions about insurance
documents, policies, procedures, guidelines,
terms, exclusions, claim requirements,
coverage rules, or document-based knowledge.

Examples:
- What documents are required for a claim?
- What does the policy say about exclusions?
- Explain the claim settlement process
- What is covered under this policy?
- What are the policy conditions?


unknown:
Use when the request does not belong to
SQL analytics, ML prediction, or document search.


Return ONLY one word:

sql
ml
rag
unknown


USER QUESTION:

{question}
"""

    response = llm.invoke(prompt)

    route = response.content.strip().lower()


    # --------------------------------------------------
    # Deterministic cleanup
    # --------------------------------------------------

    if "sql" in route:
        route = "sql"

    elif "ml" in route:
        route = "ml"

    elif "rag" in route:
        route = "rag"

    else:
        route = "unknown"


    return {
        "route": route
    }


# --------------------------------------------------
# SQL Agent Node
# --------------------------------------------------

def sql_node(state: CopilotState):

    result = ask_sql_agent(
        state["question"]
    )

    return {
        "result": result
    }


# --------------------------------------------------
# ML Agent Node
# --------------------------------------------------

def ml_node(state: CopilotState):

    result = ask_ml_agent(
        question=state["question"],
        input_data=state.get(
            "input_data",
            {}
        )
    )

    return {
        "result": result
    }


# --------------------------------------------------
# RAG Agent Node
# --------------------------------------------------

def rag_node(state: CopilotState):

    result = ask_rag_agent(
        question=state["question"]
    )

    return {
        "result": result
    }


# --------------------------------------------------
# Unknown Node
# --------------------------------------------------

def unknown_node(state: CopilotState):

    return {
        "result": {
            "error": (
                "This request is not currently supported "
                "by SQL, ML, or RAG agents."
            )
        }
    }


# --------------------------------------------------
# Route Selector
# --------------------------------------------------

def choose_route(state: CopilotState):

    return state["route"]


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

builder = StateGraph(CopilotState)


# Add nodes
builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "sql_agent",
    sql_node
)

builder.add_node(
    "ml_agent",
    ml_node
)

builder.add_node(
    "rag_agent",
    rag_node
)

builder.add_node(
    "unknown",
    unknown_node
)


# START -> Router
builder.add_edge(
    START,
    "router"
)


# Router -> Correct Agent
builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "sql": "sql_agent",
        "ml": "ml_agent",
        "rag": "rag_agent",
        "unknown": "unknown"
    }
)


# Agents -> END
builder.add_edge(
    "sql_agent",
    END
)

builder.add_edge(
    "ml_agent",
    END
)

builder.add_edge(
    "rag_agent",
    END
)

builder.add_edge(
    "unknown",
    END
)


# --------------------------------------------------
# Compile Graph
# --------------------------------------------------

copilot_graph = builder.compile()


# --------------------------------------------------
# Main Copilot Function
# --------------------------------------------------

def ask_copilot(
    question: str,
    input_data: dict | None = None
):

    initial_state = {
        "question": question,
        "input_data": input_data or {}
    }


    final_state = copilot_graph.invoke(
        initial_state
    )


    return {
        "route": final_state.get("route"),
        "result": final_state.get("result")
    }


# --------------------------------------------------
# Quick Test
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "What documents are required "
        "for an insurance claim?"
    )

    result = ask_copilot(
        question=question
    )

    print("\nCopilot Result:")
    print(result)