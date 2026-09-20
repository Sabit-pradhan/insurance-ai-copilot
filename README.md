# Insurance AI Copilot

An end-to-end Insurance AI platform combining:

- Machine Learning
- PostgreSQL
- FastAPI
- LangGraph
- RAG
- Natural Language to SQL
- NVIDIA Nemotron via OpenRouter

The system routes user requests automatically to the appropriate AI agent for database analytics, machine-learning predictions, or insurance document retrieval.

---

## Project Architecture

```text
                         USER
                           |
                           v
                      FastAPI API
                           |
                           v
                   LangGraph Router
               _________|_________
              |         |         |
              v         v         v
         SQL Agent   ML Agent   RAG Agent
              |         |         |
              v         v         v
        PostgreSQL   XGBoost   Documents
                        |
            ____________|____________
           |            |            |
           v            v            v
       Renewal        Fraud     Underwriting
        Model         Model        Model