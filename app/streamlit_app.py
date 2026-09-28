import uuid

import pandas as pd
import requests
import streamlit as st


# ==========================================================
# CONFIGURATION
# ==========================================================

API_BASE_URL = "http://127.0.0.1:8001"


# ==========================================================
# MODEL OPTIONS
# ==========================================================

GENDER_OPTIONS = [
    "Other",
    "Female",
    "Male",
]

MARITAL_STATUS_OPTIONS = [
    "Married",
    "Single",
    "Divorced",
    "Widowed",
]

OCCUPATION_OPTIONS = [
    "Retired",
    "Salaried",
    "Self Employed",
    "Business Owner",
    "Government Employee",
    "Professional",
]

STATE_OPTIONS = [
    "Uttar Pradesh",
    "West Bengal",
    "Odisha",
    "Rajasthan",
    "Tamil Nadu",
    "Gujarat",
    "Karnataka",
    "Maharashtra",
    "Kerala",
    "Telangana",
    "Delhi",
]

CUSTOMER_RISK_OPTIONS = [
    "Medium",
    "Low",
    "High",
]

POLICY_TYPE_OPTIONS = [
    "Motor",
    "Term Life",
    "Commercial",
    "Home",
    "Whole Life",
    "Health",
    "Travel",
    "Personal Accident",
]

SALES_CHANNEL_OPTIONS = [
    "Corporate Partner",
    "Agent",
    "Bancassurance",
    "Broker",
    "Online",
    "Direct",
]

PAYMENT_MODE_OPTIONS = [
    "Annual",
    "Monthly",
    "Quarterly",
    "Half-Yearly",
]

RISK_BAND_OPTIONS = [
    "Low",
    "Medium",
    "High",
]

CLAIM_TYPE_OPTIONS = [
    "Vehicle Damage",
    "Theft",
    "Death",
    "Liability",
    "Travel Disruption",
    "Property Damage",
    "Hospitalization",
    "Accident",
]

CLAIM_SOURCE_OPTIONS = [
    "Surveyor",
    "Garage",
    "Hospital",
    "Police",
    "Third Party",
    "Customer",
]

CLAIM_SEVERITY_OPTIONS = [
    "Medium",
    "Low",
    "High",
]

LIFESTYLE_OPTIONS = [
    "Good",
    "Excellent",
    "Average",
    "Poor",
]

YES_NO_OPTIONS = [
    "No",
    "Yes",
]

OCCUPATION_RISK_OPTIONS = [
    "Medium",
    "Low",
    "High",
]


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="INSURE AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# API HELPERS
# ==========================================================

def backend_is_online():

    try:

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def call_api(
    endpoint,
    payload,
):

    try:

        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=90,
        )

        response.raise_for_status()

        return response.json()

    except requests.HTTPError as error:

        try:

            detail = error.response.json()

        except Exception:

            detail = str(error)

        st.error(
            f"API Error: {detail}"
        )

        return None

    except requests.RequestException as error:

        st.error(
            f"Unable to connect to FastAPI: {error}"
        )

        return None


def call_get_api(
    endpoint,
    params=None,
):

    try:

        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        st.error(
            f"Monitoring API Error: {error}"
        )

        return None


def clear_copilot_session(
    session_id,
):

    if not session_id:

        return False

    try:

        response = requests.delete(
            (
                f"{API_BASE_URL}"
                f"/ask/copilot/session/"
                f"{session_id}"
            ),
            timeout=15,
        )

        response.raise_for_status()

        return True

    except requests.RequestException:

        return False


# ==========================================================
# GENERIC HELPERS
# ==========================================================

def safe_float(
    value,
    default=0.0,
):

    try:

        if value is None:

            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


# ==========================================================
# EXPLAINABILITY
# ==========================================================

def show_explainability(
    result,
):

    explainability = (
        result.get(
            "explainability"
        )
        or
        {}
    )


    if (
        explainability.get(
            "status"
        )
        ==
        "unavailable"
    ):

        st.warning(
            (
                "Prediction completed, but "
                "explainability is unavailable."
            )
        )


        if explainability.get(
            "error"
        ):

            with st.expander(
                "Explainability Error"
            ):

                st.code(
                    str(
                        explainability.get(
                            "error"
                        )
                    )
                )

        return


    factors = (
        explainability.get(
            "top_factors"
        )
        or
        explainability.get(
            "top_features"
        )
        or
        []
    )


    if not factors:

        return


    st.subheader(
        "Why the model predicted this"
    )


    rows = []


    for factor in factors:

        if not isinstance(
            factor,
            dict,
        ):

            continue


        effect = str(
            factor.get(
                "effect",
                "",
            )
        )


        if (
            "support"
            in effect.lower()
        ):

            effect_text = (
                "Supports prediction"
            )

        elif (
            "oppose"
            in effect.lower()
        ):

            effect_text = (
                "Opposes prediction"
            )

        else:

            effect_text = effect


        rows.append(
            {
                "Feature":
                    factor.get(
                        "feature"
                    ),

                "Value":
                    factor.get(
                        "value"
                    ),

                "SHAP Value":
                    factor.get(
                        "shap_value"
                    ),

                "Impact":
                    factor.get(
                        "absolute_impact"
                    ),

                "Effect":
                    effect_text,
            }
        )


    if rows:

        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
        )


    if explainability.get(
        "note"
    ):

        st.caption(
            explainability.get(
                "note"
            )
        )


# ==========================================================
# MONITORING METADATA
# ==========================================================

def show_monitoring(
    result,
):

    monitoring = (
        result.get(
            "monitoring"
        )
        or
        {}
    )


    if not monitoring:

        return


    st.subheader(
        "Production Metadata"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    c1.metric(
        "Prediction ID",
        monitoring.get(
            "prediction_id",
            "—",
        ),
    )


    c2.metric(
        "Model Version",
        monitoring.get(
            "model_version",
            "—",
        ),
    )


    latency = monitoring.get(
        "latency_ms"
    )


    if latency is None:

        latency_text = "—"

    else:

        latency_text = (
            f"{safe_float(latency):.0f} ms"
        )


    c3.metric(
        "Latency",
        latency_text,
    )


    c4.metric(
        "Status",
        monitoring.get(
            "status",
            "Logged",
        ),
    )


# ==========================================================
# DATAFRAME HELPERS
# ==========================================================

def dataframe_from_api(
    data,
):

    if data is None:

        return pd.DataFrame()


    if isinstance(
        data,
        list,
    ):

        return pd.DataFrame(
            data
        )


    if isinstance(
        data,
        dict,
    ):

        for key in [
            "data",
            "results",
            "predictions",
            "items",
            "models",
        ]:

            value = data.get(
                key
            )


            if isinstance(
                value,
                list,
            ):

                return pd.DataFrame(
                    value
                )


        if (
            data
            and
            all(
                isinstance(
                    value,
                    dict,
                )
                for value
                in data.values()
            )
        ):

            rows = []


            for (
                key,
                value,
            ) in data.items():

                row = {
                    "name":
                        key
                }


                row.update(
                    value
                )


                rows.append(
                    row
                )


            return pd.DataFrame(
                rows
            )


        return pd.DataFrame(
            [data]
        )


    return pd.DataFrame()


def performance_items(
    data,
):

    if not data:

        return []


    if isinstance(
        data,
        list,
    ):

        return [
            (
                item.get(
                    "model_name",
                    "Model",
                ),
                item,
            )
            for item in data
            if isinstance(
                item,
                dict,
            )
        ]


    if isinstance(
        data,
        dict,
    ):

        if (
            "model_name"
            in data
            and
            "status"
            in data
        ):

            return [
                (
                    data.get(
                        "model_name",
                        "Model",
                    ),
                    data,
                )
            ]


        if isinstance(
            data.get(
                "models"
            ),
            dict,
        ):

            return list(
                data[
                    "models"
                ].items()
            )


        if isinstance(
            data.get(
                "results"
            ),
            list,
        ):

            return [
                (
                    item.get(
                        "model_name",
                        "Model",
                    ),
                    item,
                )
                for item
                in data[
                    "results"
                ]
                if isinstance(
                    item,
                    dict,
                )
            ]


        return [
            (
                key,
                value,
            )
            for (
                key,
                value,
            ) in data.items()
            if isinstance(
                value,
                dict,
            )
        ]


    return []


# ==========================================================
# COPILOT HELPERS
# ==========================================================

def get_copilot_answer(
    response,
):

    if not response:

        return (
            "No response returned."
        )


    result = response.get(
        "result"
    )


    if isinstance(
        result,
        dict,
    ):

        answer = (
            result.get(
                "answer"
            )
            or
            result.get(
                "response"
            )
            or
            result.get(
                "result"
            )
        )


        if answer:

            return str(
                answer
            )


    if isinstance(
        result,
        str,
    ):

        return result


    return str(
        response.get(
            "answer"
        )
        or
        response.get(
            "response"
        )
        or
        response
    )


def get_sources(
    response,
):

    if not response:

        return []


    result = response.get(
        "result"
    )


    if isinstance(
        result,
        dict,
    ):

        return (
            result.get(
                "sources"
            )
            or
            result.get(
                "citations"
            )
            or
            []
        )


    return (
        response.get(
            "sources"
        )
        or
        []
    )


def show_sources(
    sources,
):

    if not sources:

        return


    with st.expander(
        "Knowledge Sources"
    ):

        for source in sources:

            if isinstance(
                source,
                str,
            ):

                st.write(
                    source
                )

                continue


            source_name = (
                source.get(
                    "source",
                    "Unknown",
                )
            )


            page_number = (
                source.get(
                    "page"
                )
            )


            if page_number is not None:

                st.write(
                    (
                        f"📄 {source_name} "
                        f"— Page {page_number}"
                    )
                )

            else:

                st.write(
                    f"📄 {source_name}"
                )


# ==========================================================
# SESSION STATE
# ==========================================================

if (
    "chat_history"
    not in st.session_state
):

    st.session_state.chat_history = []


if (
    "copilot_session_id"
    not in st.session_state
):

    st.session_state.copilot_session_id = (
        uuid.uuid4().hex
    )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title(
        "🛡️ INSURE AI"
    )


    st.caption(
        "INSURANCE INTELLIGENCE"
    )


    st.divider()


    page = st.radio(
        "Navigation",
        [
            "Home",
            "AI Copilot",
            "Renewal Intelligence",
            "Fraud Intelligence",
            "Underwriting",
            "Production Monitoring",
        ],
    )


    st.divider()


    if backend_is_online():

        st.success(
            "● AI SYSTEM ONLINE"
        )

    else:

        st.error(
            "● SERVICE OFFLINE"
        )


    with st.expander(
        "System Architecture"
    ):

        st.write(
            "FastAPI"
        )

        st.write(
            "PostgreSQL"
        )

        st.write(
            "XGBoost"
        )

        st.write(
            "SHAP Explainability"
        )

        st.write(
            "LangGraph Agents"
        )

        st.write(
            "RAG Knowledge Search"
        )

        st.write(
            "Production Monitoring"
        )


# ==========================================================
# HOME
# ==========================================================

if page == "Home":

    st.title(
        "INSURE AI"
    )


    st.subheader(
        "AI-Powered Insurance Decision Intelligence"
    )


    st.write(
        """
        A single workspace for insurance analytics,
        renewal prediction, fraud-risk detection,
        underwriting decision support, AI Copilot,
        explainability and production monitoring.
        """
    )


    st.divider()


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        with st.container(
            border=True
        ):

            st.subheader(
                "🤖 AI Copilot"
            )

            st.write(
                (
                    "Ask insurance, SQL, "
                    "analytics and ML questions."
                )
            )


    with c2:

        with st.container(
            border=True
        ):

            st.subheader(
                "🔄 Renewal"
            )

            st.write(
                (
                    "Enter customer and policy "
                    "features and analyze renewal risk."
                )
            )


    with c3:

        with st.container(
            border=True
        ):

            st.subheader(
                "🚨 Fraud"
            )

            st.write(
                (
                    "Enter claim details and generate "
                    "a fraud-risk signal."
                )
            )


    with c4:

        with st.container(
            border=True
        ):

            st.subheader(
                "📋 Underwriting"
            )

            st.write(
                (
                    "Enter applicant features and "
                    "generate decision support."
                )
            )


    st.divider()


    st.subheader(
        "Platform Flow"
    )


    st.code(
        """
Manual Business Inputs
        ↓
Feature Validation
        ↓
ML Model
        ↓
Prediction / Probability
        ↓
SHAP Explainability
        ↓
Prediction Monitoring
        """,
        language="text",
    )


# ==========================================================
# AI COPILOT
# ==========================================================

elif page == "AI Copilot":

    st.title(
        "🤖 Insurance AI Copilot"
    )


    st.caption(
        (
            "Ask questions about insurance, "
            "SQL, analytics, models and project knowledge."
        )
    )


    _, button_col = (
        st.columns(
            [5, 1]
        )
    )


    with button_col:

        if st.button(
            "New Chat",
            use_container_width=True,
        ):

            clear_copilot_session(
                st.session_state
                .copilot_session_id
            )


            st.session_state.chat_history = []


            st.session_state.copilot_session_id = (
                uuid.uuid4().hex
            )


            st.rerun()


    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message[
                "role"
            ]
        ):

            st.markdown(
                message[
                    "content"
                ]
            )


            show_sources(
                message.get(
                    "sources"
                )
            )


    question = st.chat_input(
        (
            "Ask about insurance, SQL, "
            "models or analytics..."
        )
    )


    if question:

        st.session_state.chat_history.append(
            {
                "role":
                    "user",

                "content":
                    question,
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        payload = {
            "question":
                question,

            "input_data":
                None,

            "session_id":
                st.session_state
                .copilot_session_id,
        }


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                response = call_api(
                    "/ask/copilot",
                    payload,
                )


            if response:

                answer = (
                    get_copilot_answer(
                        response
                    )
                )


                sources = (
                    get_sources(
                        response
                    )
                )


                st.markdown(
                    answer
                )


                show_sources(
                    sources
                )


                st.session_state.chat_history.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            answer,

                        "sources":
                            sources,
                    }
                )


# ==========================================================
# RENEWAL INTELLIGENCE
# ==========================================================

elif page == "Renewal Intelligence":

    st.title(
        "🔄 Renewal Intelligence"
    )


    st.caption(
        (
            "Enter customer, policy, payment and claim "
            "features to estimate renewal behavior."
        )
    )


    with st.form(
        "renewal_form"
    ):

        st.subheader(
            "Financial & Behavioral Profile"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
            )


            income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=600000.0,
            )


            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=750,
            )


            sum_insured = st.number_input(
                "Sum Insured",
                min_value=0.0,
                value=500000.0,
            )


            annual_premium = st.number_input(
                "Annual Premium",
                min_value=0.0,
                value=20000.0,
            )


        with c2:

            risk_score = st.number_input(
                "Risk Score",
                min_value=0.0,
                value=50.0,
            )


            premium_increase = st.number_input(
                "Premium Increase %",
                value=5.0,
            )


            tenure = st.number_input(
                "Customer Tenure Years",
                min_value=0.0,
                value=5.0,
            )


            total_payments = st.number_input(
                "Total Payments",
                min_value=0.0,
                value=12.0,
            )


            avg_delay = st.number_input(
                "Average Payment Delay",
                value=2.0,
            )


        with c3:

            max_delay = st.number_input(
                "Maximum Payment Delay",
                value=5.0,
            )


            total_claims = st.number_input(
                "Total Claims",
                min_value=0.0,
                value=1.0,
            )


            total_claim_amount = (
                st.number_input(
                    "Total Claim Amount",
                    min_value=0.0,
                    value=10000.0,
                )
            )


            avg_claim_amount = (
                st.number_input(
                    "Average Claim Amount",
                    min_value=0.0,
                    value=10000.0,
                )
            )


            max_claim_amount = (
                st.number_input(
                    "Maximum Claim Amount",
                    min_value=0.0,
                    value=10000.0,
                )
            )


        st.subheader(
            "Customer & Policy Profile"
        )


        c4, c5, c6 = (
            st.columns(3)
        )


        with c4:

            gender = st.selectbox(
                "Gender",
                GENDER_OPTIONS,
                index=2,
            )


            marital_status = (
                st.selectbox(
                    "Marital Status",
                    MARITAL_STATUS_OPTIONS,
                    index=0,
                )
            )


            occupation = st.selectbox(
                "Occupation",
                OCCUPATION_OPTIONS,
                index=1,
            )


        with c5:

            state = st.selectbox(
                "State",
                STATE_OPTIONS,
                index=6,
            )


            customer_risk = (
                st.selectbox(
                    "Customer Risk Segment",
                    CUSTOMER_RISK_OPTIONS,
                    index=1,
                )
            )


            policy_type = (
                st.selectbox(
                    "Policy Type",
                    POLICY_TYPE_OPTIONS,
                    index=5,
                )
            )


        with c6:

            sales_channel = (
                st.selectbox(
                    "Sales Channel",
                    SALES_CHANNEL_OPTIONS,
                    index=1,
                )
            )


            payment_mode = (
                st.selectbox(
                    "Payment Mode",
                    PAYMENT_MODE_OPTIONS,
                    index=0,
                )
            )


            risk_band = st.selectbox(
                "Risk Band",
                RISK_BAND_OPTIONS,
                index=0,
            )


        renewal_submit = (
            st.form_submit_button(
                "Analyze Renewal Risk",
                type="primary",
                use_container_width=True,
            )
        )


    if renewal_submit:

        payload = {
            "AGE":
                age,

            "ANNUAL_INCOME":
                income,

            "CREDIT_SCORE":
                credit_score,

            "SUM_INSURED":
                sum_insured,

            "ANNUAL_PREMIUM":
                annual_premium,

            "RISK_SCORE":
                risk_score,

            "PREMIUM_INCREASE_PCT":
                premium_increase,

            "CUSTOMER_TENURE_YEARS":
                tenure,

            "TOTAL_PAYMENTS":
                total_payments,

            "AVG_PAYMENT_DELAY":
                avg_delay,

            "MAX_PAYMENT_DELAY":
                max_delay,

            "TOTAL_CLAIMS":
                total_claims,

            "TOTAL_CLAIM_AMOUNT":
                total_claim_amount,

            "AVG_CLAIM_AMOUNT":
                avg_claim_amount,

            "MAX_CLAIM_AMOUNT":
                max_claim_amount,

            "GENDER":
                gender,

            "MARITAL_STATUS":
                marital_status,

            "OCCUPATION":
                occupation,

            "STATE":
                state,

            "CUSTOMER_RISK_SEGMENT":
                customer_risk,

            "POLICY_TYPE":
                policy_type,

            "SALES_CHANNEL":
                sales_channel,

            "PAYMENT_MODE":
                payment_mode,

            "RISK_BAND":
                risk_band,
        }


        with st.spinner(
            "Analyzing renewal behavior..."
        ):

            result = call_api(
                "/predict/renewal",
                payload,
            )


        if result:

            st.divider()


            renewal_probability = (
                safe_float(
                    result.get(
                        "renewal_probability"
                    )
                )
            )


            churn_probability = (
                safe_float(
                    result.get(
                        "churn_probability"
                    )
                )
            )


            threshold = (
                safe_float(
                    result.get(
                        "threshold"
                    )
                )
            )


            prediction = (
                result.get(
                    "prediction",
                    "Unavailable",
                )
            )


            m1, m2, m3 = (
                st.columns(3)
            )


            m1.metric(
                "Renewal Probability",
                (
                    f"{renewal_probability * 100:.1f}%"
                ),
            )


            m2.metric(
                "Churn Probability",
                (
                    f"{churn_probability * 100:.1f}%"
                ),
            )


            m3.metric(
                "Decision Threshold",
                (
                    f"{threshold * 100:.1f}%"
                ),
            )


            st.subheader(
                "Renewal Confidence"
            )


            st.progress(
                min(
                    max(
                        renewal_probability,
                        0.0,
                    ),
                    1.0,
                )
            )


            if (
                churn_probability
                >=
                threshold
            ):

                st.error(
                    f"⚠ {prediction}"
                )

            else:

                st.success(
                    f"✓ {prediction}"
                )


            show_explainability(
                result
            )


            show_monitoring(
                result
            )


            with st.expander(
                "Technical Model Output"
            ):

                st.json(
                    result
                )


# ==========================================================
# FRAUD INTELLIGENCE
# ==========================================================

elif page == "Fraud Intelligence":

    st.title(
        "🚨 Fraud Intelligence"
    )


    st.caption(
        (
            "Generate a fraud-risk signal to prioritize "
            "claims for additional investigation."
        )
    )


    st.warning(
        (
            "A risk flag is not a fraud determination. "
            "Human investigation is required."
        )
    )


    with st.form(
        "fraud_form"
    ):

        st.subheader(
            "Claim Profile"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            claim_amount = (
                st.number_input(
                    "Claim Amount",
                    min_value=0.0,
                    value=50000.0,
                )
            )


            reporting_delay = (
                st.number_input(
                    "Reporting Delay Days",
                    min_value=0.0,
                    value=2.0,
                )
            )


            incident_month = (
                st.number_input(
                    "Incident Month",
                    min_value=1,
                    max_value=12,
                    value=6,
                )
            )


        with c2:

            fraud_age = (
                st.number_input(
                    "Customer Age",
                    min_value=18,
                    max_value=100,
                    value=35,
                )
            )


            fraud_income = (
                st.number_input(
                    "Annual Income",
                    min_value=0.0,
                    value=600000.0,
                )
            )


            fraud_credit = (
                st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=900,
                    value=750,
                )
            )


        with c3:

            fraud_sum_insured = (
                st.number_input(
                    "Sum Insured",
                    min_value=0.0,
                    value=500000.0,
                )
            )


            fraud_premium = (
                st.number_input(
                    "Annual Premium",
                    min_value=0.0,
                    value=20000.0,
                )
            )


            fraud_risk_score = (
                st.number_input(
                    "Risk Score",
                    min_value=0.0,
                    value=50.0,
                )
            )


        st.subheader(
            "Claim & Customer Details"
        )


        c4, c5, c6 = (
            st.columns(3)
        )


        with c4:

            claim_type = (
                st.selectbox(
                    "Claim Type",
                    CLAIM_TYPE_OPTIONS,
                    index=6,
                )
            )


            claim_source = (
                st.selectbox(
                    "Claim Source",
                    CLAIM_SOURCE_OPTIONS,
                    index=5,
                )
            )


            claim_severity = (
                st.selectbox(
                    "Claim Severity",
                    CLAIM_SEVERITY_OPTIONS,
                    index=0,
                )
            )


            fraud_gender = (
                st.selectbox(
                    "Gender",
                    GENDER_OPTIONS,
                    index=2,
                )
            )


        with c5:

            fraud_marital = (
                st.selectbox(
                    "Marital Status",
                    MARITAL_STATUS_OPTIONS,
                    index=0,
                )
            )


            fraud_occupation = (
                st.selectbox(
                    "Occupation",
                    OCCUPATION_OPTIONS,
                    index=1,
                )
            )


            fraud_state = (
                st.selectbox(
                    "State",
                    STATE_OPTIONS,
                    index=6,
                )
            )


            fraud_customer_risk = (
                st.selectbox(
                    "Customer Risk Segment",
                    CUSTOMER_RISK_OPTIONS,
                    index=1,
                )
            )


        with c6:

            fraud_policy_type = (
                st.selectbox(
                    "Policy Type",
                    POLICY_TYPE_OPTIONS,
                    index=5,
                )
            )


            fraud_payment_mode = (
                st.selectbox(
                    "Payment Mode",
                    PAYMENT_MODE_OPTIONS,
                    index=0,
                )
            )


            fraud_risk_band = (
                st.selectbox(
                    "Risk Band",
                    RISK_BAND_OPTIONS,
                    index=0,
                )
            )


        fraud_submit = (
            st.form_submit_button(
                "Analyze Claim Risk",
                type="primary",
                use_container_width=True,
            )
        )


    if fraud_submit:

        payload = {
            "CLAIM_AMOUNT":
                claim_amount,

            "REPORTING_DELAY_DAYS":
                reporting_delay,

            "INCIDENT_MONTH":
                incident_month,

            "AGE":
                fraud_age,

            "ANNUAL_INCOME":
                fraud_income,

            "CREDIT_SCORE":
                fraud_credit,

            "SUM_INSURED":
                fraud_sum_insured,

            "ANNUAL_PREMIUM":
                fraud_premium,

            "RISK_SCORE":
                fraud_risk_score,

            "CLAIM_TYPE":
                claim_type,

            "SOURCE":
                claim_source,

            "CLAIM_SEVERITY":
                claim_severity,

            "GENDER":
                fraud_gender,

            "MARITAL_STATUS":
                fraud_marital,

            "OCCUPATION":
                fraud_occupation,

            "STATE":
                fraud_state,

            "CUSTOMER_RISK_SEGMENT":
                fraud_customer_risk,

            "POLICY_TYPE":
                fraud_policy_type,

            "PAYMENT_MODE":
                fraud_payment_mode,

            "RISK_BAND":
                fraud_risk_band,
        }


        with st.spinner(
            "Analyzing claim risk signals..."
        ):

            result = call_api(
                "/predict/fraud",
                payload,
            )


        if result:

            st.divider()


            fraud_probability = (
                safe_float(
                    result.get(
                        "fraud_probability"
                    )
                )
            )


            threshold = (
                safe_float(
                    result.get(
                        "threshold"
                    )
                )
            )


            prediction = (
                result.get(
                    "prediction",
                    "Unavailable",
                )
            )


            m1, m2 = (
                st.columns(2)
            )


            m1.metric(
                "Fraud Risk Probability",
                (
                    f"{fraud_probability * 100:.1f}%"
                ),
            )


            m2.metric(
                "Investigation Threshold",
                (
                    f"{threshold * 100:.1f}%"
                ),
            )


            st.subheader(
                "Risk Signal"
            )


            st.progress(
                min(
                    max(
                        fraud_probability,
                        0.0,
                    ),
                    1.0,
                )
            )


            if (
                fraud_probability
                >=
                threshold
            ):

                st.error(
                    f"⚠ {prediction}"
                )

            else:

                st.success(
                    f"✓ {prediction}"
                )


            st.caption(
                (
                    "This is a prioritization signal only. "
                    "It does not establish that fraud occurred."
                )
            )


            show_explainability(
                result
            )


            show_monitoring(
                result
            )


            with st.expander(
                "Technical Model Output"
            ):

                st.json(
                    result
                )


# ==========================================================
# UNDERWRITING
# ==========================================================

elif page == "Underwriting":

    st.title(
        "📋 Underwriting Decision Support"
    )


    st.caption(
        (
            "Analyze applicant characteristics and generate "
            "model-based decision support."
        )
    )


    st.warning(
        (
            "Final underwriting decisions require "
            "qualified human review."
        )
    )


    with st.form(
        "underwriting_form"
    ):

        left, right = (
            st.columns(2)
        )


        with left:

            uw_age = st.number_input(
                "Applicant Age",
                min_value=18,
                max_value=100,
                value=35,
            )


            health_score = (
                st.number_input(
                    "Health Score",
                    min_value=0.0,
                    value=75.0,
                )
            )


            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=24.5,
            )


            uw_credit = (
                st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=900,
                    value=750,
                )
            )


        with right:

            lifestyle = (
                st.selectbox(
                    "Lifestyle",
                    LIFESTYLE_OPTIONS,
                    index=0,
                )
            )


            medical_history = (
                st.selectbox(
                    "Medical History",
                    YES_NO_OPTIONS,
                    index=0,
                )
            )


            smoker = st.selectbox(
                "Smoker",
                YES_NO_OPTIONS,
                index=0,
            )


            occupation_risk = (
                st.selectbox(
                    "Occupation Risk",
                    OCCUPATION_RISK_OPTIONS,
                    index=1,
                )
            )


        underwriting_submit = (
            st.form_submit_button(
                "Run Assessment",
                type="primary",
                use_container_width=True,
            )
        )


    if underwriting_submit:

        payload = {
            "AGE":
                uw_age,

            "HEALTH_SCORE":
                health_score,

            "BMI":
                bmi,

            "CREDIT_SCORE":
                uw_credit,

            "LIFESTYLE":
                lifestyle,

            "MEDICAL_HISTORY_FLAG":
                medical_history,

            "SMOKER_FLAG":
                smoker,

            "OCCUPATION_RISK":
                occupation_risk,
        }


        with st.spinner(
            "Evaluating applicant profile..."
        ):

            result = call_api(
                "/predict/underwriting",
                payload,
            )


        if result:

            st.divider()


            decision = (
                result.get(
                    "decision"
                )
                or
                result.get(
                    "prediction"
                )
                or
                result.get(
                    "predicted_class"
                )
                or
                "Unavailable"
            )


            if (
                decision
                ==
                "Declined"
            ):

                st.error(
                    (
                        "Model Recommendation: "
                        f"{decision}"
                    )
                )

            elif (
                decision
                ==
                "Approved with Loading"
            ):

                st.warning(
                    (
                        "Model Recommendation: "
                        f"{decision}"
                    )
                )

            else:

                st.success(
                    (
                        "Model Recommendation: "
                        f"{decision}"
                    )
                )


            probabilities = (
                result.get(
                    "probabilities"
                )
                or
                {}
            )


            if probabilities:

                st.subheader(
                    "Class Probabilities"
                )


                columns = st.columns(
                    len(
                        probabilities
                    )
                )


                for index, (
                    class_name,
                    probability,
                ) in enumerate(
                    probabilities.items()
                ):

                    columns[
                        index
                    ].metric(
                        class_name,
                        (
                            f"{safe_float(probability) * 100:.1f}%"
                        ),
                    )


            show_explainability(
                result
            )


            show_monitoring(
                result
            )


            st.caption(
                (
                    "This model provides decision-support "
                    "signals only and must not be used as "
                    "an autonomous eligibility decision."
                )
            )


            with st.expander(
                "Technical Model Output"
            ):

                st.json(
                    result
                )


# ==========================================================
# PRODUCTION MONITORING
# ==========================================================

elif page == "Production Monitoring":

    st.title(
        "📊 Production Monitoring"
    )


    st.caption(
        (
            "Monitor prediction traffic, latency, "
            "ground-truth coverage and observed "
            "model performance."
        )
    )


    refresh_col, _ = (
        st.columns(
            [1, 5]
        )
    )


    with refresh_col:

        if st.button(
            "Refresh",
            use_container_width=True,
        ):

            st.rerun()


    summary_data = call_get_api(
        "/monitoring/summary"
    )


    model_data = call_get_api(
        "/monitoring/models"
    )


    coverage_data = call_get_api(
        "/monitoring/outcome-coverage"
    )


    performance_data = call_get_api(
        "/monitoring/performance"
    )


    recent_data = call_get_api(
        "/monitoring/recent",
        params={
            "limit":
                20,
        },
    )


    # ======================================================
    # OPERATIONAL SUMMARY
    # ======================================================

    st.subheader(
        "Operational Summary"
    )


    if isinstance(
        summary_data,
        dict,
    ):

        total = (
            summary_data.get(
                "total_predictions"
            )
            or
            summary_data.get(
                "total"
            )
            or
            0
        )


        successful = (
            summary_data.get(
                "successful_predictions"
            )
            or
            summary_data.get(
                "successful"
            )
            or
            summary_data.get(
                "success_count"
            )
            or
            0
        )


        failed = (
            summary_data.get(
                "failed_predictions"
            )
            or
            summary_data.get(
                "failed"
            )
            or
            summary_data.get(
                "error_count"
            )
            or
            0
        )


        avg_latency = (
            summary_data.get(
                "avg_latency_ms"
            )
            or
            summary_data.get(
                "average_latency_ms"
            )
            or
            0
        )


        c1, c2, c3, c4 = (
            st.columns(4)
        )


        c1.metric(
            "Total Predictions",
            total,
        )


        c2.metric(
            "Successful",
            successful,
        )


        c3.metric(
            "Errors",
            failed,
        )


        c4.metric(
            "Average Latency",
            (
                f"{safe_float(avg_latency):.0f} ms"
            ),
        )


        with st.expander(
            "Raw Summary"
        ):

            st.json(
                summary_data
            )


    # ======================================================
    # MODEL OPERATIONS
    # ======================================================

    st.divider()


    st.subheader(
        "Model-wise Operations"
    )


    model_df = (
        dataframe_from_api(
            model_data
        )
    )


    if not model_df.empty:

        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            (
                "No model operational "
                "data available yet."
            )
        )


    # ======================================================
    # GROUND TRUTH
    # ======================================================

    st.divider()


    st.subheader(
        "Ground Truth Coverage"
    )


    coverage_df = (
        dataframe_from_api(
            coverage_data
        )
    )


    if not coverage_df.empty:

        st.dataframe(
            coverage_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            (
                "No ground-truth coverage "
                "data available yet."
            )
        )


    # ======================================================
    # PERFORMANCE
    # ======================================================

    st.divider()


    st.subheader(
        "Observed Model Performance"
    )


    items = performance_items(
        performance_data
    )


    if not items:

        st.info(
            (
                "No labeled performance "
                "data available yet."
            )
        )


    for (
        model_name,
        model,
    ) in items:

        if not isinstance(
            model,
            dict,
        ):

            continue


        with st.container(
            border=True
        ):

            st.subheader(
                str(
                    model_name
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )


            if (
                model.get(
                    "status"
                )
                !=
                "success"
            ):

                st.info(
                    model.get(
                        "message",
                        (
                            "Not enough labeled "
                            "predictions yet."
                        ),
                    )
                )

                continue


            evaluated = int(
                model.get(
                    "evaluated_predictions",
                    0,
                )
                or
                0
            )


            accuracy = (
                safe_float(
                    model.get(
                        "accuracy"
                    )
                )
            )


            precision = (
                safe_float(
                    model.get(
                        "precision"
                    )
                )
            )


            recall = (
                safe_float(
                    model.get(
                        "recall"
                    )
                )
            )


            f1 = (
                safe_float(
                    model.get(
                        "f1_score"
                    )
                )
            )


            p1, p2, p3, p4, p5 = (
                st.columns(5)
            )


            p1.metric(
                "Evaluated",
                evaluated,
            )


            p2.metric(
                "Accuracy",
                f"{accuracy:.2%}",
            )


            p3.metric(
                "Macro Precision",
                f"{precision:.2%}",
            )


            p4.metric(
                "Macro Recall",
                f"{recall:.2%}",
            )


            p5.metric(
                "Macro F1",
                f"{f1:.2%}",
            )


            # ==============================================
            # CLASS METRICS
            # ==============================================

            class_metrics = (
                model.get(
                    "class_metrics"
                )
                or
                []
            )


            if class_metrics:

                st.write(
                    "Class-wise Metrics"
                )


                st.dataframe(
                    pd.DataFrame(
                        class_metrics
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


            # ==============================================
            # CONFUSION MATRIX
            # ==============================================

            confusion = (
                model.get(
                    "confusion_matrix"
                )
                or
                {}
            )


            labels = (
                confusion.get(
                    "labels"
                )
                or
                []
            )


            matrix = (
                confusion.get(
                    "matrix"
                )
                or
                []
            )


            if labels and matrix:

                st.write(
                    "Confusion Matrix"
                )


                confusion_df = (
                    pd.DataFrame(
                        matrix,

                        index=[
                            f"Actual {label}"
                            for label
                            in labels
                        ],

                        columns=[
                            f"Predicted {label}"
                            for label
                            in labels
                        ],
                    )
                )


                st.dataframe(
                    confusion_df,
                    use_container_width=True,
                )


            # ==============================================
            # ACTUAL VS PREDICTED
            # ==============================================

            actual_distribution = (
                model.get(
                    "actual_distribution"
                )
                or
                {}
            )


            predicted_distribution = (
                model.get(
                    "predicted_distribution"
                )
                or
                {}
            )


            if (
                actual_distribution
                and
                predicted_distribution
            ):

                labels_all = sorted(
                    set(
                        actual_distribution.keys()
                    )
                    |
                    set(
                        predicted_distribution.keys()
                    )
                )


                chart_df = (
                    pd.DataFrame(
                        {
                            "Actual": [
                                actual_distribution.get(
                                    label,
                                    0,
                                )
                                for label
                                in labels_all
                            ],

                            "Predicted": [
                                predicted_distribution.get(
                                    label,
                                    0,
                                )
                                for label
                                in labels_all
                            ],
                        },

                        index=
                            labels_all,
                    )
                )


                st.write(
                    "Actual vs Predicted"
                )


                st.bar_chart(
                    chart_df
                )


            st.caption(
                (
                    "Source: "
                    f"{model.get('evaluation_source', '—')} | "
                    "Endpoint: "
                    f"{model.get('endpoint_filter', '—')} | "
                    "Deduplication: "
                    f"{model.get('deduplication', '—')}"
                )
            )


    # ======================================================
    # RECENT PREDICTIONS
    # ======================================================

    st.divider()


    st.subheader(
        "Recent Predictions"
    )


    recent_df = (
        dataframe_from_api(
            recent_data
        )
    )


    if not recent_df.empty:

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            (
                "No recent prediction "
                "logs available yet."
            )
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()


st.caption(
    (
        "INSURE AI • Explainable AI • "
        "Production Monitoring • Human-in-the-Loop"
    )
)