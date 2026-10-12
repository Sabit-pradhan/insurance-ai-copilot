import uuid
import html

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
# PREMIUM WHITE + PURPLE UI THEME
# ==========================================================

st.markdown("""
<style>
    :root {
        --purple: #6D28D9;
        --purple-dark: #4C1D95;
        --purple-light: #8B5CF6;
        --purple-soft: #F3E8FF;
        --lavender: #FAF7FF;
        --ink: #211A2E;
        --muted: #6B6475;
        --border: #E8DDF7;
        --white: #FFFFFF;
        --green: #16A34A;
        --red: #DC2626;
        --shadow: 0 10px 30px rgba(76, 29, 149, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(139,92,246,.10), transparent 24%),
            radial-gradient(circle at 5% 25%, rgba(109,40,217,.06), transparent 22%),
            #FFFFFF;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,.82);
        backdrop-filter: blur(14px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAF7FF 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 7px;
    }

    [data-testid="stSidebar"] .stRadio label {
        border-radius: 12px;
        padding: 7px 10px;
        transition: all .2s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: var(--purple-soft);
    }

    h1 {
        color: var(--purple-dark) !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
    }

    h2, h3 {
        color: var(--ink) !important;
        font-weight: 750 !important;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #E8DDF7;
        border-radius: 24px;
        background:
            radial-gradient(circle at 90% 10%, rgba(139,92,246,.20), transparent 28%),
            linear-gradient(135deg, #FFFFFF 0%, #FAF7FF 60%, #F3E8FF 100%);
        box-shadow: var(--shadow);
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: var(--purple);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .hero-title {
        color: var(--purple-dark);
        font-size: 34px;
        font-weight: 850;
        line-height: 1.1;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 15px;
        margin-top: 10px;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(76,29,149,.05);
        margin: 8px 0 16px 0;
    }

    .result-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FBF8FF 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 22px;
        box-shadow: var(--shadow);
        height: 100%;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .metric-value {
        color: var(--purple-dark);
        font-size: 30px;
        font-weight: 850;
        margin-top: 3px;
    }

    .metric-small {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }

    .risk-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: .02em;
    }

    .risk-high {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }

    .risk-low {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }

    .purple-badge {
        background: var(--purple-soft);
        color: var(--purple-dark);
        border: 1px solid #DDD0F7;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 750;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 12px !important;
        border: 1px solid #6D28D9 !important;
        background: linear-gradient(135deg, #7C3AED, #5B21B6) !important;
        color: white !important;
        font-weight: 750 !important;
        box-shadow: 0 7px 18px rgba(109,40,217,.20) !important;
        transition: transform .18s ease, box-shadow .18s ease !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(109,40,217,.28) !important;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 15px 16px;
        box-shadow: 0 6px 18px rgba(76,29,149,.05);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--purple-dark) !important;
        font-weight: 800 !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: #DDD0F7 !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: var(--purple-light) !important;
        box-shadow: 0 0 0 2px rgba(139,92,246,.13) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: #FFFFFF;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8B5CF6, #5B21B6) !important;
    }

    .stAlert {
        border-radius: 14px !important;
    }

    .mini-stat {
        text-align: center;
        padding: 15px 10px;
        border-radius: 15px;
        background: #FAF7FF;
        border: 1px solid #E8DDF7;
    }

    .mini-stat strong {
        display: block;
        color: var(--purple-dark);
        font-size: 20px;
    }

    .mini-stat span {
        color: var(--muted);
        font-size: 11px;
    }

    .feature-chip {
        display: inline-block;
        margin: 4px 5px 4px 0;
        padding: 7px 10px;
        border-radius: 10px;
        background: #FAF7FF;
        border: 1px solid #E8DDF7;
        color: #4C1D95;
        font-size: 12px;
        font-weight: 650;
    }

    @keyframes pulsePurple {
        0% { box-shadow: 0 0 0 0 rgba(124,58,237,.25); }
        70% { box-shadow: 0 0 0 10px rgba(124,58,237,0); }
        100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); }
    }

    .live-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        background: #22C55E;
        border-radius: 50%;
        margin-right: 7px;
        animation: pulsePurple 1.8s infinite;
    }


    /* ======================================================
       COPILOT: FORCE DARK, READABLE TEXT ON WHITE CARDS
       ====================================================== */
    [data-testid="stChatMessage"] {
        border: 1px solid #E8DDF7 !important;
        border-radius: 18px !important;
        background: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(76,29,149,.05) !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
        color: #211A2E !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
        color: #4C1D95 !important;
        background: #F3E8FF !important;
        border: 1px solid #E8DDF7 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
        color: #6D28D9 !important;
        font-weight: 700 !important;
    }

    /* ======================================================
       EXPLAINABILITY — SHARED BY ALL 3 MODELS
       ====================================================== */
    .explain-card {
        background: #FFFFFF !important;
        color: #211A2E !important;
        border: 1px solid #E8DDF7 !important;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 10px 28px rgba(76,29,149,.07);
        margin: 18px 0 22px 0;
        overflow: hidden;
    }

    .explain-title {
        color: #4C1D95 !important;
        font-size: 23px;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .explain-subtitle {
        color: #6B6475 !important;
        font-size: 13px;
        line-height: 1.55;
        margin-bottom: 16px;
    }

    .explain-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }

    .explain-pill {
        background: #F3E8FF !important;
        color: #4C1D95 !important;
        border: 1px solid #DDD0F7 !important;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 12px;
        font-weight: 800;
    }

    .base-value-card {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        background: #FAF7FF !important;
        color: #4C1D95 !important;
        border: 1px solid #E8DDF7 !important;
        border-radius: 12px;
        padding: 9px 13px;
        margin-bottom: 15px;
        font-size: 13px;
    }

    .base-value-card strong {
        color: #6D28D9 !important;
        font-size: 16px;
    }

    .explain-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #E8DDF7;
        border-radius: 14px;
        background: #FFFFFF !important;
    }

    .explain-table {
        width: 100%;
        min-width: 760px;
        border-collapse: separate;
        border-spacing: 0;
        background: #FFFFFF !important;
        color: #211A2E !important;
    }

    .explain-table th {
        background: #F3E8FF !important;
        color: #4C1D95 !important;
        text-align: left;
        font-size: 12px;
        font-weight: 850;
        padding: 13px 12px;
        border-bottom: 1px solid #E8DDF7;
        white-space: nowrap;
    }

    .explain-table td {
        background: #FFFFFF !important;
        color: #211A2E !important;
        font-size: 13px;
        padding: 12px;
        border-bottom: 1px solid #F0EAF8;
        vertical-align: middle;
    }

    .explain-table tr:hover td {
        background: #FBF8FF !important;
    }

    .explain-table tr:last-child td {
        border-bottom: none;
    }

    .feature-cell {
        color: #4C1D95 !important;
    }

    .shap-cell {
        font-variant-numeric: tabular-nums;
        font-weight: 850;
    }

    .shap-positive {
        color: #B91C1C !important;
    }

    .shap-negative {
        color: #15803D !important;
    }

    .effect-support {
        color: #B91C1C !important;
        background: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 11px;
        font-weight: 850;
        display: inline-block;
        white-space: nowrap;
    }

    .effect-oppose {
        color: #15803D !important;
        background: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 11px;
        font-weight: 850;
        display: inline-block;
        white-space: nowrap;
    }

    .effect-neutral {
        color: #6B6475 !important;
        background: #F9FAFB !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 11px;
        font-weight: 750;
        display: inline-block;
        white-space: nowrap;
    }

    .explain-note {
        margin-top: 14px;
        color: #6B6475 !important;
        font-size: 12px;
        line-height: 1.55;
    }

    .empty-explain {
        background: #FAF7FF !important;
        color: #6B6475 !important;
        border: 1px dashed #D8C8F1 !important;
        border-radius: 12px;
        padding: 14px;
    }

    .explain-warning {
        border-color: #F0D9A6 !important;
        background: linear-gradient(135deg, #FFFFFF, #FFFBF2) !important;
    }

    /* Streamlit dataframe fallback: never let model explanation tables go dark */
    [data-testid="stDataFrame"] {
        border: 1px solid #E8DDF7 !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    /* ======================================================
       COPILOT — FORCE READABLE DARK TEXT
       ====================================================== */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        color: #211A2E !important;
        border: 1px solid #E8DDF7 !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(76,29,149,.06) !important;
    }

    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] blockquote {
        color: #211A2E !important;
    }

    [data-testid="stChatMessage"] code,
    [data-testid="stChatMessage"] pre {
        color: #4C1D95 !important;
        background: #F3E8FF !important;
        border: 1px solid #E8DDF7 !important;
    }

    [data-testid="stChatMessage"] a {
        color: #6D28D9 !important;
        font-weight: 750 !important;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


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
    model_name="Model",
):
    """
    Universal explainability renderer for Renewal, Fraud and Underwriting.

    IMPORTANT:
    This version intentionally uses native Streamlit components instead of
    injecting a large HTML table through st.markdown/st.html. That prevents
    Streamlit from displaying the HTML source as a dark code block.
    """

    explainability = result.get("explainability") or {}
    status = str(explainability.get("status", "")).lower()

    st.markdown(
        f"### 🧠 Why did the {model_name} model make this prediction?"
    )

    if status != "success":
        st.warning(
            "The prediction completed, but model contribution details "
            "were not returned by the backend."
        )

        error = explainability.get("error")
        if error:
            with st.expander(
                f"🔎 {model_name} explainability diagnostic",
                expanded=False,
            ):
                st.code(str(error))

        return

    factors = (
        explainability.get("top_factors")
        or explainability.get("top_features")
        or []
    )

    method = (
        explainability.get("method")
        or explainability.get("explanation_method")
        or "SHAP"
    )

    base_value = explainability.get("base_value")

    # Summary row
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(
            f"""
            <div style="
                background:#F3E8FF;
                border:1px solid #DDD0F7;
                border-radius:12px;
                padding:10px 14px;
                color:#4C1D95;
                font-weight:700;
            ">
                🔬 Explanation: {method}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        if base_value is not None:
            try:
                base_text = f"{float(base_value):.4f}"
            except (TypeError, ValueError):
                base_text = str(base_value)

            st.metric(
                "Model Base Value",
                base_text,
            )

    if not factors:
        st.info(
            f"{model_name}: no SHAP contribution factors were returned."
        )
        return

    rows = []

    for factor in factors:
        if not isinstance(factor, dict):
            continue

        feature = (
            factor.get("feature")
            or factor.get("name")
            or factor.get("feature_name")
            or "Unknown"
        )

        value = factor.get("value", "—")

        shap_value = factor.get("shap_value")
        if shap_value is None:
            shap_value = factor.get("impact")
        if shap_value is None:
            shap_value = factor.get("contribution")

        absolute_impact = factor.get("absolute_impact")

        if absolute_impact is None and shap_value is not None:
            try:
                absolute_impact = abs(float(shap_value))
            except (TypeError, ValueError):
                absolute_impact = None

        effect = (
            factor.get("effect")
            or factor.get("direction")
            or ""
        )

        try:
            shap_num = float(shap_value)
            shap_text = f"{shap_num:+.4f}"
            impact_text = (
                f"{float(absolute_impact):.4f}"
                if absolute_impact is not None
                else f"{abs(shap_num):.4f}"
            )

            if shap_num > 0:
                effect_text = "Supports prediction"
            elif shap_num < 0:
                effect_text = "Opposes prediction"
            else:
                effect_text = "Neutral"

        except (TypeError, ValueError):
            shap_text = "—"
            impact_text = "—"
            effect_text = str(effect) if effect else "Model contribution"

        # If backend supplies a meaningful effect, preserve it.
        if effect:
            effect_text = str(effect).replace("_", " ").title()

        rows.append(
            {
                "Feature": str(feature),
                "Value": str(value),
                "SHAP Value": shap_text,
                "Impact": impact_text,
                "Effect": effect_text,
            }
        )

    if not rows:
        st.info(
            f"{model_name}: no usable SHAP factors were returned."
        )
        return

    # Native dataframe = reliable rendering across Streamlit versions.
    import pandas as pd

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn(
                "Feature",
                width="medium",
            ),
            "Value": st.column_config.TextColumn(
                "Value",
                width="medium",
            ),
            "SHAP Value": st.column_config.TextColumn(
                "SHAP Value",
                width="small",
            ),
            "Impact": st.column_config.TextColumn(
                "Impact",
                width="small",
            ),
            "Effect": st.column_config.TextColumn(
                "Effect",
                width="medium",
            ),
        },
    )

    note = explainability.get("note")
    if note:
        st.caption(str(note))



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

        st.markdown(
            '<div class="section-card" style="padding:12px 14px;margin:0;">'
            '<span class="live-dot"></span><b style="color:#15803D;">AI SYSTEM ONLINE</b>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="section-card" style="padding:12px 14px;margin:0;">'
            '<b style="color:#B91C1C;">● SERVICE OFFLINE</b>'
            '</div>',
            unsafe_allow_html=True
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

    st.markdown(
        """
        <div class="section-card">
            <div style="font-size:12px;font-weight:800;color:#6D28D9;
                        text-transform:uppercase;letter-spacing:.12em;">
                AI ASSISTANT
            </div>
            <div style="font-size:14px;color:#6B6475;margin-top:5px;">
                Answers are shown in dark text for comfortable reading.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
# ==========================================================
# RENEWAL INTELLIGENCE - PREMIUM V3
# ==========================================================

elif page == "Renewal Intelligence":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">AI RENEWAL INTELLIGENCE</div>
        <div class="hero-title">Renewal Risk Prediction</div>
        <div class="hero-subtitle">
            Renewal V3 • 32-feature XGBoost model • SHAP explainability • Human-in-the-loop
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------
    # Quick model strip
    # ------------------------------------------------------
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown('<div class="mini-stat"><strong>V3</strong><span>MODEL VERSION</span></div>',
                    unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="mini-stat"><strong>32</strong><span>MODEL FEATURES</span></div>',
                    unsafe_allow_html=True)
    with q3:
        st.markdown('<div class="mini-stat"><strong>30.5%</strong><span>DECISION THRESHOLD</span></div>',
                    unsafe_allow_html=True)
    with q4:
        online = backend_is_online()
        status = "ONLINE" if online else "OFFLINE"
        dot = '<span class="live-dot"></span>' if online else "● "
        st.markdown(
            f'<div class="mini-stat"><strong>{dot}{status}</strong><span>FASTAPI STATUS</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # ------------------------------------------------------
    # Form
    # ------------------------------------------------------
    with st.form("renewal_form_v3_premium"):

        st.markdown("### 💜 Policy & Premium Profile")
        st.caption("Core pricing, policy tenure and risk characteristics used by Renewal V3.")

        c1, c2, c3 = st.columns(3)

        with c1:
            old_premium = st.number_input(
                "Old Premium (₹)", min_value=0.0, value=20000.0,
                step=100.0, help="Previous renewal premium."
            )
            new_premium = st.number_input(
                "New Premium (₹)", min_value=0.0, value=21000.0,
                step=100.0, help="Current renewal premium."
            )
            premium_increase_pct = st.number_input(
                "Premium Increase (%)", min_value=-100.0, value=5.0,
                step=0.1
            )
            premium_increase_amount = st.number_input(
                "Increase Amount (₹)", min_value=-100000000.0, value=1000.0,
                step=100.0
            )

        with c2:
            loyalty_years = st.number_input(
                "Loyalty Years", min_value=0.0, value=5.0, step=0.5
            )
            sum_insured = st.number_input(
                "Sum Insured (₹)", min_value=0.0, value=500000.0,
                step=10000.0
            )
            annual_premium = st.number_input(
                "Annual Premium (₹)", min_value=0.0, value=20000.0,
                step=100.0
            )
            risk_score = st.number_input(
                "Risk Score", min_value=0.0, value=50.0, step=1.0
            )

        with c3:
            discount_pct = st.number_input(
                "Discount (%)", min_value=0.0, max_value=100.0,
                value=10.0, step=0.1
            )
            commission_pct = st.number_input(
                "Commission (%)", min_value=0.0, max_value=100.0,
                value=2.5, step=0.1
            )
            policy_tenure_days = st.number_input(
                "Policy Tenure (Days)", min_value=0.0, value=1095.0, step=1.0
            )
            policy_tenure_years = st.number_input(
                "Policy Tenure (Years)", min_value=0.0, value=3.0, step=0.1
            )

        st.divider()

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            premium_to_sum_insured = st.number_input(
                "Premium / Sum Insured", min_value=0.0,
                value=0.04, step=0.001, format="%.6f"
            )
        with r2:
            discounted_premium_ratio = st.number_input(
                "Discounted Premium Ratio", min_value=0.0,
                value=0.036, step=0.001, format="%.6f"
            )
        with r3:
            claim_history = st.selectbox("Claim History", YES_NO_OPTIONS, index=0)
        with r4:
            policy_type = st.selectbox("Policy Type", POLICY_TYPE_OPTIONS, index=0)

        st.markdown("### 💳 Payment Behaviour")
        st.caption("Prior payment activity is represented using time-safe historical features.")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            total_payment_txns_prior = st.number_input(
                "Previous Transactions", min_value=0.0, value=2.0, step=1.0
            )
            paid_txns_prior = st.number_input(
                "Paid Transactions", min_value=0.0, value=2.0, step=1.0
            )

        with p2:
            failed_txns_prior = st.number_input(
                "Failed Transactions", min_value=0.0, value=0.0, step=1.0
            )
            overdue_txns_prior = st.number_input(
                "Overdue Transactions", min_value=0.0, value=0.0, step=1.0
            )

        with p3:
            no_delay_history = st.checkbox(
                "No payment-delay history",
                value=False,
                help="Sends None for average and maximum payment delay."
            )
            avg_payment_delay_prior = st.number_input(
                "Average Delay (Days)", min_value=0.0, value=5.0,
                step=0.5, disabled=no_delay_history
            )
            max_payment_delay_prior = st.number_input(
                "Maximum Delay (Days)", min_value=0.0, value=7.0,
                step=0.5, disabled=no_delay_history
            )

        with p4:
            total_premium_prior = st.number_input(
                "Previous Premium Total (₹)", min_value=0.0,
                value=25000.0, step=100.0
            )
            avg_premium_prior = st.number_input(
                "Previous Average Premium (₹)", min_value=0.0,
                value=12500.0, step=100.0
            )

        p5, p6, p7, p8 = st.columns(4)

        with p5:
            paid_rate_prior = st.number_input(
                "Paid Rate", min_value=0.0, max_value=1.0,
                value=0.90, step=0.01, format="%.4f"
            )
        with p6:
            failed_rate_prior = st.number_input(
                "Failed Rate", min_value=0.0, max_value=1.0,
                value=0.02, step=0.01, format="%.4f"
            )
        with p7:
            overdue_rate_prior = st.number_input(
                "Overdue Rate", min_value=0.0, max_value=1.0,
                value=0.03, step=0.01, format="%.4f"
            )
        with p8:
            premium_payment_ratio = st.number_input(
                "Premium Payment Ratio", min_value=0.0,
                value=0.90, step=0.01, format="%.4f"
            )

        st.divider()

        st.markdown("### 🏷️ Customer & Channel Profile")

        c4, c5, c6, c7 = st.columns(4)

        with c4:
            payment_mode = st.selectbox(
                "Payment Mode", PAYMENT_MODE_OPTIONS, index=1
            )
        with c5:
            sales_channel = st.selectbox(
                "Sales Channel", SALES_CHANNEL_OPTIONS, index=1
            )
        with c6:
            contact_channel = st.selectbox(
                "Contact Channel",
                ["Email", "SMS", "WhatsApp", "Phone", "Agent"],
                index=2
            )
        with c7:
            risk_band = st.selectbox(
                "Risk Band", RISK_BAND_OPTIONS, index=0
            )

        st.markdown(
            """
            <div class="section-card">
                <span class="purple-badge">MODEL INPUT</span>
                <span class="feature-chip">32 V3 Features</span>
                <span class="feature-chip">XGBoost</span>
                <span class="feature-chip">SHAP</span>
                <span class="feature-chip">Time-safe payment history</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        renewal_submit = st.form_submit_button(
            "✨ Analyze Renewal Risk",
            type="primary",
            use_container_width=True
        )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------
    if renewal_submit:

        claim_history_flag = 1.0 if claim_history == "Yes" else 0.0

        avg_delay_payload = (
            None if no_delay_history else float(avg_payment_delay_prior)
        )
        max_delay_payload = (
            None if no_delay_history else float(max_payment_delay_prior)
        )

        payload = {
            "OLD_PREMIUM": float(old_premium),
            "NEW_PREMIUM": float(new_premium),
            "PREMIUM_INCREASE_PCT": float(premium_increase_pct),
            "PREMIUM_INCREASE_AMOUNT": float(premium_increase_amount),
            "LOYALTY_YEARS": float(loyalty_years),
            "CLAIM_HISTORY_FLAG": claim_history_flag,
            "SUM_INSURED": float(sum_insured),
            "ANNUAL_PREMIUM": float(annual_premium),
            "RISK_SCORE": float(risk_score),
            "DISCOUNT_PCT": float(discount_pct),
            "COMMISSION_PCT": float(commission_pct),
            "POLICY_TENURE_DAYS": float(policy_tenure_days),
            "POLICY_TENURE_YEARS": float(policy_tenure_years),
            "PREMIUM_TO_SUM_INSURED": float(premium_to_sum_insured),
            "DISCOUNTED_PREMIUM_RATIO": float(discounted_premium_ratio),
            "TOTAL_PAYMENT_TXNS_PRIOR": float(total_payment_txns_prior),
            "PAID_TXNS_PRIOR": float(paid_txns_prior),
            "FAILED_TXNS_PRIOR": float(failed_txns_prior),
            "OVERDUE_TXNS_PRIOR": float(overdue_txns_prior),
            "AVG_PAYMENT_DELAY_PRIOR": avg_delay_payload,
            "MAX_PAYMENT_DELAY_PRIOR": max_delay_payload,
            "TOTAL_PREMIUM_PRIOR": float(total_premium_prior),
            "AVG_PREMIUM_PRIOR": float(avg_premium_prior),
            "PAID_RATE_PRIOR": float(paid_rate_prior),
            "FAILED_RATE_PRIOR": float(failed_rate_prior),
            "OVERDUE_RATE_PRIOR": float(overdue_rate_prior),
            "PREMIUM_PAYMENT_RATIO": float(premium_payment_ratio),
            "POLICY_TYPE": policy_type,
            "PAYMENT_MODE": payment_mode,
            "SALES_CHANNEL": sales_channel,
            "CONTACT_CHANNEL": contact_channel,
            "RISK_BAND": risk_band,
        }

        with st.spinner("Running Renewal V3 intelligence..."):
            result = call_api("/predict/renewal", payload)

        if result:
            st.divider()

            renewal_probability = safe_float(
                result.get("renewal_probability")
            )
            churn_probability = safe_float(
                result.get("churn_probability")
            )
            threshold = safe_float(result.get("threshold"), 0.305)
            prediction = result.get("prediction", "Unavailable")
            model_version = result.get("model_version", "V3")

            is_churn = bool(result.get("churn", False))

            # ----------------------------
            # Result banner
            # ----------------------------
            if is_churn:
                badge = '<span class="risk-badge risk-high">⚠ ELEVATED NON-RENEWAL RISK</span>'
                headline = "Churn Risk Detected"
                supporting = "The model probability is above the configured V3 decision threshold."
            else:
                badge = '<span class="risk-badge risk-low">✓ LOWER NON-RENEWAL RISK</span>'
                headline = "Renewal Signal"
                supporting = "The model probability is below the configured V3 decision threshold."

            st.markdown(
                f"""
                <div class="result-card">
                    {badge}
                    <h2 style="margin:12px 0 4px 0;color:#4C1D95;">{headline}</h2>
                    <div style="color:#6B6475;">{supporting}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

            # ----------------------------
            # Probability cards
            # ----------------------------
            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Renewal Probability</div>
                        <div class="metric-value">{renewal_probability * 100:.2f}%</div>
                        <div class="metric-small">Estimated likelihood of renewal</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Churn Probability</div>
                        <div class="metric-value">{churn_probability * 100:.2f}%</div>
                        <div class="metric-small">Estimated likelihood of non-renewal</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m3:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">V3 Decision Threshold</div>
                        <div class="metric-value">{threshold * 100:.2f}%</div>
                        <div class="metric-small">Binary decision boundary</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("")
            st.subheader("📈 Renewal vs Churn")

            # Two-sided visual comparison
            left, right = st.columns(2)
            with left:
                st.caption("Renewal")
                st.progress(min(max(renewal_probability, 0.0), 1.0))
                st.markdown(
                    f"**{renewal_probability * 100:.2f}%**",
                    unsafe_allow_html=True
                )
            with right:
                st.caption("Churn")
                st.progress(min(max(churn_probability, 0.0), 1.0))
                st.markdown(
                    f"**{churn_probability * 100:.2f}%**",
                    unsafe_allow_html=True
                )

            st.markdown("### 🧾 Policy Snapshot")

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Policy Type", policy_type)
            s2.metric("Payment Mode", payment_mode)
            s3.metric("Sales Channel", sales_channel)
            s4.metric("Risk Band", risk_band)
            s5.metric("Claim History", claim_history)

            st.markdown("### 🤖 Model Information")

            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Model", model_version)
            i2.metric("Features", "32")
            i3.metric("Threshold", f"{threshold * 100:.2f}%")
            i4.metric("Decision", prediction)

            # ----------------------------
            # SHAP / EXPLAINABILITY
            # ----------------------------
            show_explainability(
                result,
                model_name="Renewal V3",
            )

            # If the backend explainer itself failed, surface the exact
            # backend message instead of silently hiding the problem.
            renewal_explainability = result.get("explainability") or {}
            if str(renewal_explainability.get("status", "")).lower() != "success":
                st.caption(
                    "Renewal prediction is working. SHAP details depend on "
                    "the backend renewal_explainer response."
                )

            # ----------------------------
            # Monitoring
            # ----------------------------
            show_monitoring(result)

            # ----------------------------
            # Technical details
            # ----------------------------
            with st.expander("🔧 Technical Model Output"):
                st.json(result)

            with st.expander("📦 V3 Request Payload"):
                st.json(payload)

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
                result,
                model_name="Fraud",
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
                result,
                model_name="Underwriting",
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