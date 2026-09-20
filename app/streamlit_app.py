import requests
import streamlit as st
from textwrap import dedent


# ==========================================================
# CONFIGURATION
# ==========================================================

API_BASE_URL = "http://127.0.0.1:8000"


# ==========================================================
# MODEL CATEGORIES
# ==========================================================

GENDER_OPTIONS = ["Other", "Female", "Male"]

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
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="INSURE AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# HTML HELPERS
# ==========================================================

def clean_html(content: str) -> str:
    """
    Remove blank lines and extra indentation so Streamlit
    renders HTML instead of displaying it as text/code.
    """

    content = dedent(content).strip()

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def render_html(content: str):

    st.markdown(
        clean_html(content),
        unsafe_allow_html=True,
    )


def render_sidebar_html(content: str):

    st.sidebar.markdown(
        clean_html(content),
        unsafe_allow_html=True,
    )


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
<style>

/* ========================================================
   APP BACKGROUND
======================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(20, 95, 255, 0.17),
            transparent 30%
        ),
        radial-gradient(
            circle at 70% 80%,
            rgba(100, 55, 255, 0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 5% 85%,
            rgba(0, 190, 220, 0.07),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #020409 0%,
            #07101d 45%,
            #050712 100%
        );

    background-size: 180% 180%;
    animation: backgroundMove 18s ease infinite;
}


@keyframes backgroundMove {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}


/* ========================================================
   PAGE WIDTH
======================================================== */

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 4rem;
}


/* ========================================================
   SIDEBAR
======================================================== */

[data-testid="stSidebar"] {
    background: rgba(3, 6, 14, 0.96);
    border-right: 1px solid rgba(148, 163, 184, 0.09);
}


/* ========================================================
   BRAND
======================================================== */

.brand-wrap {
    display: flex;
    align-items: center;
    gap: 13px;
    margin: 5px 0 25px 0;
}


.brand-symbol {
    width: 54px;
    height: 54px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 16px;

    background:
        linear-gradient(
            145deg,
            #1769ff,
            #5947ff,
            #00a9c7
        );

    color: white;

    font-size: 18px;
    font-weight: 900;
    letter-spacing: 1px;

    box-shadow:
        0 0 30px
        rgba(37, 99, 235, 0.35);

    animation:
        logoGlow
        4s ease-in-out infinite;
}


@keyframes logoGlow {

    0%,
    100% {
        box-shadow:
            0 0 18px
            rgba(37, 99, 235, 0.25);
    }

    50% {
        box-shadow:
            0 0 42px
            rgba(79, 70, 229, 0.55);
    }
}


.brand-main {
    color: white;
    font-size: 21px;
    font-weight: 900;
    letter-spacing: 1px;
}


.brand-small {
    margin-top: 3px;
    color: #64748b;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
}


/* ========================================================
   SIDEBAR NAVIGATION
======================================================== */

[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 7px;
}


[data-testid="stSidebar"] div[role="radiogroup"] label {

    padding: 11px 13px;

    border-radius: 11px;

    background:
        rgba(255, 255, 255, 0.018);

    border:
        1px solid transparent;

    transition:
        all 0.25s ease;
}


[data-testid="stSidebar"] div[role="radiogroup"] label:hover {

    transform:
        translateX(4px);

    background:
        rgba(37, 99, 235, 0.10);

    border-color:
        rgba(59, 130, 246, 0.20);
}


/* ========================================================
   STATUS
======================================================== */

.online-status {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding:
        7px 12px;

    border-radius:
        25px;

    color:
        #86efac;

    background:
        rgba(34, 197, 94, 0.08);

    border:
        1px solid
        rgba(34, 197, 94, 0.17);

    font-size:
        11px;

    font-weight:
        700;
}


.online-dot {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        #22c55e;

    box-shadow:
        0 0 10px
        #22c55e;

    animation:
        pulseDot
        1.6s infinite;
}


@keyframes pulseDot {

    0%,
    100% {
        opacity: 0.4;
    }

    50% {
        opacity: 1;
    }
}


.offline-status {

    display: inline-flex;

    padding:
        7px 12px;

    border-radius:
        25px;

    background:
        rgba(239, 68, 68, 0.08);

    color:
        #fca5a5;

    font-size:
        11px;

    font-weight:
        700;
}


/* ========================================================
   HOME HERO
======================================================== */

.home-hero {

    min-height:
        570px;

    display:
        flex;

    align-items:
        center;
}


.hero-tag {

    color:
        #60a5fa;

    font-size:
        11px;

    font-weight:
        800;

    letter-spacing:
        2.2px;

    margin-bottom:
        15px;
}


.hero-title {

    margin:
        0;

    color:
        #f8fafc;

    font-size:
        clamp(
            62px,
            8vw,
            110px
        );

    line-height:
        0.84;

    letter-spacing:
        -5px;

    font-weight:
        950;
}


.hero-title span {

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #93c5fd,
            #818cf8
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


.hero-description {

    max-width:
        560px;

    margin-top:
        28px;

    color:
        #94a3b8;

    font-size:
        17px;

    line-height:
        1.7;
}


.capability-list {

    margin-top:
        27px;

    color:
        #cbd5e1;

    font-size:
        12px;

    font-weight:
        750;

    letter-spacing:
        1px;

    line-height:
        2.1;
}


.capability-list span {

    color:
        #3b82f6;

    margin-right:
        8px;
}


/* ========================================================
   HOME MOSAIC
======================================================== */

.mosaic {

    position:
        relative;

    min-height:
        570px;

    width:
        100%;
}


.mosaic-card {

    position:
        absolute;

    overflow:
        hidden;

    padding:
        22px;

    border:
        1px solid
        rgba(255, 255, 255, 0.08);

    border-radius:
        6px;

    box-shadow:
        0 30px 70px
        rgba(0, 0, 0, 0.35);

    transition:
        all 0.35s ease;
}


.mosaic-card:hover {

    transform:
        translateY(-6px)
        scale(1.01);

    border-color:
        rgba(96, 165, 250, 0.30);

    box-shadow:
        0 35px 90px
        rgba(37, 99, 235, 0.16);
}


.card-copilot {

    top:
        15px;

    left:
        14%;

    width:
        71%;

    height:
        210px;

    background:
        linear-gradient(
            145deg,
            rgba(5, 24, 46, 0.98),
            rgba(6, 57, 76, 0.82)
        );
}


.card-renewal {

    top:
        188px;

    left:
        0;

    width:
        56%;

    height:
        205px;

    background:
        linear-gradient(
            145deg,
            rgba(20, 22, 40, 0.98),
            rgba(49, 46, 129, 0.78)
        );
}


.card-fraud {

    top:
        266px;

    right:
        0;

    width:
        47%;

    height:
        175px;

    background:
        linear-gradient(
            145deg,
            rgba(43, 14, 27, 0.98),
            rgba(127, 29, 29, 0.67)
        );
}


.card-underwriting {

    top:
        405px;

    left:
        24%;

    width:
        57%;

    height:
        145px;

    background:
        linear-gradient(
            145deg,
            rgba(8, 28, 35, 0.98),
            rgba(15, 118, 110, 0.57)
        );
}


.mosaic-label {

    color:
        rgba(255, 255, 255, 0.50);

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        1.5px;
}


.mosaic-title {

    margin-top:
        9px;

    color:
        white;

    font-size:
        24px;

    font-weight:
        850;
}


.mosaic-text {

    max-width:
        350px;

    margin-top:
        8px;

    color:
        rgba(255, 255, 255, 0.67);

    font-size:
        13px;

    line-height:
        1.5;
}


.mosaic-number {

    position:
        absolute;

    right:
        18px;

    bottom:
        7px;

    color:
        rgba(255, 255, 255, 0.055);

    font-size:
        72px;

    font-weight:
        950;
}


/* ========================================================
   PAGE HEADERS
======================================================== */

.page-label {

    margin-bottom:
        7px;

    color:
        #60a5fa;

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        2px;
}


.page-title {

    color:
        white;

    margin:
        0 0 8px 0;

    font-size:
        40px;

    font-weight:
        850;

    letter-spacing:
        -1px;
}


.page-description {

    max-width:
        800px;

    margin-bottom:
        26px;

    color:
        #94a3b8;

    font-size:
        14px;

    line-height:
        1.6;
}


/* ========================================================
   METRIC CARDS
======================================================== */

div[data-testid="stMetric"] {

    padding:
        18px;

    border-radius:
        15px;

    background:
        rgba(255, 255, 255, 0.028);

    border:
        1px solid
        rgba(148, 163, 184, 0.09);
}


/* ========================================================
   BUTTONS
======================================================== */

.stButton > button,
.stFormSubmitButton > button {

    min-height:
        44px;

    border-radius:
        9px;

    border:
        1px solid
        rgba(96, 165, 250, 0.20);

    font-weight:
        700;

    transition:
        all 0.25s ease;
}


.stButton > button:hover,
.stFormSubmitButton > button:hover {

    transform:
        translateY(-2px);

    border-color:
        #3b82f6;

    box-shadow:
        0 10px 28px
        rgba(37, 99, 235, 0.18);
}


/* ========================================================
   CHAT
======================================================== */

[data-testid="stChatMessage"] {

    margin-bottom:
        10px;

    padding:
        13px;

    border-radius:
        15px;

    background:
        rgba(255, 255, 255, 0.025);

    border:
        1px solid
        rgba(148, 163, 184, 0.08);
}


.agent-badge {

    display:
        inline-block;

    margin-top:
        8px;

    padding:
        5px 10px;

    border-radius:
        20px;

    background:
        rgba(59, 130, 246, 0.09);

    border:
        1px solid
        rgba(59, 130, 246, 0.17);

    color:
        #93c5fd;

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        1px;
}


/* ========================================================
   FOOTER
======================================================== */

.footer {

    margin-top:
        50px;

    padding-top:
        18px;

    border-top:
        1px solid
        rgba(148, 163, 184, 0.08);

    color:
        #475569;

    text-align:
        center;

    font-size:
        11px;

    letter-spacing:
        0.5px;
}


/* ========================================================
   SCROLLBAR
======================================================== */

::-webkit-scrollbar {
    width:
        6px;
}


::-webkit-scrollbar-track {
    background:
        transparent;
}


::-webkit-scrollbar-thumb {

    border-radius:
        20px;

    background:
        rgba(59, 130, 246, 0.32);
}

</style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# API HELPERS
# ==========================================================

def call_api(endpoint: str, payload: dict):

    try:

        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=90,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:

        st.error(
            f"Unable to connect to AI service: {error}"
        )

        return None


def backend_is_online():

    try:

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


# ==========================================================
# SESSION STATE
# ==========================================================

NAV_OPTIONS = [
    "Home",
    "AI Copilot",
    "Renewal Intelligence",
    "Fraud Intelligence",
    "Underwriting",
]


if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


if "pending_page" in st.session_state:

    st.session_state.current_page = (
        st.session_state.pending_page
    )

    del st.session_state.pending_page


if st.session_state.current_page not in NAV_OPTIONS:
    st.session_state.current_page = "Home"


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "quick_question" not in st.session_state:
    st.session_state.quick_question = None


def navigate(page_name: str):

    st.session_state.pending_page = page_name

    st.rerun()


# ==========================================================
# RAG SOURCE RENDERER
# ==========================================================

def render_sources(sources):

    if not sources:
        return

    with st.expander(
        "Knowledge Sources"
    ):

        for item in sources:

            source_name = item.get(
                "source",
                "Unknown",
            )

            page_number = item.get(
                "page"
            )

            chunk = item.get(
                "chunk"
            )

            score = item.get(
                "score"
            )

            text = f"📄 {source_name}"

            if page_number is not None:
                text += f" | Page {page_number}"

            elif chunk is not None:
                text += f" | Chunk {chunk}"

            if score is not None:
                text += f" | Similarity {score:.2f}"

            st.write(
                text
            )


# ==========================================================
# SIDEBAR BRAND
# ==========================================================

render_sidebar_html(
    """
    <div class="brand-wrap">
        <div class="brand-symbol">IA</div>
        <div>
            <div class="brand-main">INSURE AI</div>
            <div class="brand-small">INSURANCE INTELLIGENCE</div>
        </div>
    </div>
    """
)


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

NAV_LABELS = {
    "Home":
        "⌂  Home",

    "AI Copilot":
        "✦  AI Copilot",

    "Renewal Intelligence":
        "↻  Renewal Intelligence",

    "Fraud Intelligence":
        "◉  Fraud Intelligence",

    "Underwriting":
        "◇  Underwriting",
}


radio_options = [
    NAV_LABELS[item]
    for item in NAV_OPTIONS
]


selected_label = st.sidebar.radio(
    "Navigation",
    options=radio_options,
    index=NAV_OPTIONS.index(
        st.session_state.current_page
    ),
    label_visibility="collapsed",
)


reverse_navigation = {
    value: key
    for key, value
    in NAV_LABELS.items()
}


selected_page = reverse_navigation[
    selected_label
]


if selected_page != st.session_state.current_page:

    st.session_state.current_page = selected_page


page = st.session_state.current_page


# ==========================================================
# BACKEND STATUS
# ==========================================================

st.sidebar.divider()


if backend_is_online():

    render_sidebar_html(
        """
        <div class="online-status">
            <span class="online-dot"></span>
            AI SYSTEM ONLINE
        </div>
        """
    )

else:

    render_sidebar_html(
        """
        <div class="offline-status">
            ● SERVICE OFFLINE
        </div>
        """
    )


# ==========================================================
# OPTIONAL PLATFORM DETAILS
# ==========================================================

st.sidebar.markdown("")


with st.sidebar.expander(
    "About the platform"
):

    st.caption(
        "Technical architecture"
    )

    st.write(
        "Backend: FastAPI"
    )

    st.write(
        "Database: PostgreSQL"
    )

    st.write(
        "ML: XGBoost"
    )

    st.write(
        "Agent Orchestration: LangGraph"
    )

    st.write(
        "Knowledge Search: Semantic RAG"
    )


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    left, right = st.columns(
        [0.92, 1.18],
        gap="large",
    )


    # ------------------------------------------------------
    # LEFT HERO
    # ------------------------------------------------------

    with left:

        render_html(
            """
            <div class="home-hero">
                <div>
                    <div class="hero-tag">
                        INTELLIGENT INSURANCE PLATFORM
                    </div>
                    <h1 class="hero-title">
                        INSURE<br>
                        <span>AI.</span>
                    </h1>
                    <div class="hero-description">
                        Insurance intelligence built for faster analysis,
                        better risk visibility and smarter human decisions.
                    </div>
                    <div class="capability-list">
                        <div><span>•</span> ASK YOUR INSURANCE DATA</div>
                        <div><span>•</span> PREDICT POLICY RENEWAL RISK</div>
                        <div><span>•</span> PRIORITIZE SUSPICIOUS CLAIMS</div>
                        <div><span>•</span> SUPPORT UNDERWRITING TEAMS</div>
                        <div><span>•</span> SEARCH INSURANCE KNOWLEDGE</div>
                    </div>
                </div>
            </div>
            """
        )


        button1, button2 = st.columns(
            [1.1, 1]
        )


        with button1:

            if st.button(
                "✦ Explore AI Copilot",
                type="primary",
                use_container_width=True,
            ):

                navigate(
                    "AI Copilot"
                )


        with button2:

            if st.button(
                "View Renewal Intelligence",
                use_container_width=True,
            ):

                navigate(
                    "Renewal Intelligence"
                )


    # ------------------------------------------------------
    # RIGHT CINEMATIC CARDS
    # ------------------------------------------------------

    with right:

        render_html(
            """
            <div class="mosaic">
                <div class="mosaic-card card-copilot">
                    <div class="mosaic-label">
                        CONVERSATIONAL INTELLIGENCE
                    </div>
                    <div class="mosaic-title">
                        AI Copilot
                    </div>
                    <div class="mosaic-text">
                        Ask questions across insurance data and documents
                        using natural language.
                    </div>
                    <div class="mosaic-number">
                        01
                    </div>
                </div>
                <div class="mosaic-card card-renewal">
                    <div class="mosaic-label">
                        CUSTOMER RETENTION
                    </div>
                    <div class="mosaic-title">
                        Renewal Intelligence
                    </div>
                    <div class="mosaic-text">
                        Understand renewal probability and identify
                        elevated churn risk.
                    </div>
                    <div class="mosaic-number">
                        02
                    </div>
                </div>
                <div class="mosaic-card card-fraud">
                    <div class="mosaic-label">
                        CLAIM INVESTIGATION
                    </div>
                    <div class="mosaic-title">
                        Fraud Intelligence
                    </div>
                    <div class="mosaic-text">
                        Prioritize claims requiring additional investigation.
                    </div>
                    <div class="mosaic-number">
                        03
                    </div>
                </div>
                <div class="mosaic-card card-underwriting">
                    <div class="mosaic-label">
                        RISK ASSESSMENT
                    </div>
                    <div class="mosaic-title">
                        Underwriting Support
                    </div>
                    <div class="mosaic-text">
                        Model-assisted applicant risk assessment.
                    </div>
                    <div class="mosaic-number">
                        04
                    </div>
                </div>
            </div>
            """
        )


    # ------------------------------------------------------
    # QUICK ACCESS
    # ------------------------------------------------------

    st.divider()


    st.subheader(
        "Quick Access"
    )


    q1, q2, q3, q4 = st.columns(4)


    with q1:

        if st.button(
            "✦ AI Copilot",
            use_container_width=True,
        ):

            navigate(
                "AI Copilot"
            )


    with q2:

        if st.button(
            "↻ Renewal",
            use_container_width=True,
        ):

            navigate(
                "Renewal Intelligence"
            )


    with q3:

        if st.button(
            "◉ Fraud",
            use_container_width=True,
        ):

            navigate(
                "Fraud Intelligence"
            )


    with q4:

        if st.button(
            "◇ Underwriting",
            use_container_width=True,
        ):

            navigate(
                "Underwriting"
            )


# ==========================================================
# AI COPILOT PAGE
# ==========================================================

elif page == "AI Copilot":

    render_html(
        """
        <div class="page-label">
            CONVERSATIONAL INTELLIGENCE
        </div>
        <div class="page-title">
            AI Copilot
        </div>
        <div class="page-description">
            Ask questions about customers, policies, claims,
            premiums or insurance documents.
        </div>
        """
    )


    # ------------------------------------------------------
    # QUICK QUESTIONS
    # ------------------------------------------------------

    q1, q2, q3 = st.columns(3)


    with q1:

        if st.button(
            "Customer Distribution",
            use_container_width=True,
        ):

            st.session_state.quick_question = (
                "Which states have the highest number of customers?"
            )


    with q2:

        if st.button(
            "Claim Requirements",
            use_container_width=True,
        ):

            st.session_state.quick_question = (
                "According to the claim guide, "
                "what documents are required "
                "for an insurance claim?"
            )


    with q3:

        if st.button(
            "Premium Analysis",
            use_container_width=True,
        ):

            st.session_state.quick_question = (
                "Which policy types generate "
                "the highest annual premium?"
            )


    # ------------------------------------------------------
    # CLEAR CHAT
    # ------------------------------------------------------

    clear_col, _ = st.columns(
        [1, 5]
    )


    with clear_col:

        if st.button(
            "Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.chat_history = []

            st.rerun()


    st.divider()


    # ------------------------------------------------------
    # EXISTING CHAT
    # ------------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


            if message["role"] == "assistant":

                route = message.get(
                    "route"
                )


                if route:

                    render_html(
                        f"""
                        <div class="agent-badge">
                            {route.upper()} INTELLIGENCE
                        </div>
                        """
                    )


                sql = message.get(
                    "sql"
                )


                if sql:

                    with st.expander(
                        "Generated SQL"
                    ):

                        st.code(
                            sql,
                            language="sql",
                        )


                results = (
                    message.get("results")
                    or message.get("rows")
                )


                if results:

                    st.dataframe(
                        results,
                        use_container_width=True,
                    )


                render_sources(
                    message.get(
                        "sources",
                        [],
                    )
                )


    # ------------------------------------------------------
    # CHAT INPUT
    # ------------------------------------------------------

    typed_question = st.chat_input(
        "Ask your insurance question..."
    )


    user_question = (
        st.session_state.quick_question
        or typed_question
    )


    # ------------------------------------------------------
    # PROCESS QUESTION
    # ------------------------------------------------------

    if user_question:

        st.session_state.quick_question = None


        st.session_state.chat_history.append(
            {
                "role":
                    "user",

                "content":
                    user_question,
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_question
            )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing your request..."
            ):

                data = call_api(
                    "/ask/copilot",
                    {
                        "question":
                            user_question,

                        "input_data":
                            None,
                    },
                )


            if data:

                route = data.get(
                    "route",
                    "unknown",
                )


                result = data.get(
                    "result",
                    {},
                )


                answer = result.get(
                    "answer"
                )


                if not answer:

                    answer = result.get(
                        "prediction"
                    )


                if not answer:

                    answer = result.get(
                        "error"
                    )


                if not answer:

                    answer = (
                        "Request processed successfully."
                    )


                st.markdown(
                    str(answer)
                )


                render_html(
                    f"""
                    <div class="agent-badge">
                        {route.upper()} INTELLIGENCE
                    </div>
                    """
                )


                sql = result.get(
                    "sql"
                )


                rows = (
                    result.get("results")
                    or result.get("rows")
                )


                sources = result.get(
                    "sources",
                    [],
                )


                if sql:

                    with st.expander(
                        "Generated SQL"
                    ):

                        st.code(
                            sql,
                            language="sql",
                        )


                if rows:

                    st.dataframe(
                        rows,
                        use_container_width=True,
                    )


                render_sources(
                    sources
                )


                st.session_state.chat_history.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            str(answer),

                        "route":
                            route,

                        "sql":
                            sql,

                        "results":
                            rows,

                        "sources":
                            sources,
                    }
                )


# ==========================================================
# RENEWAL INTELLIGENCE PAGE
# ==========================================================

elif page == "Renewal Intelligence":

    render_html(
        """
        <div class="page-label">
            CUSTOMER RETENTION
        </div>
        <div class="page-title">
            Renewal Intelligence
        </div>
        <div class="page-description">
            Estimate renewal probability and identify
            customers with elevated policy churn risk.
        </div>
        """
    )


    with st.form(
        "renewal_form"
    ):

        st.subheader(
            "Financial & Behavioral Profile"
        )


        c1, c2, c3 = st.columns(3)


        # --------------------------------------------------
        # COLUMN 1
        # --------------------------------------------------

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


        # --------------------------------------------------
        # COLUMN 2
        # --------------------------------------------------

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


        # --------------------------------------------------
        # COLUMN 3
        # --------------------------------------------------

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


            total_claim_amount = st.number_input(
                "Total Claim Amount",
                min_value=0.0,
                value=10000.0,
            )


            avg_claim_amount = st.number_input(
                "Average Claim Amount",
                min_value=0.0,
                value=10000.0,
            )


            max_claim_amount = st.number_input(
                "Maximum Claim Amount",
                min_value=0.0,
                value=10000.0,
            )


        st.subheader(
            "Customer & Policy Profile"
        )


        c4, c5, c6 = st.columns(3)


        with c4:

            gender = st.selectbox(
                "Gender",
                GENDER_OPTIONS,
                index=2,
            )


            marital_status = st.selectbox(
                "Marital Status",
                MARITAL_STATUS_OPTIONS,
                index=0,
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


            customer_risk = st.selectbox(
                "Customer Risk Segment",
                CUSTOMER_RISK_OPTIONS,
                index=1,
            )


            policy_type = st.selectbox(
                "Policy Type",
                POLICY_TYPE_OPTIONS,
                index=5,
            )


        with c6:

            sales_channel = st.selectbox(
                "Sales Channel",
                SALES_CHANNEL_OPTIONS,
                index=1,
            )


            payment_mode = st.selectbox(
                "Payment Mode",
                PAYMENT_MODE_OPTIONS,
                index=0,
            )


            risk_band = st.selectbox(
                "Risk Band",
                RISK_BAND_OPTIONS,
                index=0,
            )


        renewal_submit = st.form_submit_button(
            "Analyze Renewal Risk",
            type="primary",
            use_container_width=True,
        )


    # ------------------------------------------------------
    # RENEWAL PREDICTION
    # ------------------------------------------------------

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


            renewal_probability = result.get(
                "renewal_probability",
                0,
            )


            churn_probability = result.get(
                "churn_probability",
                0,
            )


            threshold = result.get(
                "threshold",
                0,
            )


            prediction = result.get(
                "prediction",
                "Unavailable",
            )


            m1, m2, m3 = st.columns(3)


            m1.metric(
                "Renewal Probability",
                f"{renewal_probability * 100:.1f}%",
            )


            m2.metric(
                "Churn Probability",
                f"{churn_probability * 100:.1f}%",
            )


            m3.metric(
                "Decision Threshold",
                f"{threshold * 100:.1f}%",
            )


            st.markdown(
                "#### Renewal Confidence"
            )


            st.progress(
                min(
                    max(
                        float(
                            renewal_probability
                        ),
                        0.0,
                    ),
                    1.0,
                )
            )


            if churn_probability >= threshold:

                st.error(
                    f"⚠ {prediction}"
                )

            else:

                st.success(
                    f"✓ {prediction}"
                )


            with st.expander(
                "Technical Model Output"
            ):

                st.json(
                    result
                )


# ==========================================================
# FRAUD INTELLIGENCE PAGE
# ==========================================================

elif page == "Fraud Intelligence":

    render_html(
        """
        <div class="page-label">
            CLAIM INVESTIGATION
        </div>
        <div class="page-title">
            Fraud Intelligence
        </div>
        <div class="page-description">
            Generate a fraud-risk signal to prioritize
            claims for additional investigation.
        </div>
        """
    )


    st.warning(
        "A risk flag is not a fraud determination. "
        "Human investigation is required."
    )


    with st.form(
        "fraud_form"
    ):

        st.subheader(
            "Claim Profile"
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            claim_amount = st.number_input(
                "Claim Amount",
                min_value=0.0,
                value=50000.0,
            )


            reporting_delay = st.number_input(
                "Reporting Delay Days",
                min_value=0.0,
                value=2.0,
            )


            incident_month = st.number_input(
                "Incident Month",
                min_value=1,
                max_value=12,
                value=6,
            )


        with c2:

            fraud_age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=100,
                value=35,
            )


            fraud_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=600000.0,
            )


            fraud_credit = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=750,
            )


        with c3:

            fraud_sum_insured = st.number_input(
                "Sum Insured",
                min_value=0.0,
                value=500000.0,
            )


            fraud_premium = st.number_input(
                "Annual Premium",
                min_value=0.0,
                value=20000.0,
            )


            fraud_risk_score = st.number_input(
                "Risk Score",
                min_value=0.0,
                value=50.0,
            )


        st.subheader(
            "Claim & Customer Details"
        )


        c4, c5, c6 = st.columns(3)


        with c4:

            claim_type = st.selectbox(
                "Claim Type",
                CLAIM_TYPE_OPTIONS,
                index=6,
            )


            claim_source = st.selectbox(
                "Claim Source",
                CLAIM_SOURCE_OPTIONS,
                index=5,
            )


            claim_severity = st.selectbox(
                "Claim Severity",
                CLAIM_SEVERITY_OPTIONS,
                index=0,
            )


            fraud_gender = st.selectbox(
                "Gender",
                GENDER_OPTIONS,
                index=2,
            )


        with c5:

            fraud_marital = st.selectbox(
                "Marital Status",
                MARITAL_STATUS_OPTIONS,
                index=0,
            )


            fraud_occupation = st.selectbox(
                "Occupation",
                OCCUPATION_OPTIONS,
                index=1,
            )


            fraud_state = st.selectbox(
                "State",
                STATE_OPTIONS,
                index=6,
            )


            fraud_customer_risk = st.selectbox(
                "Customer Risk Segment",
                CUSTOMER_RISK_OPTIONS,
                index=1,
            )


        with c6:

            fraud_policy_type = st.selectbox(
                "Policy Type",
                POLICY_TYPE_OPTIONS,
                index=5,
            )


            fraud_payment_mode = st.selectbox(
                "Payment Mode",
                PAYMENT_MODE_OPTIONS,
                index=0,
            )


            fraud_risk_band = st.selectbox(
                "Risk Band",
                RISK_BAND_OPTIONS,
                index=0,
            )


        fraud_submit = st.form_submit_button(
            "Analyze Claim Risk",
            type="primary",
            use_container_width=True,
        )


    # ------------------------------------------------------
    # FRAUD PREDICTION
    # ------------------------------------------------------

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


            fraud_probability = result.get(
                "fraud_probability",
                0,
            )


            threshold = result.get(
                "threshold",
                0,
            )


            prediction = result.get(
                "prediction",
                "Unavailable",
            )


            m1, m2 = st.columns(2)


            m1.metric(
                "Fraud Risk Probability",
                f"{fraud_probability * 100:.1f}%",
            )


            m2.metric(
                "Investigation Threshold",
                f"{threshold * 100:.1f}%",
            )


            st.markdown(
                "#### Risk Signal"
            )


            st.progress(
                min(
                    max(
                        float(
                            fraud_probability
                        ),
                        0.0,
                    ),
                    1.0,
                )
            )


            if fraud_probability >= threshold:

                st.error(
                    "⚠ Claim flagged for "
                    "additional investigation."
                )

            else:

                st.success(
                    "✓ No elevated fraud-risk "
                    "signal detected."
                )


            st.caption(
                "This output is a prioritization signal. "
                "It does not establish that fraud occurred."
            )


            with st.expander(
                "Technical Model Output"
            ):

                st.json(
                    result
                )


# ==========================================================
# UNDERWRITING PAGE
# ==========================================================

elif page == "Underwriting":

    render_html(
        """
        <div class="page-label">
            RISK ASSESSMENT
        </div>
        <div class="page-title">
            Underwriting Support
        </div>
        <div class="page-description">
            Analyze applicant characteristics and generate
            model-based signals for qualified human review.
        </div>
        """
    )


    st.warning(
        "Final underwriting decisions require "
        "qualified human review."
    )


    with st.form(
        "underwriting_form"
    ):

        left, right = st.columns(2)


        with left:

            uw_age = st.number_input(
                "Applicant Age",
                min_value=18,
                max_value=100,
                value=35,
            )


            health_score = st.number_input(
                "Health Score",
                min_value=0.0,
                value=75.0,
            )


            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=24.5,
            )


            uw_credit = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=750,
            )


        with right:

            lifestyle = st.selectbox(
                "Lifestyle",
                LIFESTYLE_OPTIONS,
                index=0,
            )


            medical_history = st.selectbox(
                "Medical History",
                YES_NO_OPTIONS,
                index=0,
            )


            smoker = st.selectbox(
                "Smoker",
                YES_NO_OPTIONS,
                index=0,
            )


            occupation_risk = st.selectbox(
                "Occupation Risk",
                OCCUPATION_RISK_OPTIONS,
                index=1,
            )


        underwriting_submit = st.form_submit_button(
            "Run Assessment",
            type="primary",
            use_container_width=True,
        )


    # ------------------------------------------------------
    # UNDERWRITING PREDICTION
    # ------------------------------------------------------

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


            decision = result.get(
                "decision",
                "Unavailable",
            )


            st.info(
                f"Model Recommendation: **{decision}**"
            )


            probabilities = result.get(
                "probabilities",
                {},
            )


            if probabilities:

                st.markdown(
                    "### Class Probabilities"
                )


                columns = st.columns(
                    len(probabilities)
                )


                for index, (
                    class_name,
                    probability,
                ) in enumerate(
                    probabilities.items()
                ):

                    columns[index].metric(
                        class_name,
                        f"{probability * 100:.1f}%",
                    )


            st.caption(
                "The model provides decision-support signals only. "
                "It must not be used as an autonomous "
                "insurance eligibility decision."
            )


            with st.expander(
                "Technical Model Output"
            ):

                st.json(
                    result
                )


# ==========================================================
# FOOTER
# ==========================================================

render_html(
    """
    <div class="footer">
        INSURE AI • AI-powered Insurance Intelligence •
        Secure • Explainable • Human-in-the-loop
    </div>
    """
)