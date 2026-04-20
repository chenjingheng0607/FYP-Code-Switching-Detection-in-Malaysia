import json
import streamlit as st
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from db import save_chat, get_conn
from login import show_login
from models import load_model, predict


LOOKUP_FILE = "dashboard_lookup_5000.jsonl"


def normalize_text(text):
    text = text.strip().lower()

    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # remove extra spaces
    text = " ".join(text.split())

    return text

@st.cache_resource
def load_ground_truth_lookup():
    lookup = {}
    try:
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = normalize_text(" ".join(item["tokens"]))
                labels = item["ner_tags"]    
                lookup[text] = labels
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return lookup


def compute_sentence_metrics(true_labels, pred_labels):
    min_len = min(len(true_labels), len(pred_labels))
    y_true = true_labels[:min_len]
    y_pred = pred_labels[:min_len]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def render_message(role: str, msg: str, sender_name: str):
    bubble = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(
        f"""
        <div class="message-wrap">
            <div class="message-label">{sender_name}</div>
            <div class="{bubble}">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_prediction(tokens, labels):
    highlighted_html = ""
    for t, l in zip(tokens, labels):
        if l == 1:
            highlighted_html += f"<span class='tok-ms'>{t}(MS)</span> "
        elif l == 2:
            highlighted_html += f"<span class='tok-en'>{t}(EN)</span> "
        else:
            highlighted_html += f"{t} "
    return highlighted_html.strip()


st.set_page_config(
    page_title="Code-Switching Detection Dashboard",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --border: #e6ebf2;
        --text: #18212f;
        --muted: #667085;
        --primary: #1d4ed8;
        --primary-dark: #1e3a8a;
        --primary-soft: #e8f0ff;
        --success: #0f766e;
        --danger: #b42318;
        --user: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        --assistant: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        --history: #f8fafc;
        --ms: #0f766e;
        --en: #c2410c;
        --shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        --radius: 18px;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 3.8rem;
        padding-bottom: 1.5rem;
        max-width: 1600px;
    }

    h1, h2, h3, h4, h5, h6 {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    div[data-testid="stVerticalBlock"] > div {
        gap: 0.7rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        color: white;
        border-radius: 24px;
        padding: 24px 28px;
        box-shadow: 0 18px 40px rgba(29, 78, 216, 0.18);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: rgba(255,255,255,0.88);
        line-height: 1.5;
    }

    .panel-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 18px;
        margin-bottom: 0.4rem;
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 0.28rem 0;
        line-height: 1.25;
    }

    .panel-subtitle {
        font-size: 0.9rem;
        color: var(--muted);
        margin: 0 0 0.8rem 0;
        line-height: 1.45;
    }

    .history-box {
        max-height: 72vh;
        overflow-y: auto;
        padding-right: 0.25rem;
    }

    .history-box::-webkit-scrollbar {
        width: 8px;
    }

    .history-box::-webkit-scrollbar-thumb {
        background: #d0d7e2;
        border-radius: 999px;
    }

    .message-wrap {
        margin-bottom: 0.9rem;
    }

    .message-label {
        font-size: 0.78rem;
        color: var(--muted);
        font-weight: 600;
        margin-bottom: 0.3rem;
        padding-left: 0.1rem;
    }

    .user-bubble {
        background: var(--user);
        padding: 14px 16px;
        border-radius: 16px 16px 6px 16px;
        margin-bottom: 0.35rem;
        border: 1px solid #dbeafe;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
        line-height: 1.55;
        word-wrap: break-word;
    }

    .assistant-bubble {
        background: var(--assistant);
        padding: 14px 16px;
        border-radius: 16px 16px 16px 6px;
        margin-bottom: 0.35rem;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        line-height: 1.65;
        word-wrap: break-word;
    }

    .status-pill {
        display: inline-block;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0.15rem 0.35rem 0.15rem 0;
    }

    .legend-row {
        margin-top: 0.2rem;
        margin-bottom: 0.15rem;
    }

    .tok-en {
        color: var(--en);
        font-weight: 700;
    }

    .tok-ms {
        color: var(--ms);
        font-weight: 700;
    }

    .mini-note {
        color: var(--muted);
        font-size: 0.85rem;
        line-height: 1.5;
    }

    .user-card {
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
        border: 1px solid #dbe7ff;
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 0.65rem;
    }

    .user-card-label {
        font-size: 0.76rem;
        color: var(--muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.18rem;
    }

    .user-card-name {
        font-size: 1rem;
        color: var(--text);
        font-weight: 700;
        line-height: 1.35;
    }

    .stMetric {
        background: #fbfdff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {
        color: #18212f !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] p {
        color: #18212f !important;
        font-size: 1.45rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    .stChatInput {
        position: sticky;
        bottom: 0;
    }

    div.stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.18s ease;
        box-shadow: 0 8px 18px rgba(29, 78, 216, 0.18);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        color: white;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(29, 78, 216, 0.24);
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        color: white;
    }

    div.stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 0.2rem rgba(59, 130, 246, 0.22);
        color: white;
    }

    div[data-testid="stSelectbox"] label {
        font-weight: 600;
    }

    html, body, [class*="css"]  {
    color: #18212f !important;
    }

    /* ===== 修复 Metrics 颜色 ===== */

    div[data-testid="stMetricLabel"] {
        color: #18212f !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #18212f !important;
        font-size: 1.45rem !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #667085 !important;
    }

    /* ===== 修复 Selectbox label ===== */

    div[data-testid="stSelectbox"] label {
        color: #18212f !important;
        opacity: 1 !important;
    }

    /* ===== 修复 Tabs（Login/Register） ===== */

    .stTabs [data-baseweb="tab"] {
        color: #18212f !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* 当前选中的 tab */

    .stTabs [aria-selected="true"] {
        color: #1d4ed8 !important;
    }

    /* ===== 修复 TextInput label ===== */

    .stTextInput label {
        color: #18212f !important;
    }

    /* ===== 修复 输入框文字 ===== */

    div[data-baseweb="input"] input {
        color: #18212f !important;
        background-color: #ffffff !important;
    }

    /* ===== 修复 Button 文字 ===== */

    button {
        color: white !important;
    }

    /* ===== 修复 Login 页面标题 ===== */

    h1 {
        color: #18212f !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


if "page" not in st.session_state:
    st.session_state.page = "main"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "Accuracy": 0.00,
        "Precision": 0.00,
        "Recall": 0.00,
        "F1": 0.00,
    }

if "metric_source" not in st.session_state:
    st.session_state.metric_source = "No input yet"

if "current_model" not in st.session_state:
    st.session_state.current_model = "MT5"

if "last_prediction_html" not in st.session_state:
    st.session_state.last_prediction_html = None


ground_truth_lookup = load_ground_truth_lookup()


if st.session_state.page == "login":
    show_login()
    st.stop()


model_options = ["MT5", "mBERT", "XLM-R"]
if (
    "tokenizer" not in st.session_state
    or "model" not in st.session_state
    or st.session_state.current_model not in model_options
):
    tokenizer, model = load_model("MT5")
    st.session_state.tokenizer = tokenizer
    st.session_state.model = model
    st.session_state.current_model = "MT5"


st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">Malay-English Code-Switching Detection Dashboard</div>
        <div class="hero-subtitle">
            A professional interface for multilingual text analysis using transformer-based models.
            Select a model, enter a sentence, and review token-level language tagging with live performance cards.
        </div>
        <div style="margin-top:0.9rem;">
            <span class="status-pill">Active Model: {st.session_state.current_model}</span>
            <span class="status-pill">Token-Level Tagging</span>
            <span class="status-pill">Streamlit Interface</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


col_history, col_center, col_ctrl = st.columns([2.1, 5.7, 2.2], gap="large")


with col_history:

    st.markdown('<div class="panel-title">User Session</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Authentication and account access.</div>', unsafe_allow_html=True)

    if "user" not in st.session_state:
        st.markdown(
            '<div class="mini-note">You are currently browsing as a guest.</div>',
            unsafe_allow_html=True
        )

        if st.button("🔐 Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

    else:
        st.markdown(
            f"""
            <div class="user-card">
                <div class="user-card-label">Logged in as</div>
                <div class="user-card-name">👤 {st.session_state.user}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.pop("user", None)
            st.session_state.page = "main"
            st.rerun()

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Conversation History</div>
            <div class="panel-subtitle">Stored interactions for the current user account.</div>
            <div class="history-box">
        """,
        unsafe_allow_html=True,
    )

    if "user" in st.session_state:
        conn = get_conn()
        rows = conn.execute(
            "SELECT role, message FROM chats WHERE username=? ORDER BY id DESC",
            (st.session_state.user,),
        ).fetchall()

        if rows:
            for role, msg in rows:
                bubble = "user-bubble" if role == "user" else "assistant-bubble"
                sender = "User" if role == "user" else "System"

                st.markdown(
                    f"""
                    <div class="message-wrap">
                        <div class="message-label">{sender}</div>
                        <div class="{bubble}">{msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="mini-note">No saved history found yet.</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="mini-note">Login to view and store chat history.</div>',
            unsafe_allow_html=True
        )

    st.markdown("</div></div>", unsafe_allow_html=True)


with col_center:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Analysis Workspace</div>
            <div class="panel-subtitle">Enter a sentence to detect Malay-English code-switching patterns.</div>
            <div class="legend-row">
                <span class="status-pill">MS = Malay</span>
                <span class="status-pill">EN = English</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.last_prediction_html is not None:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Latest Prediction Output</div>
                <div class="panel-subtitle">Most recent token-level language tagging result.</div>
            """,
            unsafe_allow_html=True,
        )
        render_message("assistant", st.session_state.last_prediction_html, f"{st.session_state.current_model} Output")
        st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Enter a sentence for language tagging...")

if user_input:

    clean_input = normalize_text(user_input)

    result = predict(
        st.session_state.current_model,
        st.session_state.tokenizer,
        st.session_state.model,
        clean_input
    )

    if len(result) == 4:
        tokens, labels, benchmark_metrics, _ = result
    else:
        tokens, labels, benchmark_metrics = result

    normalized_input = normalize_text(user_input)

    if normalized_input in ground_truth_lookup:
        true_labels = ground_truth_lookup[normalized_input]
        dynamic_metrics = compute_sentence_metrics(true_labels, labels)
        st.session_state.metric_source = "Ground truth matched: real-time sentence metrics"
    else:
        dynamic_metrics = benchmark_metrics
        st.session_state.metric_source = "Ground truth not found: fallback to model metrics"

    st.session_state.metrics = dynamic_metrics

    highlighted_html = format_prediction(tokens, labels)
    st.session_state.last_prediction_html = highlighted_html

    st.session_state.chat_history.insert(0, ("user", user_input))
    st.session_state.chat_history.insert(0, ("assistant", highlighted_html))

    if "user" in st.session_state:
        save_chat(st.session_state.user, "user", user_input)
        save_chat(st.session_state.user, "assistant", highlighted_html)

    st.rerun()


with col_ctrl:

    st.markdown('<div class="panel-title">Model Control Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Choose a pretrained model for prediction.</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose Model",
        model_options,
        index=model_options.index(st.session_state.current_model),
    )

    if st.session_state.current_model != model_choice:
        with st.spinner(f"Loading {model_choice} model..."):
            tokenizer, model = load_model(model_choice)
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.current_model = model_choice
        st.rerun()

    st.markdown(
        f'<div class="mini-note">Current selection: <b>{st.session_state.current_model}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">Performance Metrics</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="panel-subtitle">Latest metrics based on current sentence analysis.</div>',
        unsafe_allow_html=True
    )

    m = st.session_state.metrics
    st.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
    st.metric("Precision", f"{m['Precision']*100:.1f}%")
    st.metric("Recall", f"{m['Recall']*100:.1f}%")
    st.metric("F1 Score", f"{m['F1']*100:.1f}%")

    st.markdown(
        f"<div class='mini-note'>{st.session_state.metric_source}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">Interface Notes</div>
            <div class="mini-note">
                This refreshed UI uses card-based sections, clearer hierarchy,
                softer colors, and consistent typography to make the dashboard
                more professional for demos and reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
