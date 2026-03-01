import streamlit as st
from db import save_chat, get_conn
from login import show_login
from models import load_model, predict

st.set_page_config(page_title="Malay-English Code Switching", layout="wide")

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
.history-box {
    max-height: 80vh;
    overflow-y: auto;
    padding-right: 12px;
    border-right: 2px solid #e0e0e0;
    background: #fafafa;
}
.user-bubble {
    background-color:#DCF8C6;
    padding:12px 20px;
    border-radius:14px;
    margin-bottom:10px;
}
.assistant-bubble {
    background-color:#F0F0F0;
    padding:12px 20px;
    border-radius:14px;
    margin-bottom:10px;
}
.tok-en { color:#e07b00; font-weight:600; }
.tok-ms { color:#00897b; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session Init
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Login Routing
# -------------------------
if st.session_state.page == "login":
    show_login()
    st.stop()

# -------------------------
# Layout
# -------------------------
col_history, col_center, col_ctrl = st.columns([1.5,6,2.5])

# -------------------------
# LEFT - History
# -------------------------
with col_history:
    st.markdown("<div class='history-box'>", unsafe_allow_html=True)
    st.header("History")
    if "user" in st.session_state:
        conn = get_conn()
        rows = conn.execute(
            "SELECT role, message FROM chats WHERE username=? ORDER BY id DESC",
            (st.session_state.user,)
        ).fetchall()
        for role, msg in rows:
            bubble = "user-bubble" if role=="user" else "assistant-bubble"
            st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.write("Login to view history.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# CENTER - Chat
# -------------------------
with col_center:
    st.title("Malay-English Code Switching Chat")

    user_input = st.chat_input("Type your sentence...")

    if user_input:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model

        tokens, labels = predict(tokenizer, model, user_input)

        highlighted_html = ""
        for t, l in zip(tokens, labels):
            if l == 1:
                highlighted_html += f"<span class='tok-ms'>{t}(MS)</span> "
            elif l == 2:
                highlighted_html += f"<span class='tok-en'>{t}(EN)</span> "
            else:
                highlighted_html += f"{t} "

        st.session_state.chat_history.insert(0, ("user", user_input))
        st.session_state.chat_history.insert(0, ("assistant", highlighted_html))

        if "user" in st.session_state:
            save_chat(st.session_state.user, "user", user_input)
            save_chat(st.session_state.user, "assistant", highlighted_html)

        st.rerun()

    for role, msg in st.session_state.chat_history:
        bubble = "user-bubble" if role=="user" else "assistant-bubble"
        st.markdown(f"<div class='{bubble}'>{msg}</div>", unsafe_allow_html=True)

# -------------------------
# RIGHT - Model Control
# -------------------------
with col_ctrl:

    # Login / Logout
    if "user" not in st.session_state:
        if st.button("🔐 Login"):
            st.session_state.page = "login"
            st.rerun()
    else:
        st.write(f"👤 {st.session_state.user}")
        if st.button("Logout"):
            st.session_state.pop("user", None)
            st.session_state.page = "main"
            st.rerun()

    st.header("Model & Metrics")

    model_choice = st.selectbox("Choose Model:", ["mBERT", "XLM-R"])

    if "current_model" not in st.session_state or st.session_state.current_model != model_choice:
        with st.spinner("Loading model..."):
            tokenizer, model = load_model(model_choice)
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.current_model = model_choice

    MODEL_METRICS = {
        "mBERT": {"Accuracy":0.82, "Precision":0.80, "Recall":0.79, "F1":0.79},
        "XLM-R": {"Accuracy":0.88, "Precision":0.86, "Recall":0.85, "F1":0.85},
    }

    m = MODEL_METRICS[model_choice]

    st.metric("Accuracy",  f"{m['Accuracy']*100:.1f}%")
    st.metric("Precision", f"{m['Precision']*100:.1f}%")
    st.metric("Recall",    f"{m['Recall']*100:.1f}%")
    st.metric("F1 Score",  f"{m['F1']*100:.1f}%")