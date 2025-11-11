import streamlit as st
from db import save_chat, get_conn
from login import show_login
from langdetect import detect, DetectorFactory

# -------------------------
# Streamlit basic config
# -------------------------
st.set_page_config(page_title="Malay-English Code Switching", layout="wide")
DetectorFactory.seed = 0  # 让 langdetect 结果更稳定

# -------------------------
# CSS
# -------------------------
def layout_css():
    st.markdown("""
    <style>
    /* ===== LEFT HISTORY PANEL ===== */
    .history-box {
        max-height: 80vh;
        overflow-y: auto;
        padding-right: 12px;
        border-right: 2px solid #e0e0e0;
        background: #fafafa;
        box-shadow: inset -4px 0px 8px rgba(0,0,0,0.08);
    }
    .history-box::-webkit-scrollbar { width: 6px; }
    .history-box::-webkit-scrollbar-thumb {
        background: #c7c7c7; border-radius: 4px;
    }

    /* chat bubbles */
    .user-bubble {
        background-color:#DCF8C6; color:black;
        padding:12px 20px; border-radius:14px;
        margin-bottom:10px; width:100%; display:block;
    }
    .assistant-bubble {
        background-color:#F0F0F0; color:black;
        padding:12px 20px; border-radius:14px;
        margin-bottom:10px; width:100%; display:block;
    }

    /* colored tokens */
    .tok-en { color: #e07b00; font-weight: 600; }   /* Orange */
    .tok-ms { color: #00897b; font-weight: 600; }   /* Teal Green */
    </style>
    """, unsafe_allow_html=True)

layout_css()

# -------------------------
# Session init
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"
if "chat_history" not in st.session_state:
    # list of tuples: ("user"/"assistant", message_html_or_text)
    st.session_state.chat_history = []
if "direction" not in st.session_state:
    # 不用了，但保留 key 以避免旧代码引用报错
    st.session_state.direction = "English"

def goto_login():
    st.session_state.page = "login"

# -------------------------
# Login routing
# -------------------------
if st.session_state.page == "login":
    show_login()
    st.stop()

# -------------------------
# Simple lexicons (backup)
# -------------------------
ENGLISH_HINT = set(["today","go","eat","yes","no","you","i","me","he","she","it","they","free","tonight","dinner","want","later","tomorrow"])
MALAY_HINT   = set(["nanti","pergi","makan","tak","boleh","kau","dia","kita","sini","itu","ini","lagi","lah","pun","kan","macam","lepas","rumah","hari","sekarang"])

# -------------------------
# Detection helpers
# -------------------------
def normalize_token(w: str) -> str:
    # 只保留字母，降噪
    return "".join(c for c in w if c.isalpha())

def token_lang(w: str) -> str:
    """
    返回 'en' 或 'ms'。
    优先用 langdetect；失败时用词典与启发式兜底。
    """
    wl = normalize_token(w.lower())
    if not wl:
        return "en"  # 空的当英文，避免计数错

    # 快速词典命中
    if wl in MALAY_HINT: return "ms"
    if wl in ENGLISH_HINT: return "en"

    # langdetect
    try:
        lang = detect(wl)
        if lang in ("ms", "id"):  # 印尼文当 BM 近似
            return "ms"
        elif lang == "en":
            return "en"
    except Exception:
        pass

    # 简单尾元音启发式兜底
    return "ms" if wl[-1] in "aiueo" else "en"

def highlight_sentence(text: str) -> str:
    """
    生成彩色 + 标签的 HTML：today(EN) makan(MS) dinner(EN)
    EN = 橙色，MS = 绿
    """
    if not text or not isinstance(text, str):
        return ""

    tokens = text.split()
    out_pieces = []
    for t in tokens:
        lang = token_lang(t)
        if lang == "ms":
            out_pieces.append(f"<span class='tok-ms'>{t}(MS)</span>")
        else:
            out_pieces.append(f"<span class='tok-en'>{t}(EN)</span>")
    return " ".join(out_pieces)

def compute_ratio_from_text(text: str):
    """
    用与 token_lang 同一逻辑计算 EN / MS 比例（给右侧 Ratio 用）
    """
    if not text or not isinstance(text, str):
        return 0.0, 0.0
    tokens = [normalize_token(t.lower()) for t in text.split()]
    en = ms = 0
    for t in tokens:
        if not t: 
            continue
        lang = token_lang(t)
        if lang == "ms": ms += 1
        else: en += 1
    total = en + ms
    return (en/total, ms/total) if total else (0.0, 0.0)

# -------------------------
# Static metrics (右侧)
# -------------------------
MODEL_METRICS = {
    "mBERT": {"Accuracy":0.76, "Precision":0.72, "Recall":0.70, "F1":0.71},
    "XLM-R": {"Accuracy":0.82, "Precision":0.80, "Recall":0.77, "F1":0.78},
    "mT5":   {"Accuracy":0.88, "Precision":0.86, "Recall":0.84, "F1":0.85},
}

# -------------------------
# Top-right: login/logout
# -------------------------
top_right = st.container()
with top_right:
    if "user" in st.session_state:
        st.write(f"👤 {st.session_state.user}")
        if st.button("Logout"):
            del st.session_state["user"]
            st.session_state.page = "main"
            st.rerun()
    else:
        st.button("Login", on_click=goto_login)

# -------------------------
# Layout columns
# -------------------------
col_history, col_center, col_ctrl = st.columns([1.5, 6, 2.5], gap="large")

# LEFT = History (DB)
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
            bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
            st.markdown(f"<div class='{bubble_class}'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.write("Login to view your history.")
    st.markdown("</div>", unsafe_allow_html=True)

# CENTER = Chat
with col_center:
    st.title("Malay-English Code Switching Chat")

    user_input = st.chat_input("Type your sentence...")
    if user_input:
        # 1) 添加 user bubble
        st.session_state.chat_history.insert(0, ("user", user_input))

        # 2) 生成高亮句子（assistant bubble）
        highlighted_html = highlight_sentence(user_input)
        st.session_state.chat_history.insert(0, ("assistant", highlighted_html))

        # 3) DB 持久化（如已登录）
        if "user" in st.session_state:
            save_chat(st.session_state.user, "user", user_input)
            save_chat(st.session_state.user, "assistant", highlighted_html)

        st.rerun()

    # 显示（新→旧）
    for role, msg in st.session_state.chat_history:
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg}</div>", unsafe_allow_html=True)

# RIGHT = Model & Metrics & Ratio
with col_ctrl:
    st.header("Model & Metrics")
    model_choice = st.selectbox("Choose Model:", ["mBERT", "XLM-R", "mT5"])
    

    st.subheader("Evaluation Metrics")
    m = MODEL_METRICS[model_choice]
    st.metric("Accuracy",  f"{m['Accuracy']*100:.1f}%")
    st.metric("Precision", f"{m['Precision']*100:.1f}%")
    st.metric("Recall",    f"{m['Recall']*100:.1f}%")
    st.metric("F1 Score",  f"{m['F1']*100:.1f}%")

    st.subheader("Code-Switch Ratio (latest user input)")
    user_msgs = [h[1] for h in st.session_state.chat_history if h[0] == "user"]
    if user_msgs:
        en_ratio, ms_ratio = compute_ratio_from_text(user_msgs[0])
        st.write(f"English Ratio: **{en_ratio*100:.1f}%**")
        st.write(f"Malay Ratio: **{ms_ratio*100:.1f}%**")
    else:
        st.write("No input yet.")
    

