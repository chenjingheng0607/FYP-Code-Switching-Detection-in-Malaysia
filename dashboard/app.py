import streamlit as st
from translator import ms_to_en, en_to_ms
from db import save_chat
from login import show_login
from db import get_conn

# ==========================
# CSS
# ==========================

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

    /* make scrollbar nice */
    .history-box::-webkit-scrollbar {
        width: 6px;
    }
    .history-box::-webkit-scrollbar-thumb {
        background: #c7c7c7;
        border-radius: 4px;
    }

    </style>
    """, unsafe_allow_html=True)

def chat_css():
    st.markdown("""
    <style>
    .user-bubble {
        background-color:#DCF8C6;
        color:black;
        padding:12px 20px;
        border-radius:14px;
        margin-bottom:10px;
        width:100%;
        display:block;
    }
    .assistant-bubble {
        background-color:#F0F0F0;
        color:black;
        padding:12px 20px;
        border-radius:14px;
        margin-bottom:10px;
        width:100%;
        display:block;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================
# ratio detector
# ==========================
ENGLISH_WORDS = set(["today","go","eat","yes","no","you","i","me","he","she","it","they","free","tonight","dinner","want","later","tomorrow"])
MALAY_WORDS   = set(["nanti","pergi","makan","tak","boleh","kau","dia","kita","sini","itu","ini","lagi","lah","pun","kan","macam","lepas","rumah"])

def compute_ratio(text):
    if not text or not isinstance(text,str): return 0,0
    words = ["".join(c for c in w.lower() if c.isalpha()) for w in text.split()]
    en=0; ms=0
    for w in words:
        if not w: continue
        if w in MALAY_WORDS: ms+=1; continue
        if w in ENGLISH_WORDS: en+=1; continue
        if w[-1] in "aiueo": ms+=1
        else: en+=1
    total=en+ms
    if total==0: return 0,0
    return en/total, ms/total


# ==========================
# apply CSS
chat_css()
layout_css()

# then set config
st.set_page_config(page_title="Malay-English Code Switching", layout="wide")
if "direction" not in st.session_state: st.session_state.direction="English"

if "page" not in st.session_state: st.session_state.page="main"
if "chat_history" not in st.session_state: st.session_state.chat_history=[]

def goto_login(): st.session_state.page="login"

if st.session_state.page=="login":
    show_login()
    st.stop()

MODEL_METRICS={"mBERT":{"Accuracy":0.76,"Precision":0.72,"Recall":0.70,"F1":0.71},
               "XLM-R":{"Accuracy":0.82,"Precision":0.80,"Recall":0.77,"F1":0.78},
               "mT5":{"Accuracy":0.88,"Precision":0.86,"Recall":0.84,"F1":0.85}}

top_right = st.container()
with top_right:
    if "user" in st.session_state:
        st.write(f"👤 {st.session_state.user}")
        if st.button("Logout"):
            del st.session_state["user"]
            st.session_state.page="main"
            st.rerun()
    else:
        st.button("Login", on_click=goto_login)

col_history, col_center, col_ctrl = st.columns([1.5,6,2.5], gap="large")

# left column DB history
with col_history:
    st.markdown("<div class='history-box'>", unsafe_allow_html=True)
    st.header("History")

    if "user" in st.session_state:
        conn = get_conn()
        rows = conn.execute("SELECT role, message FROM chats WHERE username=? ORDER BY id DESC",(st.session_state.user,)).fetchall()
        for role,msg in rows:
            bubble_class = "user-bubble" if role=="user" else "assistant-bubble"
            st.markdown(f"<div class='{bubble_class}'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.write("Login to view your history.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_center:
    st.title("Malay-English Code Switching Chat")

    user_input=st.chat_input("Type your sentence...")

    if user_input:
        if st.session_state.direction=="English":
            translated = ms_to_en(user_input)
        else:
            translated = en_to_ms(user_input)

        st.session_state.chat_history.insert(0, ("assistant", translated))
        st.session_state.chat_history.insert(0, ("user", user_input))

        if "user" in st.session_state:
            save_chat(st.session_state.user, "user", user_input)
            save_chat(st.session_state.user, "assistant", translated)

        st.rerun()

    # display current chat history (AFTER input, so it sits above)
    for role,msg in st.session_state.chat_history:
        bubble_class = "user-bubble" if role=="user" else "assistant-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg}</div>", unsafe_allow_html=True)

# right column (controls)
with col_ctrl:
    st.header("Model & Metrics")
    model_choice=st.selectbox("Choose Model:",["mBERT","XLM-R","mT5"])
    direction = st.radio("Translate to:", ["English","Malay"], key="direction")
    st.subheader("Evaluation Metrics")
    m=MODEL_METRICS[model_choice]
    st.metric("Accuracy",f"{m['Accuracy']*100:.1f}%")
    st.metric("Precision",f"{m['Precision']*100:.1f}%")
    st.metric("Recall",f"{m['Recall']*100:.1f}%")
    st.metric("F1 Score",f"{m['F1']*100:.1f}%")

    st.subheader("Code-Switch Ratio")

    user_msgs=[h[1] for h in st.session_state.chat_history if h[0]=="user"]
    if user_msgs:
        last=user_msgs[0]
        en_ratio, ms_ratio=compute_ratio(last)
        st.write(f"English Ratio: {en_ratio*100:.1f}%")
        st.write(f"Malay Ratio:   {ms_ratio*100:.1f}%")
    else:
        st.write("No input yet")
    

