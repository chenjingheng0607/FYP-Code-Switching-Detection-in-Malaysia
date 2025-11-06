import streamlit as st
from db import check_user, add_user

def show_login():
    st.title("Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # -------- LOGIN --------
    with tab1:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login Now"):
            if check_user(login_user, login_pass):
                st.session_state.page = "main"
                st.session_state.user = login_user
                st.success("Login Success ✅")
                st.rerun()
            else:
                st.error("Invalid username or password ❌")

    # -------- REGISTER --------
    with tab2:
        reg_user = st.text_input("New Username", key="reg_user")
        reg_pass = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Create Account"):
            if reg_user.strip()=="" or reg_pass.strip()=="":
                st.error("Both fields required!")
            else:
                add_user(reg_user, reg_pass)
                st.success("Account created. Now login!")