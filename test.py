# app.py
import streamlit as st
from predictor import predict_code_switching, format_prediction_as_html

# --- Page Configuration ---
st.set_page_config(
    page_title="Malay-English Code-Switching Detector",
    layout="wide"
)

# --- CSS Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
}
.result-box {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    font-size: 1.2rem;
    line-height: 2.5;
}
</style>
""", unsafe_allow_html=True)

# --- Main App UI ---

st.markdown("<h1 class='main-header'>Malay-English Code-Switching Detector</h1>", unsafe_allow_html=True)

# Initialize session state to store history
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Input Form ---
# Use a form to group the text input and button together
with st.form("input_form"):
    user_input = st.text_input(
        "Enter a sentence to analyze:",
        placeholder="e.g., Geng yang kena PKPD ni jangan putus asa...",
        key="input_box"
    )
    submitted = st.form_submit_button("Analyze")

# This block now only runs when the "Analyze" button is clicked
if submitted and user_input:
    # Get predictions from our model
    predictions = predict_code_switching(user_input)
    
    # Format the predictions as a color-coded HTML string
    formatted_html = format_prediction_as_html(predictions)
    
    # Add the result to the top of our history
    st.session_state.history.insert(0, (user_input, formatted_html))
    
    # We don't need to use st.rerun() here, Streamlit handles the update

# --- Display History ---
st.subheader("Analysis History")

if not st.session_state.history:
    st.info("No sentences analyzed yet. Type something above to start!")
else:
    for i, (original, result_html) in enumerate(st.session_state.history):
        with st.expander(f"**Input:** {original}", expanded=(i == 0)):
            st.markdown("### 📊 Model Output:")
            st.markdown(f"<div class='result-box'>{result_html}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🏷️ Raw Labels:")
            st.json(predict_code_switching(original))