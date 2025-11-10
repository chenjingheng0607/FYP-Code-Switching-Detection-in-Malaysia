# predictor.py (Final Corrected Version)
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Configuration ---
MODEL_PATH = "model/malay-english-codeswitch-model-xlm_500k/checkpoint-40579"

# --- Caching ---
# Use Streamlit's cache to load the model and tokenizer only once
@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    print("--- Loading model and tokenizer... ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        print("--- Model and tokenizer loaded successfully. ---")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please ensure the model path is correct in predictor.py: '{MODEL_PATH}'")
        return None, None

# Load the model right away.
tokenizer, model = load_model_and_tokenizer()

# Check if the model loaded successfully before proceeding
if tokenizer and model:
    id2label = model.config.id2label
else:
    # Stop the app if the model can't be loaded
    st.stop()


def predict_code_switching(text):
    """
    Takes a sentence and returns a list of (token, label) pairs.
    """
    if not text or not isinstance(text, str):
        return []

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predictions = torch.argmax(logits, dim=2)
    
    # Pair each token with its predicted label
    results = []
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    for token, prediction_id in zip(tokens, predictions[0].numpy()):
        if token not in (tokenizer.cls_token, tokenizer.sep_token):
            label = id2label[prediction_id]
            results.append((token, label))
            
    return results

def format_prediction_as_html(predictions):
    """
    Takes a list of (token, label) pairs and formats them as an HTML string
    with color-coded labels, correctly handling and grouping sub-words.
    """
    if not predictions:
        return ""
        
    color_map = {
        'B-MS': '#FFADAD', 'I-MS': '#FFADAD', # Light Red for Malay
        'B-EN': '#A0C4FF', 'I-EN': '#A0C4FF', # Light Blue for English
        'O': 'transparent'
    }
    
    final_html = ""
    current_word = ""
    current_label = ""

    for token, label in predictions:
        # If the token is a sub-word, just append it to the current word string
        if token.startswith("##"):
            current_word += token[2:]
        else:
            # It's a new word. First, render the previous word group if it exists.
            if current_word:
                color = color_map.get(current_label, 'transparent')
                final_html += f' <span style="background-color: {color}; padding: 3px 6px; border-radius: 5px;">{current_word}</span>'
            
            # Start the new word group
            current_word = token
            current_label = label
            
    # After the loop, render the very last word group
    if current_word:
        color = color_map.get(current_label, 'transparent')
        final_html += f' <span style="background-color: {color}; padding: 3px 6px; border-radius: 5px;">{current_word}</span>'

    return final_html.strip()