# predictor.py (Final Corrected Version)
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Configuration ---
MODELS = {
    "mBERT": "model/malay-english-codeswitch-model-mbert_full/checkpoint-307352",
    "XLM-R": "model/malay-english-codeswitch-model-xlm_full/checkpoint-120000"
}

# --- Caching ---
@st.cache_resource
def load_model_and_tokenizer(model_name):
    """Load the fine-tuned model and tokenizer based on the model name."""
    model_path = MODELS.get(model_name)
    if not model_path:
        st.error(f"Model {model_name} not found in configuration.")
        return None, None
        
    print(f"--- Loading {model_name} model and tokenizer from {model_path}... ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        print(f"--- {model_name} loaded successfully. ---")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None, None

def predict_code_switching(text, tokenizer, model):
    """
    Takes a sentence and returns a list of (token, label) pairs.
    """
    if not text or not isinstance(text, str) or tokenizer is None or model is None:
        return []

    id2label = model.config.id2label

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # Get model predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    
    predictions = torch.argmax(logits, dim=2)
    
    # Pair each token with its predicted label
    results = []
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Convert to CPU for numpy/iteration
    prediction_ids = predictions[0].cpu().numpy()

    for token, prediction_id in zip(tokens, prediction_ids):
        if token not in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, "<s>", "</s>", "<pad>"):
            label = id2label[prediction_id]
            results.append((token, label))
            
    return results

def format_prediction_as_html(predictions, model_type="mBERT"):
    """
    Takes a list of (token, label) pairs and formats them as an HTML string
    with color-coded labels, correctly handling and grouping sub-words.
    """
    if not predictions:
        return ""
        
    color_map = {
        'B-MS': '#FFADAD', 'I-MS': '#FFADAD', 'MS': '#FFADAD', # Light Red for Malay
        'B-EN': '#A0C4FF', 'I-EN': '#A0C4FF', 'EN': '#A0C4FF', # Light Blue for English
        'O': 'transparent'
    }
    
    final_html = ""
    current_word = ""
    current_label = ""

    # SentencePiece prefix used by XLM-R
    SPIECE_PREFIX = "\u2581" 

    for token, label in predictions:
        is_subword = False
        
        if model_type == "mBERT":
            if token.startswith("##"):
                is_subword = True
                clean_token = token[2:]
            else:
                clean_token = token
        else: # XLM-R or other SentencePiece models
            if not token.startswith(SPIECE_PREFIX):
                is_subword = True
                clean_token = token
            else:
                clean_token = token[1:] # Remove the prefix

        if is_subword:
            # If the token is a sub-word, just append it to the current word string
            current_word += clean_token
        else:
            # It's a new word. First, render the previous word group if it exists.
            if current_word:
                color = color_map.get(current_label, 'transparent')
                final_html += f' <span style="background-color: {color}; padding: 3px 6px; border-radius: 5px;">{current_word}</span>'
            
            # Start the new word group
            current_word = clean_token
            current_label = label
            
    # After the loop, render the very last word group
    if current_word:
        color = color_map.get(current_label, 'transparent')
        final_html += f' <span style="background-color: {color}; padding: 3px 6px; border-radius: 5px;">{current_word}</span>'

    return final_html.strip()
