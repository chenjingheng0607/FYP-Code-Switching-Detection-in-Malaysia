import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# Malay/Manglish → English  (use id→en)
@st.cache_resource
def load_ms_en():
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-id-en")
    return tok, model

def ms_to_en(text):
    tok, model = load_ms_en()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tok.decode(outputs[0], skip_special_tokens=True)


# English → Malay  (use en→id)
@st.cache_resource
def load_en_ms():
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-id")
    return tok, model

def en_to_ms(text):
    tok, model = load_en_ms()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tok.decode(outputs[0], skip_special_tokens=True)