# models.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATHS = {
    "mBERT": "model/malay-english-codeswitch-model-mbert_500k/checkpoint-40579",
    "XLM-R": "model/malay-english-codeswitch-model-xlm_500k/checkpoint-40579",
    "mT5": "your_mt5_model_path_here",
}

def load_model(name):
    model_name = MODEL_PATHS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
