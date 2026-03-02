from transformers import AutoTokenizer, AutoModelForTokenClassification, XLMRobertaTokenizerFast
import torch
import os

# 获取当前脚本绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "mBERT": os.path.join(BASE_DIR, "model", "malay-english-codeswitch-model-mbert_full"),
    "XLM-R": os.path.join(BASE_DIR, "model", "malay-english-codeswitch-model-xlm_full", "checkpoint-800000"),
}

def load_model(name):
    if name not in MODEL_PATHS:
        raise ValueError(f"Model name '{name}' not found in MODEL_PATHS")

    model_path = MODEL_PATHS[name]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    print(f"Loading model from: {model_path}")

    if name == "XLM-R":
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)

    model.eval()
    return tokenizer, model

def predict(tokenizer, model, text):
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    word_ids = inputs.word_ids()
    previous_word_idx = None
    final_labels = []

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        elif word_idx != previous_word_idx:
            label_id = predictions[0][idx].item()
            final_labels.append(label_id)
        previous_word_idx = word_idx

    return tokens, final_labels