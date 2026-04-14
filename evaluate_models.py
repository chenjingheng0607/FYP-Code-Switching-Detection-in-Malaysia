import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification
)
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Create output directory for your thesis charts
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ID2LABEL = {0: 'O', 1: 'MS', 2: 'EN'}
LABEL_NAMES = ['Other', 'Malay', 'English']

# Define paths (Change these if your folders are named differently)
MODELS_CONFIG = {
    "mBERT": {
        "model_path": "model/malay-english-codeswitch-model-mbert_full",
        "data_cache": "dataset/tokenized_data_cache_mbert"
    },
    "XLM-R": {
        "model_path": "model/malay-english-codeswitch-model-xlm_full",
        "data_cache": "dataset/tokenized_data_cache_xlm"
    }
}

BATCH_SIZE = 16  # Safe for 8GB VRAM during inference

# -----------------------------------------------------------------------------
# Chart Generation Helpers
# -----------------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"📊 Saved Confusion Matrix to: {file_path}")

def save_comparison_chart(results_dict):
    models = list(results_dict.keys())
    f1_scores = [results_dict[m]["F1"] * 100 for m in models]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e'])
    plt.title('Macro F1-Score Comparison')
    plt.ylabel('F1 Score (%)')
    plt.ylim(0, 100)
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    file_path = os.path.join(OUTPUT_DIR, "model_comparison_f1.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"📊 Saved Comparison Chart to: {file_path}")

# -----------------------------------------------------------------------------
# Evaluation Engine
# -----------------------------------------------------------------------------
def evaluate_model(model_name):
    config = MODELS_CONFIG[model_name]
    
    print(f"\n{'='*50}")
    print(f"🔍 EVALUATING: {model_name}")
    print(f"{'='*50}")

    if not os.path.exists(config["model_path"]):
        print(f"❌ ERROR: Model weights not found at {config['model_path']}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    model = AutoModelForTokenClassification.from_pretrained(config["model_path"]).to(device)
    model.eval() # Set to evaluation mode

    # 2. Load Test Dataset
    print(f"📂 Loading test dataset from {config['data_cache']}...")
    dataset = load_from_disk(config["data_cache"])
    test_dataset = dataset["test"]

    # 3. Setup DataLoader
    data_collator = DataCollatorForTokenClassification(tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

    all_preds = []
    all_labels = []

    # 4. Batch Inference Loop (Memory Safe)
    print("⏳ Running Inference...")
    for batch in tqdm(test_dataloader, desc="Evaluating Batches"):
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get highest probability prediction
        predictions = torch.argmax(outputs.logits, dim=-1)

        # IMMEDIATELY move back to CPU and convert to numpy to save VRAM
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # 5. Filter out -100 (padding and subwords)
        for pred_seq, label_seq in zip(predictions, labels):
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    all_preds.append(p)
                    all_labels.append(l)

    # 6. Calculate Metrics
    print("\n🧮 Calculating Metrics...")
    acc = accuracy_score(all_labels, all_preds)
    # Use macro average because 'O' (Other) class will be huge compared to MS/EN
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n✅ {model_name} Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")

    # 7. Generate Visuals
    save_confusion_matrix(all_labels, all_preds, model_name)

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}
    
    # Run evaluation for both models
    for model_name in MODELS_CONFIG.keys():
        metrics = evaluate_model(model_name)
        if metrics:
            results[model_name] = metrics

    # Generate final comparison chart if both models succeeded
    if len(results) == 2:
        print(f"\n{'='*50}")
        print("🏆 GENERATING FINAL COMPARISON")
        print(f"{'='*50}")
        save_comparison_chart(results)
        
    print("\n🎉 All evaluations complete! Check the 'evaluation_results' folder for your thesis charts.")