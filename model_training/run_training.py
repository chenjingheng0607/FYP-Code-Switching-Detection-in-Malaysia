import os
import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Global Configuration & Label Mapping
# -----------------------------------------------------------------------------
id2label = {0: 'O', 1: 'MS', 2: 'EN'}
label2id = {v: k for k, v in id2label.items()}
label_list = ['O', 'MS', 'EN']

load_dotenv()
token = os.getenv("HF_TOKEN")

# Model-specific settings 
MODELS_CONFIG = {
    "mbert": {
        "checkpoint": "bert-base-multilingual-cased",
        "data_cache": "dataset/tokenized_data_cache_mbert",
        "output_dir": "model/malay-english-codeswitch-model-mbert_full",
        "lr": 2e-5,
        "train_batch_size": 16,
        "grad_accum_steps": 2,  
        "eval_batch_size": 8,
        "eval_accum_steps": 50,
        "optim": "adamw_torch_fused",
        "gradient_checkpointing": False
    },
    "xlm-r": {
        "checkpoint": "xlm-roberta-base",
        "data_cache": "dataset/tokenized_data_cache_xlm",
        "output_dir": "model/malay-english-codeswitch-model-xlm_full",
        "lr": 3e-5,
        "train_batch_size": 8,
        "grad_accum_steps": 4,  
        "eval_batch_size": 8,
        "eval_accum_steps": 100,
        "optim": "adamw_bnb_8bit", 
        "gradient_checkpointing": True
    }
}

# -----------------------------------------------------------------------------
# Metrics Setup (Replaced seqeval with standard F1 and Accuracy)
# -----------------------------------------------------------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_classification_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Flatten arrays and drop -100 ignore index
    preds_flat = []
    labels_flat = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        for p_val, l_val in zip(pred_seq, label_seq):
            if l_val != -100:
                preds_flat.append(p_val)
                labels_flat.append(l_val)
                
    # Calculate metrics
    acc = accuracy_metric.compute(predictions=preds_flat, references=labels_flat)["accuracy"]
    # We use macro F1 because the 'O' class (punctuation) will be very large, and we want 
    # the model to be evaluated fairly on 'MS' and 'EN'.
    f1 = f1_metric.compute(predictions=preds_flat, references=labels_flat, average="macro")["f1"]
    
    return {
        "accuracy": acc,
        "f1": f1
    }

# -----------------------------------------------------------------------------
# Main Training Engine
# -----------------------------------------------------------------------------
def train_model(model_key, is_quick_test=False):
    config = MODELS_CONFIG[model_key]
    print(f"\n" + "="*50)
    
    # 1. Quick Test Logic
    if is_quick_test:
        print(f"🛠️  QUICK OOM TEST MODE FOR: {model_key.upper()}")
        print("   (Dataset truncated to 500 rows to force a fast epoch finish)")
        output_dir = config["output_dir"] + "_quicktest"
        epochs_to_run = 2 
    else:
        print(f"🚀 INITIALIZING FULL TRAINING FOR: {model_key.upper()}")
        output_dir = config["output_dir"]
        epochs_to_run = 3
    print("="*50)

    # 2. Hardware Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 3. Check for Tokenized Data
    if not os.path.exists(config["data_cache"]):
        print(f"\n❌ ERROR: Tokenized data cache not found at '{config['data_cache']}'")
        return  

    # 4. Load Dataset
    print(f"--- Loading cached tokenized data from {config['data_cache']} ---")
    tokenized_datasets = load_from_disk(config["data_cache"])

    if is_quick_test:
        train_dataset = tokenized_datasets["train"].select(range(500))
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]

    # 5. Initialize Tokenizer 
    if "roberta" in config["checkpoint"]:
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])

    # 6. Initialize Model and Collator
    print(f"--- Loading {config['checkpoint']} Model ---")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    model = AutoModelForTokenClassification.from_pretrained(
        config["checkpoint"], num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    if config["gradient_checkpointing"]:
        print("--- Enabling VRAM-Saving Gradient Checkpointing ---")
        model.gradient_checkpointing_enable()

    # 7. Training Arguments 
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["lr"],
        num_train_epochs=epochs_to_run,
        weight_decay=0.01,
        
        per_device_train_batch_size=config["train_batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        per_device_eval_batch_size=config["eval_batch_size"],
        
        eval_strategy="epoch",              
        save_strategy="epoch",              
        save_total_limit=2,                 
        load_best_model_at_end=True,        
        metric_for_best_model="f1",         

        bf16=True,                          
        tf32=True,
        optim=config["optim"],             
        eval_accumulation_steps=config["eval_accum_steps"],        
        dataloader_num_workers=4,
        logging_strategy="steps",
        logging_steps=500,                  # Log the training loss every 500 steps
        report_to="tensorboard"             # Save beautiful graphs for your thesis!
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_classification_metrics, # Use the new robust metric function
    )

    # 8. Start Training
    print("\n--- Starting Fine-Tuning ---")
    try:
        print("🔍 Searching for previous checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        print("ℹ️ No checkpoint found. Starting training from scratch...")
        trainer.train()
        
    print("\n✅ Training Complete!")
    print(f"💾 Saving final model to {output_dir}...")
    trainer.save_model(output_dir)

# -----------------------------------------------------------------------------
# Interactive CLI Menu
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        print("\n" + "="*45)
        print(" 🧠 HUGGING FACE MODEL TRAINING PIPELINE ")
        print("="*45)
        print("1. Train mBERT (Full Training)")
        print("2. Train XLM-RoBERTa (Full Training)")
        print("3. QUICK OOM TEST - mBERT (2 Mins)")
        print("4. QUICK OOM TEST - XLM-RoBERTa (2 Mins)")
        print("5. Exit")
        print("="*45)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            train_model("mbert", is_quick_test=False)
            break
        elif choice == '2':
            train_model("xlm-r", is_quick_test=False)
            break
        elif choice == '3':
            train_model("mbert", is_quick_test=True)
            break
        elif choice == '4':
            train_model("xlm-r", is_quick_test=True)
            break
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice. Please enter 1 to 5.")