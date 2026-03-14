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

# Model-specific settings based on your highly optimized configurations
MODELS_CONFIG = {
    "mbert": {
        "checkpoint": "bert-base-multilingual-cased",
        "data_cache": "dataset/tokenized_data_cache_mbert",
        "output_dir": "model/malay-english-codeswitch-model-mbert_full",
        "lr": 2e-5,
        "train_batch_size": 16,
        "grad_accum_steps": 2,  # Effective batch size = 32
        "eval_batch_size": 8,
        "eval_accum_steps": 50,
        "eval_save_steps": 5000,
        "optim": "adamw_torch_fused",
        "gradient_checkpointing": False
    },
    "xlm-r": {
        "checkpoint": "xlm-roberta-base",
        "data_cache": "dataset/tokenized_data_cache_xlm",
        "output_dir": "model/malay-english-codeswitch-model-xlm_full",
        "lr": 3e-5,
        "train_batch_size": 8,
        "grad_accum_steps": 4,  # Effective batch size = 32
        "eval_batch_size": 8,
        "eval_accum_steps": 100,
        "eval_save_steps": 50000,
        "optim": "adamw_bnb_8bit", # Requires 'bitsandbytes'
        "gradient_checkpointing": True
    }
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_metrics(p, seqeval_metric, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# -----------------------------------------------------------------------------
# Main Training Engine
# -----------------------------------------------------------------------------
def train_model(model_key):
    config = MODELS_CONFIG[model_key]
    print(f"\n" + "="*50)
    print(f"🚀 INITIALIZING TRAINING FOR: {model_key.upper()}")
    print("="*50)

    # 1. Hardware Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 2. Check for Tokenized Data First
    if not os.path.exists(config["data_cache"]):
        print(f"\n❌ ERROR: Tokenized data cache not found at '{config['data_cache']}'")
        print(f"⚠️  Please run 'python run_tokenization.py' first and select option to process {model_key.upper()} data.")
        return  

    # 3. Load Dataset
    print(f"--- Loading cached tokenized data from {config['data_cache']} ---")
    tokenized_datasets = load_from_disk(config["data_cache"])

    print("--- Preparing Evaluation Subset (10,000 rows) ---")
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10000))

    # 4. Initialize Tokenizer 
    if "roberta" in config["checkpoint"]:
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])

    # # 5. Initialize Model and Collator
    # print(f"--- Loading {config['checkpoint']} Model ---")
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # seqeval = evaluate.load("seqeval")
    
    # model = AutoModelForTokenClassification.from_pretrained(
    #     config["checkpoint"], num_labels=len(label_list), id2label=id2label, label2id=label2id
    # )

    # if config["gradient_checkpointing"]:
    #     print("--- Enabling VRAM-Saving Gradient Checkpointing ---")
    #     model.gradient_checkpointing_enable()

    # 5. Initialize Model and Collator
    print(f"--- Loading {config['checkpoint']} Model ---")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")
    
    model = AutoModelForTokenClassification.from_pretrained(
        config["checkpoint"], num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    # --- ADD THIS PART ---
    print("--- Compiling Model with Torch Compile ---")
    try:
        # 'reduce-overhead' is great for most training tasks
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠️ Torch compile failed or not supported: {e}")
    # ----------------------

    if config["gradient_checkpointing"]:
        print("--- Enabling VRAM-Saving Gradient Checkpointing ---")
        model.gradient_checkpointing_enable()

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["lr"],
        num_train_epochs=3,
        weight_decay=0.01,
        
        per_device_train_batch_size=config["train_batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        per_device_eval_batch_size=config["eval_batch_size"],
        
        eval_strategy="steps",
        eval_steps=config["eval_save_steps"],
        save_strategy="steps",
        save_steps=config["eval_save_steps"],
        save_total_limit=2,                 
        load_best_model_at_end=True,        
        metric_for_best_model="f1",         

        bf16=True,                          
        tf32=True,
        optim=config["optim"],             
        eval_accumulation_steps=config["eval_accum_steps"],        
        dataloader_num_workers=4,
        remove_unused_columns=False,                         
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset, 
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, seqeval, label_list),
    )

    # 7. Start Training
    print("\n--- Starting Fine-Tuning ---")
    try:
        print("🔍 Searching for previous checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        print("ℹ️ No checkpoint found. Starting training from scratch...")
        trainer.train()
        
    print("\n✅ Training Complete!")
    print(f"💾 Saving final model to {config['output_dir']}...")
    trainer.save_model(config["output_dir"])

# -----------------------------------------------------------------------------
# Interactive CLI Menu
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        print("\n" + "="*45)
        print(" 🧠 HUGGING FACE MODEL TRAINING PIPELINE ")
        print("="*45)
        print("1. Train mBERT (bert-base-multilingual-cased)")
        print("2. Train XLM-RoBERTa (xlm-roberta-base)")
        print("3. Exit")
        print("="*45)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            train_model("mbert")
            break
        elif choice == '2':
            train_model("xlm-r")
            break
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")