import os
import torch
import json
import numpy as np
import evaluate
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_FILE = 'dataset/finetuning_dataset_malaya_full.jsonl'
PROCESSED_DATA_PATH = "dataset/tokenized_data_cache_xlm"
OUTPUT_DIR = "model/malay-english-codeswitch-model-xlm_full"
MODEL_CHECKPOINT = "xlm-roberta-base" 

id2label = {0: 'O', 1: 'MS', 2: 'EN'}
label2id = {v: k for k, v in id2label.items()}
label_list = ['O', 'MS', 'EN']

# -----------------------------------------------------------------------------
# Helper Functions (Simplified)
# -----------------------------------------------------------------------------
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, seqeval, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    # Optimization for RTX 40-series
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Prevent memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    if os.path.exists(PROCESSED_DATA_PATH):
        tokenized_datasets = load_from_disk(PROCESSED_DATA_PATH)
    else:
        raw_dataset = Dataset.from_json(DATASET_FILE)
        dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        tokenized_datasets = dataset_dict.map(
            tokenize_and_align_labels, 
            batched=True,
            num_proc=os.cpu_count(),
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=dataset_dict["train"].column_names
        )
        tokenized_datasets.save_to_disk(PROCESSED_DATA_PATH)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    # VRAM Save: Enable gradient checkpointing on the model
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        
        # --- BATCH SIZE & MEMORY ---
        per_device_train_batch_size=8,      
        gradient_accumulation_steps=4,      # 8 x 4 = 32 effective batch size
        per_device_eval_batch_size=8,
        
        # --- STRATEGY (Matched for Epoch) ---
        # This saves your work every 5000 steps (~once an hour)
        eval_strategy="steps",
        eval_steps=10000,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=2,                 # Keep best + last epoch
        load_best_model_at_end=True,        # Reload the best version when finished
        metric_for_best_model="f1",         # Use F1 score to find the "best" model

        # --- HARDWARE & SPEED ---
        bf16=True,                          # RTX 40-series optimization
        tf32=True,
        optim="adamw_bnb_8bit",             # 8-bit optimizer for VRAM efficiency
        gradient_checkpointing=True,        
        eval_accumulation_steps=10000,        # Higher value for faster evaluation on Ubuntu
        dataloader_num_workers=4,           
        group_by_length=False,              
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, seqeval, label_list),
    )

    trainer.train(resume_from_checkpoint=True)
    # trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()