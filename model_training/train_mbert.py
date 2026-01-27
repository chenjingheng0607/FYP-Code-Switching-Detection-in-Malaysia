import os
import torch
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
PROCESSED_DATA_PATH = "dataset/tokenized_data_cache_mbert"
OUTPUT_DIR = "model/malay-english-codeswitch-model-mbert_full"
MODEL_CHECKPOINT = "bert-base-multilingual-cased" 

id2label = {0: 'O', 1: 'MS', 2: 'EN'}
label2id = {v: k for k, v in id2label.items()}
label_list = ['O', 'MS', 'EN']

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
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
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    # Hardware Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"--- Loading saved tokens... ---")
        tokenized_datasets = load_from_disk(PROCESSED_DATA_PATH)
    else:
        print("--- Tokenizing from scratch... ---")
        raw_dataset = Dataset.from_json(DATASET_FILE)
        dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
        num_cpus = os.cpu_count()
        tokenized_datasets = dataset_dict.map(
            tokenize_and_align_labels, 
            batched=True,
            num_proc=num_cpus,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=dataset_dict["train"].column_names
        )
        tokenized_datasets.save_to_disk(PROCESSED_DATA_PATH)

    # --- OPTIONAL: Shrink evaluation set for speed ---
    # Evaluating 1 million rows takes too long. Let's use 10,000 for quick testing.
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10000))

    print("\n--- 3. Setting Up Model and Trainer ---")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=2,
        load_best_model_at_end=True,
        # Performance & VRAM Fixes
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        group_by_length=False,
        dataloader_num_workers=8,
        # --- THE FIX: Move evaluation data to RAM instead of GPU ---
        eval_accumulation_steps=50, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset, # Using the smaller subset
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, seqeval, label_list),
    )

    # -----------------------------------------------------------------------------
    # 4. Train (With Resume)
    # -----------------------------------------------------------------------------
    print("\n--- 4. Starting Fine-Tuning ---")
    
    # This will automatically look for 'checkpoint-5000' and continue
    trainer.train(resume_from_checkpoint=True)

    print("\n--- 5. Training Complete ---")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()