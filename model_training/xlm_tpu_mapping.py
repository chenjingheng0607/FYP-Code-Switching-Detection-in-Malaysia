import os
import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

# --- Configuration ---
# Change this to where your jsonl file is on your PC
INPUT_FILE = "dataset/finetuning_dataset_malaya_full.jsonl" 
# Change this to where you want to save the processed data
SAVE_PATH = "./dataset/tokenized_data_cache_xlm_tpu" 
MODEL_CHECKPOINT = "xlm-roberta-base"

def tokenize_and_align_labels(examples, tokenizer):
    # We use truncation=True and max_length=128 to make it TPU-ready
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        max_length=128, 
        padding="max_length", 
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Security check for index bounds
                if word_idx < len(label):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    print("--- Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print(f"--- Loading Dataset from {INPUT_FILE} ---")
    raw_dataset = Dataset.from_json(INPUT_FILE)
    
    # Split the data
    dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    # Use all 8 cores of your CPU
    num_cpus = 8 
    
    print(f"--- Tokenizing using {num_cpus} CPU cores ---")
    tokenized_datasets = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=num_cpus,  # This activates your 8 cores
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset_dict["train"].column_names
    )

    print(f"--- Saving processed data to {SAVE_PATH} ---")
    tokenized_datasets.save_to_disk(SAVE_PATH)
    print("--- Done! ---")

if __name__ == "__main__":
    main()