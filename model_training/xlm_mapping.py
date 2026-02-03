import os
from datasets import Dataset
from transformers import AutoTokenizer

# --- 1. Functions and Configs can stay outside ---
DATASET_IN = "./dataset/finetuning_dataset_malaya_full.jsonl"
SAVE_PATH = "./dataset/tokenized_data_cache_xlm"
MODEL_CHECKPOINT = "xlm-roberta-base"

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

# --- 2. The Safeguard ---
if __name__ == '__main__':
    # Move ALL execution code inside here
    print("Loading local JSONL...")
    raw_dataset = Dataset.from_json(DATASET_IN)
    dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print(f"Mapping on 8 cores...")
    # Use fn_kwargs to pass the tokenizer into the function safely
    tokenized_datasets = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=8,
        fn_kwargs={"tokenizer": tokenizer}, # Pass tokenizer here
        remove_columns=dataset_dict["train"].column_names
    )

    tokenized_datasets.save_to_disk(SAVE_PATH)
    print(f"Done! Successfully saved to {SAVE_PATH}")