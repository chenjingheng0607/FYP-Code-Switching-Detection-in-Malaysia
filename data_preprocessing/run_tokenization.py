import os
from datasets import Dataset
from transformers import AutoTokenizer

# --- Configurations ---
DATASET_IN = "./dataset/finetuning_dataset_malaya_full.jsonl"

MODELS = {
    "mbert": {
        "checkpoint": "bert-base-multilingual-cased",
        "save_path": "./dataset/tokenized_data_cache_mbert"
    },
    "xlm-r": {
        "checkpoint": "xlm-roberta-base",
        "save_path": "./dataset/tokenized_data_cache_xlm"
    }
}

# --- Mapping Function ---
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

# --- Processing Engine ---
def process_model(model_key):
    config = MODELS[model_key]
    print(f"\n[{model_key.upper()}] Loading JSONL from {DATASET_IN}...")
    
    if not os.path.exists(DATASET_IN):
        print(f"❌ ERROR: Cannot find {DATASET_IN}. Did Malaya finish running?")
        return
        
    raw_dataset = Dataset.from_json(DATASET_IN)
    dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    print(f"[{model_key.upper()}] Initializing tokenizer: {config['checkpoint']}")
    
    # 🌟 CRITICAL FIX: XLM-R requires add_prefix_space=True for pre-split arrays
    if "roberta" in config["checkpoint"]:
        tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'], add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])

    print(f"[{model_key.upper()}] Mapping labels using 8 CPU cores. This will be fast...")
    tokenized_datasets = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=8,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset_dict["train"].column_names,
        desc=f"Tokenizing {model_key.upper()}"
    )

    print(f"[{model_key.upper()}] Saving to disk...")
    tokenized_datasets.save_to_disk(config["save_path"])
    print(f"✅ [{model_key.upper()}] Done! Saved to {config['save_path']}\n")


# --- Interactive Terminal Menu ---
if __name__ == '__main__':
    while True:
        print("\n" + "="*45)
        print("   🚀 HUGGING FACE TOKENIZATION PIPELINE   ")
        print("="*45)
        print("1. Process mBERT data only")
        print("2. Process XLM-RoBERTa data only")
        print("3. Process BOTH models")
        print("4. Exit")
        print("="*45)
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            process_model("mbert")
            break
        elif choice == '2':
            process_model("xlm-r")
            break
        elif choice == '3':
            process_model("mbert")
            process_model("xlm-r")
            break
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")