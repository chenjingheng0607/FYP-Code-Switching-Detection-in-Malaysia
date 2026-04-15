import os
from datasets import Dataset
from transformers import AutoTokenizer
import multiprocessing as mp

# --- Configurations ---
DATASET_IN = "./dataset/finetuning_dataset_malaya_full.jsonl"
SAVE_PATH = "./dataset/tokenized_data_cache_xlm_tpu"

# 🌟 CRITICAL FOR TPU: Fixed maximum sequence length
MAX_LEN = 128 

# --- Mapping Function ---
def tokenize_and_align_labels(examples, tokenizer):
    # 🌟 CRITICAL FOR TPU: Added padding="max_length" and max_length=MAX_LEN
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LEN,
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # This brilliantly handles [CLS], [SEP], AND the new [PAD] tokens!
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
if __name__ == '__main__':
    print("\n" + "="*50)
    print(" 🚀 TPU-OPTIMIZED TOKENIZATION (XLM-RoBERTa) ")
    print("="*50)
    
    if not os.path.exists(DATASET_IN):
        print(f"❌ ERROR: Cannot find {DATASET_IN}.")
        exit()

    print(f"Loading JSONL from {DATASET_IN}...")
    raw_dataset = Dataset.from_json(DATASET_IN)
    dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    print("Initializing XLM-R tokenizer (add_prefix_space=True)...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True)

    num_cores = max(1, mp.cpu_count())

    print(f"Mapping labels with STATIC PADDING ({MAX_LEN}) using {num_cores} cores...")
    tokenized_datasets = dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=num_cores,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing for TPU"
    )

    print(f"Saving to disk...")
    tokenized_datasets.save_to_disk(SAVE_PATH)
    print(f"✅ Done! Ready to upload {SAVE_PATH} to Google Drive!\n")