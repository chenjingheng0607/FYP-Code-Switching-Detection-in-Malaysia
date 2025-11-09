import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import evaluate

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# --- File Paths ---
# The dataset we created in the previous step
DATASET_FILE = 'dataset/finetuning_dataset_malaya_500k.jsonl'
# Where the final trained model will be saved
OUTPUT_DIR = "model/malay-english-codeswitch-model-mbert_500k"

# --- Model Configuration ---
# You can choose other models like 'xlm-roberta-base' as well
MODEL_CHECKPOINT = "bert-base-multilingual-cased" 

# --- Label Mapping ---
# This MUST match the mapping used in your data generation script
# We also need the reverse mapping (id to label)
id2label = {
    0: 'O', 1: 'B-MS', 2: 'I-MS', 3: 'B-EN', 4: 'I-EN'
}
label2id = {v: k for k, v in id2label.items()}
label_list = list(id2label.values())

# -----------------------------------------------------------------------------
# 1. Load and Prepare the Dataset
# -----------------------------------------------------------------------------
print("--- 1. Loading and Preparing Dataset ---")

# Load the entire JSONL file into a Hugging Face Dataset object
# This is memory-efficient and handles all the loading for you.
raw_dataset = Dataset.from_json(DATASET_FILE)
print(f"   ...loaded {len(raw_dataset)} examples.")

# Split the dataset into training (90%) and testing (10%)
dataset_dict = raw_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
print(f"   ...split into {len(dataset_dict['train'])} training and {len(dataset_dict['test'])} test examples.")


# -----------------------------------------------------------------------------
# 2. Tokenize and Align Labels
# -----------------------------------------------------------------------------
print("\n--- 2. Tokenizing and Aligning Labels ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples):
    # The tokenizer will split words into sub-words (e.g., "Mukhriz" -> ["Muk", "##hriz"])
    # We need to align our single label to all the sub-words.
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS] and [SEP] get a label of -100 to be ignored by the loss function
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First sub-word of a new word gets the original label
                label_ids.append(label[word_idx])
            else:
                # Subsequent sub-words also get ignored
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)
print("   ...tokenization complete.")


# -----------------------------------------------------------------------------
# 3. Set Up the Model and Trainer
# -----------------------------------------------------------------------------
print("\n--- 3. Setting Up Model and Trainer ---")

# Data collator handles padding sentences to the same length in each batch
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Load the evaluation metric
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert predictions and labels from IDs back to string labels
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

# Load the pre-trained model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

# Define the training arguments
# These are basic settings; you can tune them for better performance
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=1,  # Start with 1 epoch for a fast baseline
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("   ...setup complete.")

# -----------------------------------------------------------------------------
# 4. Start Fine-Tuning
# -----------------------------------------------------------------------------
print("\n--- 4. Starting Fine-Tuning ---")
print("   (This will take some time depending on your hardware)...")

trainer.train()

print("\n--- 5. Training Complete ---")
print(f"   -> Your fine-tuned model has been saved to the directory: '{OUTPUT_DIR}'")