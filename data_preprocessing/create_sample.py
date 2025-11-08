# create_sample.py
import json
import ijson
import os
import sys

# --- Configuration ---
TWITTER_TRAIN_FILE = 'dataset/train-set.json'
LARGE_DATASET_FILE = 'dataset/train-test.json'
OUTPUT_SAMPLE_FILE = 'dataset/raw_sample_10k.txt'
SAMPLE_SIZE = 10000

# We'll take this many sentences from each source
TWITTER_LIMIT = 5000  # Prioritize the best source
LARGE_DATASET_LIMIT = 5000

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"--- Creating a raw sample of {SAMPLE_SIZE} sentences ---")
    
    all_sentences = []
    
    # --- Part 1: Get sentences from the Twitter file (high-quality source) ---
    if os.path.exists(TWITTER_TRAIN_FILE):
        print(f"-> Reading from '{TWITTER_TRAIN_FILE}'...")
        try:
            with open(TWITTER_TRAIN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sentences_from_twitter = data.get('train_X', [])
                # Take the first N sentences
                all_sentences.extend(sentences_from_twitter[:TWITTER_LIMIT])
            print(f"   ...added {len(all_sentences)} sentences.")
        except Exception as e:
            print(f"   ...ERROR reading file: {e}")
    else:
        print(f"-> WARNING: File not found: '{TWITTER_TRAIN_FILE}'. Skipping.")

    # --- Part 2: Get sentences from the large dataset ---
    if len(all_sentences) < SAMPLE_SIZE and os.path.exists(LARGE_DATASET_FILE):
        print(f"-> Streaming from '{LARGE_DATASET_FILE}'...")
        try:
            with open(LARGE_DATASET_FILE, 'rb') as f_x:
                texts = ijson.items(f_x, 'train_X.item')
                for i, text in enumerate(texts):
                    if i >= LARGE_DATASET_LIMIT:
                        break
                    all_sentences.append(text)
            print(f"   ...added sentences. Total now: {len(all_sentences)}")
        except Exception as e:
            print(f"\n   ...ERROR streaming the large file: {e}")
            print("   Please ensure 'ijson' is installed (`uv pip install ijson`).")
    else:
        print(f"-> WARNING: File not found: '{LARGE_DATASET_FILE}'. Skipping.")
        
    # --- Part 3: Save the raw sample to a file ---
    print(f"\n-> Writing {len(all_sentences)} sentences to '{OUTPUT_SAMPLE_FILE}'...")
    with open(OUTPUT_SAMPLE_FILE, 'w', encoding='utf-8') as f_out:
        for sentence in all_sentences:
            if isinstance(sentence, str):
                f_out.write(sentence.strip() + '\n')
            
    print("\n----------------------------------------------------")
    print("✅ Success! Your raw sample is ready.")
    print(f"-> File saved as: {OUTPUT_SAMPLE_FILE}")
    print("----------------------------------------------------")