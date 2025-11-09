# create_sample_100k.py
import json
import ijson
import os
import sys

# --- Configuration ---
TWITTER_TRAIN_FILE = 'dataset/train-set.json'
TWITTER_TEST_FILE = 'dataset/test-set.json' # Added the test file
LARGE_DATASET_FILE = 'dataset/train-test.json'

# New output filename and size
OUTPUT_SAMPLE_FILE = 'dataset/raw_sample_500k.txt'
SAMPLE_SIZE = 500000

# We'll take a balanced amount from each source
TWITTER_TRAIN_LIMIT = 40000  # Take more from this high-quality source
TWITTER_TEST_LIMIT = 10000   # Add sentences from the test set as well
LARGE_DATASET_LIMIT = 450000  # Take a larger portion from the big file

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"--- Creating a raw sample of up to {SAMPLE_SIZE} sentences ---")
    
    # Using a set to automatically handle any duplicates between the files
    all_unique_sentences = set()
    
    # --- Part 1: Get sentences from the Twitter Training file ---
    if os.path.exists(TWITTER_TRAIN_FILE):
        print(f"-> Reading from '{TWITTER_TRAIN_FILE}'...")
        try:
            with open(TWITTER_TRAIN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sentences_from_twitter = data.get('train_X', [])
                for sentence in sentences_from_twitter[:TWITTER_TRAIN_LIMIT]:
                    if isinstance(sentence, str):
                        all_unique_sentences.add(sentence.strip())
            print(f"   ...total unique sentences so far: {len(all_unique_sentences)}")
        except Exception as e:
            print(f"   ...ERROR reading file: {e}")
    else:
        print(f"-> WARNING: File not found: '{TWITTER_TRAIN_FILE}'. Skipping.")

    # --- Part 2: Get sentences from the Twitter Test file ---
    if os.path.exists(TWITTER_TEST_FILE):
        print(f"-> Reading from '{TWITTER_TEST_FILE}'...")
        try:
            with open(TWITTER_TEST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sentences_from_twitter_test = data.get('test_X', [])
                for sentence in sentences_from_twitter_test[:TWITTER_TEST_LIMIT]:
                    if isinstance(sentence, str):
                        all_unique_sentences.add(sentence.strip())
            print(f"   ...total unique sentences so far: {len(all_unique_sentences)}")
        except Exception as e:
            print(f"   ...ERROR reading file: {e}")
    else:
        print(f"-> WARNING: File not found: '{TWITTER_TEST_FILE}'. Skipping.")

    # --- Part 3: Get sentences from the large dataset ---
    if len(all_unique_sentences) < SAMPLE_SIZE and os.path.exists(LARGE_DATASET_FILE):
        print(f"-> Streaming from '{LARGE_DATASET_FILE}'...")
        try:
            with open(LARGE_DATASET_FILE, 'rb') as f_x:
                texts = ijson.items(f_x, 'train_X.item')
                for i, text in enumerate(texts):
                    if i >= LARGE_DATASET_LIMIT:
                        break
                    if isinstance(text, str):
                        all_unique_sentences.add(text.strip())
            print(f"   ...total unique sentences so far: {len(all_unique_sentences)}")
        except Exception as e:
            print(f"\n   ...ERROR streaming the large file: {e}")
            print("   Please ensure 'ijson' is installed (`uv pip install ijson`).")
    else:
        print(f"-> SKIPPING large dataset (either not found or sample size already met).")
        
    # --- Part 4: Save the raw sample to a file ---
    final_sentences = list(all_unique_sentences)
    # If we have more than the target, trim it down to the exact size
    if len(final_sentences) > SAMPLE_SIZE:
        final_sentences = final_sentences[:SAMPLE_SIZE]

    print(f"\n-> Writing {len(final_sentences)} sentences to '{OUTPUT_SAMPLE_FILE}'...")
    with open(OUTPUT_SAMPLE_FILE, 'w', encoding='utf-8') as f_out:
        for sentence in final_sentences:
            f_out.write(sentence + '\n')
            
    print("\n----------------------------------------------------")
    print("✅ Success! Your new raw sample is ready.")
    print(f"-> File saved as: {OUTPUT_SAMPLE_FILE}")
    print(f"-> Contains {len(final_sentences)} unique sentences.")
    print("----------------------------------------------------")