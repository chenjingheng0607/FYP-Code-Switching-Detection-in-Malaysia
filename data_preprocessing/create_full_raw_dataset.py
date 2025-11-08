import json
import ijson
import os
import sys

# --- Configuration ---
TWITTER_TRAIN_FILE = 'dataset/train-set.json'
TWITTER_TEST_FILE = 'dataset/test-set.json'
LARGE_DATASET_FILE = 'dataset/train-test.json'

# The final output file will contain ALL raw sentences from our sources
FULL_RAW_DATASET_FILE = 'dataset/full_raw_sentences.txt'

# We will only keep sentences from the large dataset with these labels
LABELS_TO_KEEP = {'malay', 'eng', 'ind', 'manglish', 'rojak'}

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"--- Creating FULL raw dataset from all sources ---")
    
    # Using a set is a memory-efficient way to store only unique sentences
    unique_sentences = set()
    
    # --- Part 1: Get ALL sentences from the Twitter Training file ---
    if os.path.exists(TWITTER_TRAIN_FILE):
        print(f"-> Reading from '{TWITTER_TRAIN_FILE}'...")
        try:
            with open(TWITTER_TRAIN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sentences_from_twitter = data.get('train_X', [])
                for sentence in sentences_from_twitter:
                    if isinstance(sentence, str):
                        unique_sentences.add(sentence.strip())
            print(f"   ...done. Total unique sentences so far: {len(unique_sentences)}")
        except Exception as e:
            print(f"   ...ERROR reading file: {e}")
    else:
        print(f"-> WARNING: File not found: '{TWITTER_TRAIN_FILE}'. Skipping.")

    # --- Part 2: Get ALL sentences from the Twitter Test file ---
    if os.path.exists(TWITTER_TEST_FILE):
        print(f"-> Reading from '{TWITTER_TEST_FILE}'...")
        try:
            with open(TWITTER_TEST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sentences_from_twitter_test = data.get('test_X', [])
                for sentence in sentences_from_twitter_test:
                    if isinstance(sentence, str):
                        unique_sentences.add(sentence.strip())
            print(f"   ...done. Total unique sentences so far: {len(unique_sentences)}")
        except Exception as e:
            print(f"   ...ERROR reading file: {e}")
    else:
        print(f"-> WARNING: File not found: '{TWITTER_TEST_FILE}'. Skipping.")


    # --- Part 3: Stream and FILTER the entire large dataset ---
    if os.path.exists(LARGE_DATASET_FILE):
        print(f"-> Streaming and filtering '{LARGE_DATASET_FILE}' (this will take several minutes)...")
        processed_count = 0
        try:
            with open(LARGE_DATASET_FILE, 'rb') as f_x, open(LARGE_DATASET_FILE, 'rb') as f_y:
                texts = ijson.items(f_x, 'train_X.item')
                labels = ijson.items(f_y, 'train_Y.item')
                
                for text, label in zip(texts, labels):
                    processed_count += 1
                    if (processed_count % 100000 == 0):
                        print(f"   ...scanned {processed_count} records...", end='\r')

                    if label in LABELS_TO_KEEP and isinstance(text, str):
                        unique_sentences.add(text.strip())
            
            print(f"\n   ...done scanning all records. Total unique sentences so far: {len(unique_sentences)}")
        except Exception as e:
            print(f"\n   ...ERROR streaming the large file: {e}")
            print("   Please ensure 'ijson' is installed (`uv pip install ijson`) and the file is not corrupt.")
    else:
        print(f"-> WARNING: File not found: '{LARGE_DATASET_FILE}'. Skipping.")
        
    # --- Part 4: Save the final combined and deduplicated file ---
    print(f"\n-> Writing {len(unique_sentences)} unique sentences to '{FULL_RAW_DATASET_FILE}'...")
    with open(FULL_RAW_DATASET_FILE, 'w', encoding='utf-8') as f_out:
        # Sorting is optional but makes the file content consistent if you run it again
        for sentence in sorted(list(unique_sentences)):
            f_out.write(sentence + '\n')
            
    print("\n----------------------------------------------------")
    print("✅ Success! Your full raw dataset is ready.")
    print(f"-> File saved as: {FULL_RAW_DATASET_FILE}")
    print("----------------------------------------------------")