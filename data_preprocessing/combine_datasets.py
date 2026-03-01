import json
import os

def extract_from_twitter_json(filepath, key_name, output_file_handle):
    """
    Reads a twitter JSON file, extracts sentences from the specified key,
    and writes them to the provided file handle.
    """
    print(f"-> Processing '{filepath}'...")
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
            
            if key_name in data and isinstance(data[key_name], list):
                sentences = data[key_name]
                for sentence in sentences:
                    if isinstance(sentence, str):
                        output_file_handle.write(sentence + '\n')
                        count += 1
        print(f"   ...extracted {count} sentences.")
        return count
    except Exception as e:
        print(f"   ...ERROR reading '{filepath}': {e}")
        return 0

def filter_and_extract_from_large_jsonl(filepath, labels_to_keep, output_file_handle):
    """
    Streams a large .jsonl file, filters by sentence_label, and writes
    the text to the provided file handle.
    """
    print(f"-> Streaming and filtering '{filepath}' (this may take a few minutes)...")
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                data = json.loads(line)
                if data.get("sentence_label") in labels_to_keep:
                    output_file_handle.write(data["text"] + '\n')
                    count += 1
                    
                    if count % 100000 == 0:
                        print(f"  ... still working, found {count} relevant sentences so far", end='\r')

        print(f"\n   ...extracted a total of {count} relevant sentences.")
        return count
    except Exception as e:
        print(f"   ...ERROR reading '{filepath}': {e}")
        return 0

# --- Main script execution ---
if __name__ == "__main__":
    
    # --- Configuration ---
    output_filename = 'dataset/combined_raw_sentences.txt'
    
    # Source files
    twitter_train_file = 'dataset/train-set.json'
    twitter_test_file = 'dataset/test-set.json'
    large_lang_detect_file = 'dataset/language_detection_dataset.jsonl'
    
    # Labels to keep from the large file
    labels_to_keep = {'malay', 'eng', 'ind', 'manglish', 'rojak'}
    
    total_sentences_written = 0
    
    print(f"--- Starting: Combining all datasets into '{output_filename}' ---")
    
    # Open the final output file once for writing
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        
        # --- Stage 1: Process Twitter Training Set ---
        total_sentences_written += extract_from_twitter_json(
            twitter_train_file, "train_X", f_out
        )
        
        # --- Stage 2: Process Twitter Test Set ---
        total_sentences_written += extract_from_twitter_json(
            twitter_test_file, "test_X", f_out
        )
        
        # --- Stage 3: Process Large Language Detection File ---
        total_sentences_written += filter_and_extract_from_large_jsonl(
            large_lang_detect_file, labels_to_keep, f_out
        )

    print("\n----------------------------------------------------")
    print(f"✅ Success! All data has been combined.")
    print(f"   -> Your raw, combined dataset is: {output_filename}")
    print(f"   -> It contains a total of {total_sentences_written} sentences (before cleaning).")
    print("----------------------------------------------------")