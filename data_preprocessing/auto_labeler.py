import json
import re
import os

def load_word_list(filepath):
    """Loads a word list from a file into a set for fast lookups."""
    print(f"-> Loading word list from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use a set for O(1) average time complexity checks
            words = set(line.strip().lower() for line in f)
        print(f"   ...loaded {len(words)} words.")
        return words
    except FileNotFoundError:
        print(f"   ...ERROR: File not found. Please download it.")
        return set()

def auto_label_sentence(sentence, eng_words, ms_words):
    """
    Applies dictionary-based labels to a single sentence and returns
    the sentence text and a list of [start, end, label] spans.
    """
    labels = []
    # Simple word tokenization, handles basic punctuation
    tokens = re.finditer(r'\w+|[^\w\s]', sentence)
    
    for token in tokens:
        word = token.group(0).lower()
        start_char = token.start()
        end_char = token.end()
        
        label = "O" # Default to 'Outside'
        
        # Check if the word is in our dictionaries
        if word in eng_words and word in ms_words:
            # It's in both, mark it for manual review
            label = "AMBIGUOUS" 
        elif word in eng_words:
            label = "EN"
        elif word in ms_words:
            label = "MS"
        
        # We only need to add a label if it's not 'O'
        # Doccano doesn't need 'O' labels explicitly
        if label != "O":
            labels.append([start_char, end_char, label])
            
    return {"text": sentence, "labels": labels}


# --- Main script execution ---
if __name__ == "__main__":
    
    input_sentence_file = 'dataset/final_sentences_for_annotation.txt'
    output_doccano_file = 'dataset/doccano_import_draft.jsonl'
    
    # How many sentences to process. Start with a smaller number for the first run.
    SENTENCE_LIMIT = 20000 
    
    # --- 1. Load Dictionaries ---
    print("--- Loading Dictionaries ---")
    english_words = load_word_list('dataset/words_alpha.txt')
    malay_words = load_word_list('dataset/Malays.dic.txt')
    
    if not english_words or not malay_words:
        print("\n❌ Could not load one or more dictionaries. Exiting.")
    else:
        # --- 2. Process Sentences and Auto-Label ---
        print(f"\n--- Auto-labeling first {SENTENCE_LIMIT} sentences from '{input_sentence_file}' ---")
        
        processed_count = 0
        with open(input_sentence_file, 'r', encoding='utf-8') as f_in, \
             open(output_doccano_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if processed_count >= SENTENCE_LIMIT:
                    print(f"\nReached sentence limit of {SENTENCE_LIMIT}.")
                    break
                
                sentence = line.strip()
                if sentence:
                    # Get the labeled data in Doccano format
                    labeled_data = auto_label_sentence(sentence, english_words, malay_words)
                    
                    # Write it as a JSON line
                    f_out.write(json.dumps(labeled_data, ensure_ascii=False) + '\n')
                    processed_count += 1

                    if processed_count % 1000 == 0:
                        print(f"   ...processed {processed_count} sentences", end='\r')

        print("\n----------------------------------------------------")
        print(f"✅ Success! Auto-labeling complete.")
        print(f"   -> Your file is ready to be imported into Doccano:")
        print(f"   -> {output_doccano_file}")
        print("----------------------------------------------------")