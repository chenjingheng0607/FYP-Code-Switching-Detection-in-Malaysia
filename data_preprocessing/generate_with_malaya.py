# generate_with_malaya.py (Corrected)
import json
import re
import os
import malaya
import warnings

# Suppress the specific FutureWarning from Malaya's regex
warnings.filterwarnings("ignore", category=FutureWarning, module='malaya.tokenizer')

def load_word_list(filepath):
    """Loads a word list into a set for fast lookups."""
    print(f"-> Loading word list from '{filepath}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = set(line.strip().lower() for line in f if line.strip())
        print(f"   ...loaded {len(words)} words.")
        return words
    except FileNotFoundError:
        print(f"   ...ERROR: File not found. Please make sure '{filepath}' is in the same folder.")
        return None

def create_labeled_example_optimized(sentence, eng_words, ms_words, tagger, tokenizer, label_map):
    """
    Optimized to use context.
    """
    tokens = tokenizer.tokenize(sentence)
    ner_tags = []
    
    # 1. Let Malaya tag the whole sentence in one pass (Fast and Context-Aware)
    # Output looks like: [('I', 'EN'), ('makan', 'MS'), ('nasi', 'MS')]
    tagged_sentence = tagger.predict(sentence)
    
    # 2. Iterate through the tokens and format them
    for word, predicted_lang in tagged_sentence:
        word_lower = word.lower()
        
        # Handle Punctuation/Numbers
        if not re.search('[a-zA-Z]', word_lower):
            label = 'O'
        else:
            # Map Malaya's output to your labels
            if predicted_lang in ['ENG', 'EN']:
                label = 'EN'
            elif predicted_lang in ['MALAY', 'MS']:
                label = 'MS'
            else:
                label = 'O'
                
            # Optional: Override the model if you are 100% sure based on your dictionaries
            # (Only do this if the word is strictly in one dict and not the other)
            if word_lower in eng_words and word_lower not in ms_words:
                label = 'EN'
            elif word_lower in ms_words and word_lower not in eng_words:
                label = 'MS'

        ner_tags.append(label_map.get(label, 0)) # Default to 'O' (0) if not found
        
    return {"tokens": [t[0] for t in tagged_sentence], "ner_tags": ner_tags}

# --- Main script execution ---
if __name__ == "__main__":
    
    input_file = 'dataset/cleaned_dataset_full.txt'
    output_file = 'dataset/finetuning_dataset_malaya_full.jsonl'
    
    LABEL_MAP = {
        'O': 0, 'MS': 1, 'EN': 2
    }
    
    print("--- 1. Loading Dictionaries ---")
    # Corrected the filenames to match your output
    english_words = load_word_list('dataset/words_alpha.txt')
    malay_words = load_word_list('dataset/malay-text.txt')
    if not english_words or not malay_words:
        exit()
        
    print("\n--- 2. Loading Malaya Models ---")
    print("   (This may download models on the first run)...")
    lang_detection_model = malaya.language_detection.fasttext(model='mesolitica/fasttext-language-detection-v1')
    
    # ***** THE ONLY CHANGE IS ON THIS LINE *****
    word_tokenizer = malaya.tokenizer.Tokenizer()
    
    print("   ...Malaya models loaded successfully.")

    print(f"\n--- 3. Generating fine-tuning data from '{input_file}' ---")
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            sentence = line.strip()
            if sentence:
                labeled_example = create_labeled_example(
                    sentence, english_words, malay_words, lang_detection_model, word_tokenizer, LABEL_MAP
                )
                f_out.write(json.dumps(labeled_example, ensure_ascii=False) + '\n')
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"   ...processed {processed_count} sentences", end='\r')

    print(f"\n\n--- 4. Complete ---")
    print(f"Auto-labeling finished. Processed {processed_count} sentences.")
    print(f"   -> Your final fine-tuning dataset is ready: {output_file}")
    print("\n   Example of the first few lines of the output file:")
    os.system(f"head -n 3 {output_file}")