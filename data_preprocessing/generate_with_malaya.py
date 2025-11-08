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

def create_labeled_example(sentence, eng_words, ms_words, lang_model, tokenizer, label_map):
    """
    Labels a sentence using a hybrid dictionary and a Malaya model,
    and returns a dictionary with 'tokens' and 'ner_tags'.
    """
    tokens = tokenizer.tokenize(sentence)
    ner_tags = []

    for token in tokens:
        word = token.lower()
        label = 'O'

        if not re.search('[a-zA-Z]', word):
             label = 'O'
        else:
            is_english = word in eng_words
            is_malay = word in ms_words
            
            if is_english and not is_malay:
                label = 'B-EN'
            elif is_malay and not is_english:
                label = 'B-MS'
            else:
                if len(word) > 2:
                    prediction = lang_model.predict([word])[0]
                    if prediction in ('eng', 'Indo-European'):
                        label = 'B-EN'
                    elif prediction in ('malay', 'ind', 'other'):
                        label = 'B-MS'
                    else:
                        label = 'O'
                else:
                    label = 'B-MS' if is_malay else 'B-EN' if is_english else 'O'

        ner_tags.append(label_map[label])
        
    return {"tokens": tokens, "ner_tags": ner_tags}

# --- Main script execution ---
if __name__ == "__main__":
    
    input_file = 'dataset/cleaned_sample_10k.txt'
    output_file = 'dataset/finetuning_dataset_malaya.jsonl'
    
    LABEL_MAP = {
        'O': 0, 'B-MS': 1, 'I-MS': 2, 'B-EN': 3, 'I-EN': 4
    }
    
    print("--- 1. Loading Dictionaries ---")
    # Corrected the filenames to match your output
    english_words = load_word_list('dataset/words_alpha.txt')
    malay_words = load_word_list('dataset/Malays.dic.txt')
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
    print(f"✅ Auto-labeling finished. Processed {processed_count} sentences.")
    print(f"   -> Your final fine-tuning dataset is ready: {output_file}")
    print("\n   Example of the first few lines of the output file:")
    os.system(f"head -n 3 {output_file}")