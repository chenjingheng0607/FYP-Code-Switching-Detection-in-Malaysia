import json
import os
import malaya
import warnings
import multiprocessing as mp
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("HF_TOKEN")

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module='malaya.tokenizer')

# ==========================================
# Worker Initialization & Logic
# ==========================================
# We use global variables inside the workers so the model is loaded 
# once per CPU core, rather than 10 million times.
global_word_lang_model = None
LABEL_MAP = {'O': 0, 'MS': 1, 'EN': 2}

def init_worker():
    """This function runs once per CPU core when the multiprocessing pool starts."""
    global global_word_lang_model
    # Load fasttext and substring rules INSIDE the worker
    fasttext_model = malaya.language_detection.fasttext(model='mesolitica/fasttext-language-detection-v1')
    global_word_lang_model = malaya.language_detection.substring_rules(model=fasttext_model)

def process_line(line):
    """This function processes a single sentence."""
    sentence = line.strip()
    if not sentence:
        return None
        
    tokens = sentence.split()
    if not tokens:
        return None
        
    # Predict labels word-by-word
    predicted_labels = global_word_lang_model.predict(tokens)
    
    ner_tags = []
    for pred in predicted_labels:
        if pred == 'EN':
            ner_tags.append(LABEL_MAP['EN'])
        elif pred == 'MS':
            ner_tags.append(LABEL_MAP['MS'])
        else:
            ner_tags.append(LABEL_MAP['O'])
            
    return json.dumps({"tokens": tokens, "ner_tags": ner_tags}, ensure_ascii=False)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    input_file = 'dataset/cleaned_dataset_full.txt'
    output_file = 'dataset/finetuning_dataset_malaya_full.jsonl'
    
    print("\n--- 1. Checking Input File ---")
    if not os.path.exists(input_file):
        print(f"❌ ERROR: Input file '{input_file}' not found.")
        exit()
        
    print(f"--- 2. Counting lines in '{input_file}' ---")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Determine number of CPU cores to use (leaving 1 free so your PC doesn't freeze)
    # num_cores = max(1, mp.cpu_count() - 1)
    num_cores = 8
    print(f"--- 3. Starting Multiprocessing Pool with {num_cores} CPU Cores ---")
    
    processed_count = 0
    
    # Open the input and output files
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Create a multiprocessing Pool
        # chunksize=5000 means each CPU core grabs 5000 sentences at a time (highly efficient)
        with mp.Pool(processes=num_cores, initializer=init_worker) as pool:
            
            # imap_unordered is faster than normal map because it yields results as soon as they are done
            results = pool.imap_unordered(process_line, f_in, chunksize=5000)
            
            for result_json in tqdm(results, total=total_lines, desc="Processing Tokens"):
                if result_json:
                    f_out.write(result_json + '\n')
                    processed_count += 1

    print("\n--- 4. Complete ---")
    print(f"✅ Saved {processed_count} correctly labeled token sequences to: {output_file}")