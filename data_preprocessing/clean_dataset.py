import re
import os
import string
from tqdm import tqdm

def final_clean_text(text):
    """
    Cleaning function optimized for mBERT and XLM-RoBERTa tokenization.
    """
    if not isinstance(text, str):
        return ""

    # --- Pre-emptive checks ---
    # 1. If the line is mostly just a long camelCase hashtag word, discard it.
    if len(text.split()) < 5 and re.search(r'^[A-Z][a-z0-9]+[A-Z]', text):
        return "" # Example: "ReleaseTheAbductees"

    # --- Text Cleaning ---
    
    # Rule 1: Remove metadata (timestamps) INSTEAD of discarding the whole line
    # Matches (08/06/2021) or [4:14 PM] / [4:14 PETANG]
    text = re.sub(r'\(\d{2}/\d{2}/\d{4}\)', ' ', text)
    text = re.sub(r'\[\d{1,2}:\d{2}\s*(?:AM|PM|PETANG|PAGI|MALAM)?\]', ' ', text, flags=re.IGNORECASE)
    
    # Rule 2: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Rule 3: Remove user mentions and standalone (@)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'\(@\)', '', text)
    
    # Rule 4: Remove hashtags but keep the text (e.g., "#makan" -> "makan")
    text = re.sub(r'#([^\s]+)', r'\1', text)
    
    # Rule 5: Remove phone numbers (handles various formats)
    text = re.sub(r'\b\d{10,12}\b', '', text) # Simple 10-12 digit numbers
    text = re.sub(r'(\+6)?01\d-?\d{7,8}', '', text) # Malaysian phone numbers
    
    # Rule 6: Handle Special Characters and Ellipses
    text = re.sub(r'[~=+/|]', ' ', text) # Replace certain useless symbols with a space
    text = re.sub(r'\.{2,}', ' ... ', text) # Standardize ellipses and add space
    
    # Rule 7: PAD PUNCTUATION WITH SPACES (Critical for Token-Level tasks)
    # This separates punctuation from words so `split()` doesn't attach them.
    # We avoid splitting apostrophes (e.g., "don't", "tak'kan") so words stay intact.
    text = re.sub(r'([.,!?()\[\]{};:"”"“\-])', r' \1 ', text)
    
    # Rule 8: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Rule 9: Final check - if a sentence is too short after cleaning, discard it.
    # We increased this slightly to 4, as spacing punctuation increases the token count
    if len(text.split()) < 4: 
        return ""
        
    return text

# --- Main script execution ---
if __name__ == "__main__":
    
    input_file = 'dataset/full_raw_sentences.txt'
    output_file = 'dataset/cleaned_dataset_full.txt'
    
    if not os.path.exists(input_file):
        print(f"❌ ERROR: Input file '{input_file}' not found.")
    else:
        print(f"--- Applying FINAL cleaning to '{input_file}' ---")
        
        print("Calculating total lines for progress bar...")
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        cleaned_lines = []
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, total=total_lines, desc="Cleaning sentences"):
                cleaned = final_clean_text(line)
                if cleaned:
                    cleaned_lines.append(cleaned)
        
        print("Saving cleaned dataset...")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in tqdm(cleaned_lines, desc="Writing to file"):
                f_out.write(line + '\n')

        print(f"\n--- Cleaning Complete ---")
        print(f"✅ Your FINAL clean sample file is ready: {output_file}")
        print(f"   ({len(cleaned_lines)} sentences successfully cleaned and kept)")