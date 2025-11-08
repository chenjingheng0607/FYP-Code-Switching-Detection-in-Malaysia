import re
import os

def final_clean_text(text):
    """
    A final, more aggressive cleaning function based on sample inspection.
    """
    if not isinstance(text, str):
        return ""

    # --- Pre-emptive checks: Should we discard this line entirely? ---
    # 1. If the line is mostly just a long hashtag-like word, discard it.
    if len(text.split()) < 5 and re.search(r'[A-Z][a-z0-9]+[A-Z]', text):
        return "" # Example: "ReleaseTheAbductees"
        
    # 2. If the line contains a timestamp pattern, it's likely metadata. Discard.
    if re.search(r'\(\d{2}/\d{2}/\d{4}\)', text) or re.search(r'\[\d{1,2}:\d{2}\s[AP]M\]', text, re.IGNORECASE):
        return "" # Example: "(08/06/2021)...[4:14 PETANG]"

    # --- If the line is kept, proceed with cleaning ---
    
    # Rule 1: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Rule 2: Remove user mentions and standalone (@)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'\(@\)', '', text)
    
    # Rule 3: Remove hashtags but keep the text
    text = re.sub(r'#([^\s]+)', r'\1', text)
    
    # Rule 4: Remove phone numbers (handles various formats)
    text = re.sub(r'\b\d{10,12}\b', '', text) # Simple 10-12 digit numbers
    text = re.sub(r'(\+6)?01\d-?\d{7,8}', '', text) # Malaysian phone numbers
    
    # Rule 5: Standardize punctuation and remove weird characters
    text = re.sub(r'[~=+/]', ' ', text) # Replace certain symbols with a space
    text = re.sub(r'\.{2,}', '.', text) # Replace multiple dots with a single dot
    
    # Rule 6: Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Rule 7: Final check - if a sentence is too short after cleaning, discard it.
    if len(text.split()) < 3: # If there are fewer than 3 words
        return ""
        
    return text

# --- Main script execution ---
if __name__ == "__main__":
    
    input_file = 'dataset/full_raw_sentences.txt'
    output_file = 'dataset/cleaned_dataset.txt'
    
    if not os.path.exists(input_file):
        print(f"❌ ERROR: Input file '{input_file}' not found.")
    else:
        print(f"--- Applying FINAL cleaning to '{input_file}' ---")
        
        cleaned_lines = []
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                cleaned = final_clean_text(line)
                if cleaned:
                    cleaned_lines.append(cleaned)
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in cleaned_lines:
                f_out.write(line + '\n')

        print(f"\n--- Cleaning Complete ---")
        print(f"✅ Your FINAL clean sample file is ready: {output_file}")
        print(f"   ({len(cleaned_lines)} sentences after aggressive cleaning)")