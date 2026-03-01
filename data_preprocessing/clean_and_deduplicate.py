import re
import os
from collections import OrderedDict

def clean_text(text):
    """
    Cleans a single line of text by removing common noise.
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (@username)
    text = re.sub(r'@[^\s]+', '', text)
    # Remove hashtags but keep the text (e.g., #good becomes good)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # Remove leftover HTML entities like &amp;
    text = re.sub(r'&\w+;', '', text)
    # Remove extra whitespace (spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main script execution ---
if __name__ == "__main__":
    
    input_filename = 'dataset/combined_raw_sentences.txt'
    output_filename = 'dataset/final_sentences_for_annotation.txt'
    
    if not os.path.exists(input_filename):
        print(f"❌ ERROR: Input file '{input_filename}' not found. Please run the combination script first.")
    else:
        print(f"--- Starting: Cleaning and Deduplicating '{input_filename}' ---")
        
        # --- Stage 1: Read all lines and find unique ones ---
        print("-> Stage 1/3: Reading all sentences into memory to find uniques...")
        print("   (This may take a few minutes and use a lot of RAM)")
        
        with open(input_filename, 'r', encoding='utf-8') as f:
            # Using a dictionary as an ordered set is a fast way to get unique lines
            # while preserving the original order.
            unique_lines = OrderedDict.fromkeys(line.strip() for line in f)
        
        original_count = sum(1 for line in open(input_filename, 'r', encoding='utf-8'))
        unique_count = len(unique_lines)
        
        print(f"   ...Complete.")
        print(f"   Original sentence count: {original_count}")
        print(f"   Unique sentence count:   {unique_count}")
        
        # --- Stage 2: Clean each unique line ---
        print("\n-> Stage 2/3: Cleaning each unique sentence...")
        
        cleaned_lines = []
        for i, line in enumerate(unique_lines):
            cleaned = clean_text(line)
            # Only add the sentence if it's not empty after cleaning
            if cleaned:
                cleaned_lines.append(cleaned)
            
            if (i + 1) % 100000 == 0:
                print(f"   ...cleaned {i + 1}/{unique_count} sentences", end='\r')
        
        print(f"\n   ...Complete. Total sentences after cleaning: {len(cleaned_lines)}")
        
        # --- Stage 3: Save the final, clean dataset ---
        print(f"\n-> Stage 3/3: Saving final dataset to '{output_filename}'...")
        
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            for line in cleaned_lines:
                f_out.write(line + '\n')
                
        print("\n----------------------------------------------------")
        print(f"✅ All Done! Your final, clean dataset is ready.")
        print(f"   -> File saved as: {output_filename}")
        print("----------------------------------------------------")