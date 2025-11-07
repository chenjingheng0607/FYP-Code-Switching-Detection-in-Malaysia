# convert_to_conll.py
import json
import re

def convert_jsonl_to_conll(jsonl_path, conll_path):
    """
    Converts a Doccano-formatted JSONL file for sequence labeling
    into the CoNLL 2003 format.
    """
    print(f"--- Starting conversion from JSONL to CoNLL ---")
    print(f"Input file: {jsonl_path}")
    print(f"Output file: {conll_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(conll_path, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            try:
                data = json.loads(line)
                text = data['text']
                labels = data['labels']
                
                # Create a lookup array for labels, default to 'O'
                token_labels = ['O'] * len(text)
                for start, end, label in labels:
                    # Mark the beginning of the entity with B-
                    token_labels[start] = f"B-{label}"
                    # Mark the inside of the entity with I-
                    for j in range(start + 1, end):
                        token_labels[j] = f"I-{label}"

                # Simple tokenization: split by space and punctuation
                last_pos = 0
                for match in re.finditer(r'\w+|[^\w\s]', text):
                    token_text = match.group(0)
                    token_start = match.start()
                    
                    # Find the label for the first character of the token
                    # This is a simple but effective way to assign token-level labels
                    label = token_labels[token_start]
                    
                    # Write in "word<TAB>label" format
                    f_out.write(f"{token_text}\t{label}\n")

                # Write a newline to separate sentences
                f_out.write('\n')

            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                print(f"Problematic line: {line.strip()}")
                continue
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    input_file = "dataset/doccano_import_FIXED.jsonl"
    output_file = "dataset/doccano_import_conll.txt"
    convert_jsonl_to_conll(input_file, output_file)