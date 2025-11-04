import ijson
import os

def extract_key_to_file(input_file, key_to_extract, output_file):
    """
    Streams a large JSON file, extracts all items from a specified key's array,
    and writes each item to a new file, one per line.
    """
    print(f"--- Starting Extraction ---")
    print(f"Reading from: {input_file}")
    print(f"Looking for key: '{key_to_extract}'")
    print(f"Writing to: {output_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"FATAL ERROR: Input file '{input_file}' not found.")
        return

    item_count = 0
    try:
        # Open the input file in binary mode
        with open(input_file, 'rb') as f_in:
            # Create a generator for the items in the specified key's array
            items_generator = ijson.items(f_in, f'{key_to_extract}.item')

            # Open the output file in text mode
            with open(output_file, 'w', encoding='utf-8') as f_out:
                # Loop through the generator and write each item
                for item in items_generator:
                    # ijson might return various types, ensure it's a string for writing
                    f_out.write(str(item) + '\n')
                    item_count += 1
                    if item_count % 10000 == 0:
                        print(f"  ...extracted {item_count} items", end='\r')

    except Exception as e:
        print(f"\nAn error occurred during extraction: {e}")
        return

    print(f"\nExtraction complete. Total items written: {item_count}")
    print("-" * 20)


# --- Main script execution ---
if __name__ == "__main__":
    large_file = "dataset/train-test.json"

    # Stage 1: Extract train_X (the sentences)
    extract_key_to_file(large_file, 'train_X', 'dataset/temp_texts.txt')

    # Stage 2: Extract train_Y (the labels)
    extract_key_to_file(large_file, 'train_Y', 'dataset/temp_labels.txt')