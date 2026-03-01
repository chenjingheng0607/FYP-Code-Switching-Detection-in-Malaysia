import json
import os

def combine_files_to_jsonl(texts_file, labels_file, final_output_file):
    """
    Reads a text file and a label file line-by-line, combines them
    into JSON objects, and writes them to a final .jsonl file.
    """
    print(f"--- Starting Combination ---")
    print(f"Reading texts from: {texts_file}")
    print(f"Reading labels from: {labels_file}")
    print(f"Writing final dataset to: {final_output_file}")

    # Check if the temporary files exist
    if not os.path.exists(texts_file) or not os.path.exists(labels_file):
        print(f"FATAL ERROR: Could not find temp files '{texts_file}' or '{labels_file}'.")
        print("Did the extraction script run correctly and process more than 0 items?")
        return

    record_count = 0
    try:
        # Open all three files
        with open(texts_file, 'r', encoding='utf-8') as f_texts, \
             open(labels_file, 'r', encoding='utf-8') as f_labels, \
             open(final_output_file, 'w', encoding='utf-8') as f_out:

            # Use zip to read one line from each input file at the same time
            for text_line, label_line in zip(f_texts, f_labels):

                # Create the JSON object
                data_object = {
                    "text": text_line.strip(),  # .strip() removes the newline character
                    "sentence_label": label_line.strip()
                }

                # Write the object to the final .jsonl file
                f_out.write(json.dumps(data_object, ensure_ascii=False) + '\n')
                record_count += 1

                if record_count % 10000 == 0:
                    print(f"  ...combined {record_count} records", end='\r')

    except Exception as e:
        print(f"\nAn error occurred during combination: {e}")
        return

    print(f"\nCombination complete. Total records written: {record_count}")
    print(f"Your final dataset is ready: '{final_output_file}'")
    print("-" * 20)


# --- Main script execution ---
if __name__ == "__main__":
    final_file = "dataset/language_detection_dataset.jsonl"
    combine_files_to_jsonl('dataset/temp_texts.txt', 'dataset/temp_labels.txt', final_file)