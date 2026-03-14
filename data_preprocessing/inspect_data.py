import json

def preview_jsonl(filename, num_lines=5):
    """
    Reads and prints the first few lines of a JSON Lines (.jsonl) file.
    """
    print(f"--- 🕵️  Previewing first {num_lines} lines of: {filename} ---")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    # Parse the JSON string from the line
                    data = json.loads(line)

                    # Pretty-print the JSON object for readability
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    print("-" * 25) # Separator

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line {i+1} as JSON. Content: {line.strip()}")
                    print("-" * 25)

    except FileNotFoundError:
        print(f"❌ Error: The file '{filename}' was not found.")
        print("Please make sure it's in the same directory as this script.")

    print(f"--- ✅ End of preview for: {filename} ---\n")


# --- Main script execution ---
if __name__ == "__main__":
    # The list of all your dataset files
    files_to_inspect = [
        "dataset/finetuning_dataset_malaya_full.jsonl"
    ]

    for file in files_to_inspect:
        preview_jsonl(file)