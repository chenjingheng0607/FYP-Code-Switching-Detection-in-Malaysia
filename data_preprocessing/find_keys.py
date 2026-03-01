import re
import os

def find_array_keys_in_chunks(file_path, chunk_size=4096):
    """
    Reads a large file in small chunks to find all keys that are
    followed by an opening square bracket '['. This method is designed
    to use very little memory, even for single-line multi-GB files.
    """
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Searching for array keys in '{file_path}' using a low-memory method...")

    # Regex to find a pattern like: "any_key_name": [
    # We use a bytes pattern because we will read the file in binary mode for safety.
    pattern = re.compile(b'"([^"]+)":\\s*\\[')
    
    found_keys = set()
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                # Read a small, safe chunk of the file
                chunk = f.read(chunk_size)
                
                # If chunk is empty, we've reached the end of the file
                if not chunk:
                    break
                
                # Find all matches in the current chunk
                matches = pattern.findall(chunk)
                for match in matches:
                    # Decode the found key from bytes to a string and add it to our set
                    # The set automatically handles duplicates for us.
                    found_keys.add(match.decode('utf-8', errors='ignore'))

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")
        return

    if found_keys:
        print("\nSuccess! Found the following keys that are followed by a list '[':")
        for key in found_keys:
            print(f"- {key}")
        print("\nPlease use these key names in the 'extract_data.py' script.")
    else:
        print("\nSearch complete, but no matching keys were found.")

# --- Main script execution ---
if __name__ == "__main__":
    large_file = "dataset/train-test.json"
    find_array_keys_in_chunks(large_file)