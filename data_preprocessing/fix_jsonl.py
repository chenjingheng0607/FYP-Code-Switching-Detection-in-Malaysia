# fix_jsonl.py
import os

input_file = 'dataset/doccano_import_draft.jsonl'
output_file = 'dataset/doccano_import_FIXED.jsonl'

print(f"--- Starting: Cleaning '{input_file}' ---")

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"❌ ERROR: Cannot find the file '{input_file}'. Make sure it's in the same folder.")
else:
    non_empty_lines = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Read the input file line by line
        for line in f_in:
            # If the line is NOT just whitespace...
            if line.strip():
                # ...write it to the new file.
                f_out.write(line)
                non_empty_lines += 1

    print(f"✅ Success! Cleaned the file and removed blank lines.")
    print(f"   -> Found {non_empty_lines} valid lines.")
    print(f"   -> Your new, fixed file is ready: {output_file}")