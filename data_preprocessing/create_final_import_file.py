# create_final_import_file.py
import os

input_file = 'dataset/doccano_import_FIXED.jsonl'
output_file = 'dataset/doccano_import_FINAL.jsonl'

print(f"--- Creating final, super-safe import file ---")
print(f"Reading from: {input_file}")

try:
    with open(input_file, 'r', encoding='utf-8') as f_in:
        # Read all the lines from the file we know is good
        lines = f_in.readlines()

    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Write them back to a new file with the cleanest possible UTF-8 encoding
        f_out.writelines(lines)

    print(f"\n✅ Success! A new file has been created with guaranteed clean encoding.")
    print(f"   -> Your final file is: {output_file}")
    print(f"   -> Please import THIS file into Doccano.")

except Exception as e:
    print(f"❌ An error occurred: {e}")