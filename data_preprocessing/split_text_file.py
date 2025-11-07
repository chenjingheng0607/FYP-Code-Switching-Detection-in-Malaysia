# split_text_file.py
import os

def split_file(input_path, lines_per_file=5000000):
    """Splits a large text file into smaller chunks."""
    
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Input file not found: {input_path}")
        return

    print(f"--- Splitting '{input_path}' into smaller files... ---")
    
    file_count = 1
    line_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        # Create the first output file
        output_path = f"{os.path.splitext(input_path)[0]}_part_{file_count}.txt"
        f_out = open(output_path, 'w', encoding='utf-8')
        print(f"-> Creating '{output_path}'")

        for line in f_in:
            if line_count > 0 and line_count % lines_per_file == 0:
                f_out.close()
                file_count += 1
                output_path = f"{os.path.splitext(input_path)[0]}_part_{file_count}.txt"
                f_out = open(output_path, 'w', encoding='utf-8')
                print(f"-> Creating '{output_path}'")
            
            f_out.write(line)
            line_count += 1
            
        f_out.close()
        
    print("\n✅ Splitting complete!")


if __name__ == "__main__":
    large_file = 'dataset/final_sentences_for_annotation.txt'
    # Splitting into chunks of 5 million lines. This should result in 
    # files of about 500-600MB each, well under the 1GB limit.
    split_file(large_file, lines_per_file=5000000)