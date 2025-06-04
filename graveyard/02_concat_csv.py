import pandas as pd
import glob
import os

def concat_csv_files(input_dir: str, output_file: str) -> None:
    """
    Concatenates all CSV files in the input_dir into a single output_file.
    """
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    # Sort files to ensure consistent order if needed, e.g., by filename
    csv_files.sort()

    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: {f} is empty and will be skipped.")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No data to concatenate after processing files.")
        return

    combined_csv = pd.concat(all_data, ignore_index=True)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    combined_csv.to_csv(output_file, index=False)
    print(f"Successfully concatenated {len(all_data)} CSV files into {output_file}")

if __name__ == '__main__':
    input_directory = 'simpleqa_output_openrouter_qwen3_32b'
    output_csv_file = 'output/combined_qwen3_32b_openrouter.csv'
    concat_csv_files(input_directory, output_csv_file) 