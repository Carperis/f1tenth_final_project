# Create combine_poses.py
import os
import pandas as pd
import argparse

def combine_pose_files(input_dir, output_file):
    """
    Reads all CSV files from an input directory, sorts them by filename (timestamp),
    and combines them into a single output CSV file.

    Args:
        input_dir (str): Path to the directory containing individual pose CSV files.
        output_file (str): Path to save the combined CSV file.
    """
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"Error: No CSV files found in directory '{input_dir}'.")
        return

    # Sort files based on the timestamp in the filename
    all_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    all_poses_df = pd.DataFrame()

    print(f"Found {len(all_files)} pose files. Combining...")

    for i, filename in enumerate(all_files):
        file_path = os.path.join(input_dir, filename)
        try:
            # Read each CSV, assuming no header and 7 columns
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] != 7:
                 print(f"Warning: Skipping file '{filename}' - expected 7 columns, found {df.shape[1]}.")
                 continue
            all_poses_df = pd.concat([all_poses_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file '{filename}'.")
        except Exception as e:
            print(f"Warning: Error reading file '{filename}': {e}")

        # Print progress
        if (i + 1) % 100 == 0 or (i + 1) == len(all_files):
            print(f"Processed {i + 1}/{len(all_files)} files...")

    if all_poses_df.empty:
        print("Error: No valid pose data found in any files.")
        return

    # Save the combined data without header and index
    try:
        all_poses_df.to_csv(output_file, header=False, index=False)
        print(f"Successfully combined poses into '{output_file}'.")
    except Exception as e:
        print(f"Error saving combined file '{output_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine individual pose CSV files into a single file.")
    parser.add_argument("input_dir", help="Directory containing the individual pose CSV files.")
    parser.add_argument("--output", default="combined_poses.csv", help="Output file path for the combined poses (default: combined_poses.csv).")

    args = parser.parse_args()

    combine_pose_files(args.input_dir, args.output)