# benchmark_utils/process_angles.py
import os
import pandas as pd
import numpy as np
import glob
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import benchmark_utils.file_utils as fileutil

# --- Configuration ---
# Define the base directory where your 3D pose estimation CSVs are located.
# This script will process data from this folder.
# NOTE: This assumes one primary source of video data. If you need to process
# MotionBERT, MMPose, etc., separately, you'd run this script for each one
# by changing the CSV_INPUT_BASE_DIR.
CSV_INPUT_BASE_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/benchmark/jointangles/jointangles_motionbert" # Example: Using MotionBERT data

# Define the directory where the new angle CSVs will be saved.
ANGLE_OUTPUT_BASE_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/jointangles_motionbert_dataset"

# Define patterns to find subjects and trial files
SUBJECT_DIRS_PATTERN = "S*"
TRIAL_FILE_PATTERN = "S*_A01_T*.csv" # Match the pose estimation CSVs
SAMPLING_RATE_HZ = 30.0

# --- Main Processing Function ---

def process_and_save_all_angles():
    """
    Iterates through all subjects and trials, calculates all defined joint angles
    from 3D keypoint data, and saves them to new CSV files.
    """
    print(f"Starting angle processing...")
    print(f"Input data source: {CSV_INPUT_BASE_DIR}")
    print(f"Output directory: {ANGLE_OUTPUT_BASE_DIR}")

    # Find all subject directories
    subject_dirs = sorted(glob.glob(os.path.join(CSV_INPUT_BASE_DIR, SUBJECT_DIRS_PATTERN)))
    if not subject_dirs:
        print(f"Error: No subject directories found matching pattern '{SUBJECT_DIRS_PATTERN}' in '{CSV_INPUT_BASE_DIR}'")
        return

    total_files_processed = 0
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        print(f"\nProcessing Subject: {subject_id}")

        # Find all trial files for this subject
        trial_files = sorted(glob.glob(os.path.join(subj_dir, TRIAL_FILE_PATTERN)))

        if not trial_files:
            print(f"  No trial files found for subject {subject_id}.")
            continue

        # Create corresponding output directory for this subject
        output_subject_dir = os.path.join(ANGLE_OUTPUT_BASE_DIR, subject_id)
        os.makedirs(output_subject_dir, exist_ok=True)

        for csv_filepath in trial_files:
            try:
                # Load the 3D keypoint data
                dfcsv = pd.read_csv(csv_filepath)
                if dfcsv is None or dfcsv.empty:
                    print(f"  Skipping empty or unreadable file: {os.path.basename(csv_filepath)}")
                    continue
                
                print(f"  Processing file: {os.path.basename(csv_filepath)}")

                # Calculate all joint angles using the utility function
                # This function returns 6 angle arrays:
                # arm_flex_r, arm_flex_l, elbow_flex_r, elbow_flex_l, knee_angle_r, knee_angle_l
                angle_data_arrays = fileutil.getMainJointAnglesFromCSV2(dfcsv)
                
                angle_names = [
                    'arm_flex_r', 'arm_flex_l', 
                    'elbow_flex_r', 'elbow_flex_l',
                    'knee_angle_r', 'knee_angle_l'
                ]

                # Create the time vector
                num_frames = len(angle_data_arrays[0]) # Get length from the first angle array
                time_vector = np.arange(num_frames) / SAMPLING_RATE_HZ

                # Construct the output DataFrame
                output_data = {'time': time_vector}
                for i, name in enumerate(angle_names):
                    output_data[name] = angle_data_arrays[i]
                
                output_df = pd.DataFrame(output_data)

                # Define the output filename
                base_filename = os.path.basename(csv_filepath)
                # Create a new, clean name for the output file
                output_filename = os.path.splitext(base_filename)[0] + ".csv"
                output_filepath = os.path.join(output_subject_dir, output_filename)

                # Save the DataFrame to a new CSV file
                output_df.to_csv(output_filepath, index=False, float_format='%.5f')
                print(f"    -> Saved angles to: {output_filepath}")
                total_files_processed += 1

            except Exception as e:
                print(f"  ERROR processing file {os.path.basename(csv_filepath)}: {e}")

    print(f"\nProcessing complete. Total files created: {total_files_processed}")


# --- Script Execution ---

if __name__ == '__main__':
    process_and_save_all_angles()