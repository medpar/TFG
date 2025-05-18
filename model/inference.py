# model/inference.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob # For finding .raw files

import config
from model import GaitLSTM
import data_loader # For GaitPhaseDataset and create_sequences
import utils

# Attempt to make gaitseg_utils accessible for plotting
# This assumes 'model' and 'gaitseg' are sibling directories
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR) # VIDIMU directory in your case
GAITSEG_DIR = os.path.join(GRANDPARENT_DIR, "gaitseg") # Path to gaitseg directory

if GAITSEG_DIR not in sys.path:
    sys.path.append(GAITSEG_DIR)

try:
    import gaitseg_utils as gal
except ImportError:
    print("Warning: Could not import gaitseg_utils. Inference plotting against angular velocity will be limited.")
    gal = None


class InferenceDataset(Dataset):
    """
    A dataset tailored for inference, processing one file at a time
    and keeping track of original timestamps and features for output.
    """
    def __init__(self, filepath, sequence_length, feature_cols, target_col, time_col, scaler=None):
        self.filepath = filepath
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col 
        self.time_col = time_col
        self.scaler = scaler

        self.sequences_data = [] 
        self.all_trial_dfs = {} 
        self._load_and_create_sequences_for_file()

    def _load_and_create_sequences_for_file(self):
        try:
            df = pd.read_csv(self.filepath)
            if df.empty:
                print(f"Warning: File {self.filepath} is empty.")
                return
        except Exception as e:
            print(f"Error reading {self.filepath}: {e}")
            return

        self.all_trial_dfs[self.filepath] = df.copy() 

        if not all(col in df.columns for col in self.feature_cols):
            print(f"Warning: Missing feature columns in {self.filepath}. Available: {df.columns.tolist()}")
            return
        
        # features_original = df[self.feature_cols].values.copy() 
        
        if self.scaler:
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        
        features_scaled = df[self.feature_cols].values
        
        # targets = df[self.target_col].values if self.target_col in df.columns else np.full(len(df), -1)
        # targets_original = targets.copy()
        # times = df[self.time_col].values if self.time_col in df.columns else np.arange(len(df))
        
        num_samples = len(df)
        if num_samples < self.sequence_length:
            print(f"Warning: File {self.filepath} has {num_samples} samples, less than sequence length {self.sequence_length}. Skipping.")
            return
            
        for i in range(num_samples - self.sequence_length + 1):
            feature_seq = features_scaled[i : i + self.sequence_length]
            self.sequences_data.append({
                "feature_seq": feature_seq,
                "original_start_index": i, 
                "filepath": self.filepath
            })
        
        if not self.sequences_data:
            print(f"No sequences created for {self.filepath} (length {num_samples}, seq_len {self.sequence_length})")


    def __len__(self):
        return len(self.sequences_data)

    def __getitem__(self, idx):
        data_item = self.sequences_data[idx]
        feature_tensor = torch.tensor(data_item["feature_seq"], dtype=torch.float32)
        return feature_tensor, data_item["original_start_index"], data_item["filepath"]

    def get_original_df(self, filepath):
        return self.all_trial_dfs.get(filepath)

def predict(model, data_loader, device):
    model.eval()
    file_predictions = {} 

    with torch.no_grad():
        for features_batch, start_indices_batch, filepaths_batch in tqdm(data_loader, desc="Inferencing"):
            features_batch = features_batch.to(device)
            
            outputs_batch = model(features_batch) 
            _, predicted_classes_batch = torch.max(outputs_batch, 2) 
            
            predicted_classes_batch = predicted_classes_batch.cpu().numpy()
            start_indices_batch = start_indices_batch.cpu().numpy()

            for i in range(features_batch.size(0)): 
                filepath = filepaths_batch[i]
                seq_preds = predicted_classes_batch[i] 
                seq_start_idx_original = start_indices_batch[i]

                if filepath not in file_predictions:
                    file_predictions[filepath] = {}
                
                for step_in_seq, pred_class in enumerate(seq_preds):
                    original_file_idx = seq_start_idx_original + step_in_seq
                    file_predictions[filepath][original_file_idx] = pred_class
                    
    return file_predictions


def find_corresponding_raw_file(csv_filepath):
    """
    Tries to find the corresponding .raw IMU file for a given _corrected.csv or _predictions.csv.
    Assumes .raw files are in a structure like:
    .../VIDIMU/benchmark/jointangles/jointangles_imus/SXX/SXX_A01_TYY.raw
    and CSVs are from .../VIDIMU/gaitseg/SXX/SXX_A01_TYY_corrected.csv
    or .../VIDIMU/model/results/inference_results/SXX_A01_TYY_predictions.csv
    """
    if gal is None: return None

    base_csv_name = os.path.basename(csv_filepath)
    
    # Try to extract subject and trial identifier
    # e.g., S40_A01_T01 from S40_A01_T01_corrected.csv or S40_A01_T01_predictions.csv
    parts = base_csv_name.split('_')
    if len(parts) < 3: # Expecting at least SXX_A01_TXX
        print(f"Could not parse subject/trial from CSV name: {base_csv_name}")
        return None
    
    subject_id = parts[0]
    activity_id = parts[1] # A01
    trial_num_id = parts[2] # TXX

    raw_filename_stem = f"{subject_id}_{activity_id}_{trial_num_id}" # e.g., S40_A01_T01
    
    # Construct path to where .raw files are expected (defined in gaitseg_utils)
    # GUI_BASE_DATA_DIR in gaitseg_gui.py points to gal.root_dir
    # gal.root_dir = '/Users/mario/Documents/TFG_VIDIMU/vidiMU/benchmark/jointangles/jointangles_imus'
    
    # Check if gal.root_dir is defined and accessible
    if not hasattr(gal, 'root_dir') or not gal.root_dir:
        print("Warning: gaitseg_utils.root_dir not defined. Cannot locate .raw files.")
        # Fallback: try to guess relative to the CSV path's grandparent (VIDIMU folder)
        # This is fragile and depends on your exact directory structure.
        # Example: csv_filepath = .../VIDIMU/gaitseg/S40/S40_A01_T01.csv
        # grandparent = .../VIDIMU
        vidimu_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(csv_filepath))))
        raw_search_dir_base = os.path.join(vidimu_root_guess, "benchmark", "jointangles", "jointangles_imus")
        if not os.path.isdir(raw_search_dir_base):
            print(f"Fallback raw search directory does not exist: {raw_search_dir_base}")
            return None
    else:
        raw_search_dir_base = gal.root_dir

    raw_file_path = os.path.join(raw_search_dir_base, subject_id, f"{raw_filename_stem}.raw")

    if os.path.exists(raw_file_path):
        return raw_file_path
    else:
        # Try searching more broadly if specific path failed (e.g. if CSVs are not in `gaitseg` folder)
        # This part is more complex if the relation between CSV and RAW dir isn't fixed.
        # For now, stick to the expected path based on gal.root_dir.
        print(f"Corresponding .raw file not found at expected path: {raw_file_path}")
        # Alternative search: if gal.root_dir is the base IMU data dir (e.g., .../vidiMU/benchmark/IMU_data/...)
        # This would require knowing the structure of where original .raw files are.
        # Let's assume the GUI_BASE_DATA_DIR in gaitseg_gui (which is gal.root_dir) is correct.
        return None


def run_inference(model_path, inference_input_path, output_csv_dir):
    """
    Main function to run inference.
    inference_input_path: Can be a directory of _corrected.csv files (e.g., gaitseg_corrected)
                         or a single _corrected.csv file.
    """
    print(f"--- Starting Inference ---")
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    if checkpoint is None:
        print(f"Failed to load checkpoint from {model_path}")
        return

    model = GaitLSTM(
        input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=0, linear_dropout=0
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    if scaler is None and config.NORMALIZATION_METHOD is not None:
        print("Warning: Scaler not found in checkpoint. Predictions may be inaccurate if model was trained with scaling.")

    if os.path.isdir(inference_input_path):
        # Expecting _corrected.csv files as input for inference
        files_to_process = []
        for subj_folder in glob.glob(os.path.join(inference_input_path, config.SUBJECT_DIRS_PATTERN)):
            if os.path.isdir(subj_folder):
                 # Pattern should match _corrected.csv or _labeled.csv, etc.
                files_to_process.extend(glob.glob(os.path.join(subj_folder, "*.csv")))
        # Filter for files that are likely corrected data, not prediction outputs
        files_to_process = [f for f in files_to_process if "_corrected" in f or "_labeled" in f and "_predictions" not in f]

    elif os.path.isfile(inference_input_path) and inference_input_path.endswith(".csv"):
        files_to_process = [inference_input_path]
    else:
        print(f"Error: {inference_input_path} is not a valid CSV file or directory of CSVs.")
        return

    if not files_to_process:
        print(f"No suitable CSV files found for inference in {inference_input_path}.")
        return

    os.makedirs(output_csv_dir, exist_ok=True)

    aggregated_true_phases = []
    aggregated_pred_phases = []

    for csv_filepath in files_to_process:
        print(f"\nProcessing CSV file: {csv_filepath}")
        inference_dataset = InferenceDataset(
            filepath=csv_filepath, sequence_length=config.SEQUENCE_LENGTH,
            feature_cols=config.FEATURE_COLUMNS, target_col=config.TARGET_COLUMN,
            time_col=config.TIME_COLUMN, scaler=scaler
        )
        if len(inference_dataset) == 0:
            print(f"Skipping {csv_filepath} due to no sequences created.")
            continue

        inference_loader = DataLoader(inference_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)
        raw_predictions_map = predict(model, inference_loader, config.DEVICE)
        
        original_df = inference_dataset.get_original_df(csv_filepath)
        if original_df is None: continue

        num_original_samples = len(original_df)
        trial_predicted_phases = np.full(num_original_samples, -1, dtype=int)

        if csv_filepath in raw_predictions_map:
            preds_for_file = raw_predictions_map[csv_filepath]
            for original_idx, pred_class in preds_for_file.items():
                if 0 <= original_idx < num_original_samples:
                    trial_predicted_phases[original_idx] = pred_class
        
        # Create output DataFrame
        output_df = original_df[[config.TIME_COLUMN] + config.FEATURE_COLUMNS].copy()
        output_df['predicted_phase_model'] = trial_predicted_phases
        
        true_phases_available = False
        if config.TARGET_COLUMN in original_df.columns:
            output_df['true_phase_gui'] = original_df[config.TARGET_COLUMN]
            true_phases_available = True
            
            # For overall metrics calculation
            valid_preds_mask = (trial_predicted_phases != -1)
            if np.any(valid_preds_mask):
                aggregated_true_phases.extend(original_df[config.TARGET_COLUMN][valid_preds_mask].values)
                aggregated_pred_phases.extend(trial_predicted_phases[valid_preds_mask])
        
        # Save predictions to CSV
        base_filename = os.path.basename(csv_filepath)
        # Make output name distinct, e.g. S40_A01_T01_corrected_predictions.csv
        output_filename = os.path.splitext(base_filename)[0] + "_model_predictions.csv"
        output_filepath = os.path.join(output_csv_dir, output_filename)
        output_df.to_csv(output_filepath, index=False)
        print(f"Saved model predictions for {base_filename} to {output_filepath}")

        # --- Individual File Plotting ---
        plot_title_suffix = f" ({base_filename})"

        # 1. Plot: Model Predicted Phases vs. True GUI Phases (if available)
        if true_phases_available:
            utils.plot_model_predictions_vs_true_phases(
                timestamps=original_df[config.TIME_COLUMN].values,
                true_phases=original_df[config.TARGET_COLUMN].values,
                predicted_phases=trial_predicted_phases,
                title_suffix=plot_title_suffix
            )
        else: # Plot just model predictions if no true phases
             utils.plot_model_predictions_vs_true_phases(
                timestamps=original_df[config.TIME_COLUMN].values,
                true_phases=None, # No true phases
                predicted_phases=trial_predicted_phases,
                title_suffix=plot_title_suffix + " (No True Phases)"
            )


        # 2. Plot: Model Predicted Phases vs. Angular Velocity
        raw_imu_file_for_plot = find_corresponding_raw_file(csv_filepath)
        if raw_imu_file_for_plot and gal:
            utils.plot_model_predictions_vs_angular_velocity(
                timestamps_csv=original_df[config.TIME_COLUMN].values,
                predicted_phases_csv=trial_predicted_phases,
                raw_imu_filepath=raw_imu_file_for_plot,
                title_suffix=plot_title_suffix
            )
        else:
            print(f"Skipping angular velocity plot for {base_filename}: .raw file not found or gaitseg_utils not available.")


    # --- Overall Metrics (if true labels were available across files) ---
    if aggregated_true_phases and aggregated_pred_phases:
        print("\n--- Overall Inference Metrics (on samples with predictions and true labels) ---")
        metrics = utils.calculate_metrics(aggregated_true_phases, aggregated_pred_phases)
        print(metrics)
        class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)] # Assumes classes 0, 1, 2...
        # Adjust class_names if your phases are different (e.g., if -1 is a valid predicted class to evaluate)
        # For CM, usually only evaluate on the actual classes (0,1,2).
        # Filter out -1 if they are not part of the evaluation classes
        true_eval = [p for p in aggregated_true_phases if p in range(config.NUM_CLASSES)]
        pred_eval = [p for idx, p in enumerate(aggregated_pred_phases) if aggregated_true_phases[idx] in range(config.NUM_CLASSES)]

        if true_eval and pred_eval:
             utils.plot_confusion_matrix_custom(true_eval, pred_eval, class_names, title="Overall Inference CM")
        else:
            print("Not enough valid class data for overall confusion matrix after filtering.")
            
    else:
        print("\nNo true labels found across inference files for overall metrics calculation.")

    print("\n--- Inference Finished ---")


if __name__ == '__main__':
    # Example usage:
    # Infer on the _corrected.csv files produced by the GUI
    inference_source_dir = os.path.expanduser("~/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected") # From gaitseg_gui.py
    
    output_prediction_dir = os.path.join(config.OUTPUT_DIR, "inference_model_vs_imus")
    
    if not os.path.exists(config.DEFAULT_MODEL_PATH):
        print(f"Warning: Default model path {config.DEFAULT_MODEL_PATH} not found. Inference will likely fail.")
        print("Please train a model first or provide a valid model path.")
    elif not os.path.isdir(inference_source_dir) or not os.listdir(inference_source_dir):
        print(f"Warning: Inference source directory {inference_source_dir} is empty or does not exist.")
        print("Please ensure you have _corrected.csv files from the GUI.")
    else:
         run_inference(
            model_path=config.DEFAULT_MODEL_PATH, 
            inference_input_path=inference_source_dir,
            output_csv_dir=output_prediction_dir
        )