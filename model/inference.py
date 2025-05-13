# inference.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from model import GaitLSTM
import data_loader # For GaitPhaseDataset and create_sequences
import utils

class InferenceDataset(Dataset):
    """
    A dataset tailored for inference, processing one file at a time
    and keeping track of original timestamps and features for output.
    """
    def __init__(self, filepath, sequence_length, feature_cols, target_col, time_col, scaler=None):
        self.filepath = filepath
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col # May or may not be present in inference data
        self.time_col = time_col
        self.scaler = scaler

        self.sequences_data = [] # Stores (feature_seq, original_full_targets, original_full_times, original_full_features)
        self.all_trial_dfs = {} # Stores original DataFrames for reconstruction {filepath: df}
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

        self.all_trial_dfs[self.filepath] = df.copy() # Store original df

        # Select features
        if not all(col in df.columns for col in self.feature_cols):
            print(f"Warning: Missing feature columns in {self.filepath}. Available: {df.columns.tolist()}")
            return
        
        features_original = df[self.feature_cols].values.copy() # Keep original for output
        
        if self.scaler:
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        
        features_scaled = df[self.feature_cols].values
        
        # Targets might not be available or needed for pure inference, but handle if present
        targets = df[self.target_col].values if self.target_col in df.columns else np.full(len(df), -1) # Dummy if not present
        targets_original = targets.copy()

        times = df[self.time_col].values if self.time_col in df.columns else np.arange(len(df)) # Dummy times if not present
        
        num_samples = len(df)
        for i in range(num_samples - self.sequence_length + 1):
            feature_seq = features_scaled[i : i + self.sequence_length]
            
            # For inference, we mainly need the feature sequence.
            # We'll reconstruct predictions for the entire trial later.
            # Store enough info to map back predictions.
            # We store the *last* target and time of each sequence for simplicity,
            # as LSTM typically predicts for each step.
            self.sequences_data.append({
                "feature_seq": feature_seq,
                "original_start_index": i, # Start index of this sequence in the original trial
                "filepath": self.filepath
            })
        
        if not self.sequences_data:
            print(f"No sequences created for {self.filepath} (length {num_samples}, seq_len {self.sequence_length})")


    def __len__(self):
        return len(self.sequences_data)

    def __getitem__(self, idx):
        data_item = self.sequences_data[idx]
        feature_tensor = torch.tensor(data_item["feature_seq"], dtype=torch.float32)
        
        # Return data useful for piecing together predictions later
        return feature_tensor, data_item["original_start_index"], data_item["filepath"]

    def get_original_df(self, filepath):
        return self.all_trial_dfs.get(filepath)

def predict(model, data_loader, device):
    """
    Performs prediction on the data provided by the data_loader.
    Returns predictions aggregated by original file path.
    Each prediction is for the *last* element of the sequence.
    This needs to be adapted if per-timestep predictions are desired for the whole sequence.
    """
    model.eval()
    # Store predictions per file: {filepath: {start_index: [predicted_class_for_each_step_in_seq]}}
    # Or simpler: {filepath: {original_idx_in_file: predicted_class}}
    file_predictions = {} 

    with torch.no_grad():
        for features_batch, start_indices_batch, filepaths_batch in tqdm(data_loader, desc="Inferencing"):
            features_batch = features_batch.to(device)
            
            outputs_batch = model(features_batch) # Shape: (batch_size, seq_len, num_classes)
            _, predicted_classes_batch = torch.max(outputs_batch, 2) # Shape: (batch_size, seq_len)
            
            predicted_classes_batch = predicted_classes_batch.cpu().numpy() # (batch_size, seq_len)
            start_indices_batch = start_indices_batch.cpu().numpy() # (batch_size)

            for i in range(features_batch.size(0)): # Iterate through batch items
                filepath = filepaths_batch[i]
                seq_preds = predicted_classes_batch[i] # Predictions for one sequence (seq_len)
                seq_start_idx_original = start_indices_batch[i]

                if filepath not in file_predictions:
                    file_predictions[filepath] = {}
                
                # Store prediction for each step in the original file's timeline
                # Overlapping predictions will be overwritten by later sequences,
                # or one could average/vote. For simplicity, let's take the prediction
                # from the sequence that covers that timestep.
                # A common approach is to take the prediction for the last element of the sequence,
                # or for the center element. Here, we store all predictions from a sequence.
                for step_in_seq, pred_class in enumerate(seq_preds):
                    original_file_idx = seq_start_idx_original + step_in_seq
                    # If multiple sequences predict for the same original_file_idx,
                    # we can decide how to handle (e.g., last one wins, average logits if available)
                    # For now, last one wins.
                    file_predictions[filepath][original_file_idx] = pred_class
                    
    return file_predictions


def run_inference(model_path, inference_files_or_dir, output_csv_dir):
    """
    Main function to run inference on a list of files or all files in a directory.
    """
    print(f"--- Starting Inference ---")
    print(f"Loading model from: {model_path}")
    
    # --- Load Model and Scaler ---
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    if checkpoint is None:
        print(f"Failed to load checkpoint from {model_path}")
        return

    model = GaitLSTM(
        input_size=config.NUM_FEATURES,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM,
        lstm_dropout=0, # No dropout during inference typically
        linear_dropout=0
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    if scaler is None and config.NORMALIZATION_METHOD is not None:
        print("Warning: Model was trained with scaling, but scaler not found in checkpoint. Predictions may be inaccurate.")
        # Optionally, try to fit a new scaler if training data stats were available, or error out.

    # --- Prepare list of files for inference ---
    if os.path.isdir(inference_files_or_dir):
        # Assuming similar file structure for inference data as training data
        subject_trial_files = data_loader.get_subject_trial_files(
            inference_files_or_dir, 
            config.SUBJECT_DIRS_PATTERN, 
            config.TRIAL_FILE_PATTERN
        )
        files_to_process = []
        for subj_id in subject_trial_files:
            files_to_process.extend(subject_trial_files[subj_id])
    elif os.path.isfile(inference_files_or_dir):
        files_to_process = [inference_files_or_dir]
    else:
        print(f"Error: {inference_files_or_dir} is not a valid file or directory.")
        return

    if not files_to_process:
        print("No files found for inference.")
        return

    os.makedirs(output_csv_dir, exist_ok=True)

    # --- Process each file ---
    all_true_phases_flat = []
    all_pred_phases_flat = []
    all_timestamps_flat = []
    all_features_for_plot = []


    for filepath in files_to_process:
        print(f"\nProcessing file: {filepath}")
        inference_dataset = InferenceDataset(
            filepath=filepath,
            sequence_length=config.SEQUENCE_LENGTH,
            feature_cols=config.FEATURE_COLUMNS,
            target_col=config.TARGET_COLUMN, # To load true labels if available
            time_col=config.TIME_COLUMN,
            scaler=scaler
        )
        if len(inference_dataset) == 0:
            print(f"Skipping {filepath} due to no sequences created.")
            continue

        inference_loader = DataLoader(inference_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)
        
        # file_predictions: {filepath: {original_idx_in_file: predicted_class}}
        raw_predictions_map = predict(model, inference_loader, config.DEVICE)
        
        # --- Reconstruct full trial predictions and save ---
        original_df = inference_dataset.get_original_df(filepath)
        if original_df is None:
            print(f"Could not retrieve original DataFrame for {filepath}. Skipping output.")
            continue

        num_original_samples = len(original_df)
        trial_predicted_phases = np.full(num_original_samples, -1, dtype=int) # Default unpredicted

        if filepath in raw_predictions_map:
            preds_for_file = raw_predictions_map[filepath]
            for original_idx, pred_class in preds_for_file.items():
                if 0 <= original_idx < num_original_samples:
                    trial_predicted_phases[original_idx] = pred_class
        
        # Prepare output DataFrame
        output_df = original_df[[config.TIME_COLUMN] + config.FEATURE_COLUMNS].copy()
        output_df['predicted_phase'] = trial_predicted_phases
        
        if config.TARGET_COLUMN in original_df.columns:
            output_df['true_phase'] = original_df[config.TARGET_COLUMN]
            # For overall metrics and plotting later
            valid_preds_mask = (trial_predicted_phases != -1) # Only where we made a prediction
            all_true_phases_flat.extend(original_df[config.TARGET_COLUMN][valid_preds_mask].values)
            all_pred_phases_flat.extend(trial_predicted_phases[valid_preds_mask])
        
        all_timestamps_flat.extend(original_df[config.TIME_COLUMN].values)
        all_features_for_plot.extend(original_df[config.FEATURE_COLUMNS].values)


        # Save to CSV
        base_filename = os.path.basename(filepath)
        output_filename = os.path.splitext(base_filename)[0] + "_predictions.csv"
        output_filepath = os.path.join(output_csv_dir, output_filename)
        output_df.to_csv(output_filepath, index=False)
        print(f"Saved predictions for {base_filename} to {output_filepath}")

    # --- Overall Metrics and Plotting (if true labels were available) ---
    if all_true_phases_flat and all_pred_phases_flat:
        print("\n--- Overall Inference Metrics (on samples with predictions and true labels) ---")
        metrics = utils.calculate_metrics(all_true_phases_flat, all_pred_phases_flat)
        print(metrics)
        class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
        utils.plot_confusion_matrix_custom(all_true_phases_flat, all_pred_phases_flat, class_names, title="Inference Confusion Matrix")

        # Plot for the first processed file (or a concatenated version)
        # For simplicity, let's plot a segment of the concatenated data
        if all_timestamps_flat:
            utils.plot_predictions_vs_actual(
                np.array(all_timestamps_flat),
                np.array(all_features_for_plot), # This should be un-normalized features ideally
                np.array(all_true_phases_flat) if all_true_phases_flat else np.array(all_pred_phases_flat), # Show pred if true not avail
                np.array(all_pred_phases_flat),
                title="Sample Inference Predictions vs Actual"
            )
    else:
        print("\nNo true labels found in inference data for overall metrics calculation.")
        if all_timestamps_flat and all_pred_phases_flat: # Still plot predictions if available
             utils.plot_predictions_vs_actual(
                np.array(all_timestamps_flat),
                np.array(all_features_for_plot),
                np.array(all_pred_phases_flat), # No true phases to plot against
                np.array(all_pred_phases_flat),
                title="Sample Inference Predictions"
            )


if __name__ == '__main__':
    # Example usage:
    # Ensure you have a trained model saved, e.g., "best_model.pth" in config.OUTPUT_DIR
    
    # Option 1: Infer on a specific directory (e.g., a test subjects directory)
    # Create a dummy test subject folder for this example
    dummy_test_subj_dir = os.path.join(config.BASE_DATA_DIR, "S_test")
    os.makedirs(dummy_test_subj_dir, exist_ok=True)
    
    # Copy one of your existing CSVs to this dummy dir for testing inference, e.g., S40_A01_T01.csv
    # For this example, let's assume config.BASE_DATA_DIR contains at least one file matching the pattern.
    # Find an example file to copy:
    example_file_to_copy = None
    subj_files_map = data_loader.get_subject_trial_files(config.BASE_DATA_DIR, config.SUBJECT_DIRS_PATTERN, config.TRIAL_FILE_PATTERN)
    if subj_files_map:
        first_subj = list(subj_files_map.keys())[0]
        if subj_files_map[first_subj]:
            example_file_to_copy = subj_files_map[first_subj][0]
            
    if example_file_to_copy:
        import shutil
        dummy_file_name = os.path.basename(example_file_to_copy).replace(first_subj, "S_test")
        shutil.copy(example_file_to_copy, os.path.join(dummy_test_subj_dir, dummy_file_name))
        print(f"Copied {example_file_to_copy} to {os.path.join(dummy_test_subj_dir, dummy_file_name)} for inference test.")
        
        inference_target_dir = config.BASE_DATA_DIR # Or point to a specific test subjects parent dir
        output_prediction_dir = os.path.join(config.OUTPUT_DIR, "inference_results")
        
        # Make sure a model exists. If not, this will fail.
        # You might need to run train.py first to generate a model.
        # For testing, create a dummy model checkpoint if one doesn't exist.
        if not os.path.exists(config.DEFAULT_MODEL_PATH):
            print(f"Warning: Default model path {config.DEFAULT_MODEL_PATH} not found. Inference will likely fail.")
            print("Please train a model first or provide a valid model path.")
        else:
             run_inference(
                model_path=config.DEFAULT_MODEL_PATH, 
                inference_files_or_dir=inference_target_dir, # Infer on the whole base dir or a specific test dir
                output_csv_dir=output_prediction_dir
            )
    else:
        print("No example CSV file found in data directory to set up dummy inference test.")


    # Option 2: Infer on a single file
    # if example_file_to_copy:
    #     single_file_to_infer = os.path.join(dummy_test_subj_dir, dummy_file_name)
    #     if os.path.exists(single_file_to_infer) and os.path.exists(config.DEFAULT_MODEL_PATH):
    #         run_inference(
    #             model_path=config.DEFAULT_MODEL_PATH,
    #             inference_files_or_dir=single_file_to_infer,
    #             output_csv_dir=output_prediction_dir
    #         )