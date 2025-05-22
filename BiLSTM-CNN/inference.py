# model/inference.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob 

import config
from model import BiLSTMCnn1D # Import the new model
import data_loader 
import utils

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GAITSEG_DIR = os.path.join(PARENT_DIR, "gaitseg") 

if GAITSEG_DIR not in sys.path:
    sys.path.append(GAITSEG_DIR)
try:
    import gaitseg_utils as gal
except ImportError:
    print("Warning: Could not import gaitseg_utils. Inference plotting against angular velocity will be limited.")
    gal = None

# InferenceDataset, predict, find_corresponding_raw_file remain the same as your last provided version.
# I'll include them for completeness if they were not modified.

class InferenceDataset(Dataset):
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

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing feature columns {missing_cols} in {self.filepath}. Available: {df.columns.tolist()}. Skipping file.")
            return
        
        features_to_scale_df = df[self.feature_cols]
        
        if self.scaler:
            scaled_values = self.scaler.transform(features_to_scale_df.values)
            features_processed = pd.DataFrame(scaled_values, columns=self.feature_cols, index=df.index)
        else:
            features_processed = features_to_scale_df
        
        features_for_sequencing = features_processed.values
        
        num_samples = len(df)
        if num_samples < self.sequence_length:
            print(f"Warning: File {self.filepath} has {num_samples} samples, less than sequence length {self.sequence_length}. Skipping.")
            return
            
        for i in range(num_samples - self.sequence_length + 1):
            feature_seq = features_for_sequencing[i : i + self.sequence_length]
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
    if gal is None: return None
    base_csv_name = os.path.basename(csv_filepath)
    # Adapt to various suffixes: _corrected.csv, _BiLSTMCnn_predictions.csv, etc.
    name_part = base_csv_name
    suffixes_to_remove = ["_corrected.csv", "_BiLSTMCnn_predictions.csv", "_model_predictions.csv", "_predictions.csv"]
    for suffix in suffixes_to_remove:
        if name_part.endswith(suffix):
            name_part = name_part[:-len(suffix)]
            break # Remove only the first matching suffix from the end
    
    parts = name_part.split('_')
    if len(parts) < 3: # SXX_A01_TYY
        print(f"Could not reliably parse subject/trial from: {name_part} (derived from {base_csv_name})")
        return None

    subject_id = parts[0]
    raw_filename_stem = name_part # SXX_A01_TYY
    
    raw_search_dir_base = None
    if hasattr(gal, 'GUI_BASE_DATA_DIR') and gal.GUI_BASE_DATA_DIR:
         raw_search_dir_base = gal.GUI_BASE_DATA_DIR
    elif hasattr(gal, 'root_dir') and gal.root_dir:
         raw_search_dir_base = gal.root_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root_guess = os.path.dirname(script_dir) 
        raw_search_dir_base = os.path.join(project_root_guess, "benchmark", "jointangles", "jointangles_imus")
        if not os.path.isdir(raw_search_dir_base):
             print(f"Warning: Fallback raw file search directory does not exist: {raw_search_dir_base}")
             return None
        print(f"Guessed raw file base directory: {raw_search_dir_base}")

    raw_file_path = os.path.join(raw_search_dir_base, subject_id, f"{raw_filename_stem}.raw")
    if os.path.exists(raw_file_path):
        return raw_file_path
    else:
        print(f"Corresponding .raw file not found at expected path: {raw_file_path}")
        return None

def run_inference(model_path, inference_input_path, output_csv_dir):
    print(f"--- Starting Inference for BiLSTMCnn1D model ---")
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False) # Set weights_only based on trust
    if checkpoint is None:
        print(f"Failed to load checkpoint from {model_path}")
        return

    # Instantiate the BiLSTMCnn1D model
    # It's good practice to save model architecture parameters in checkpoint or deduce them.
    # For now, we rely on the current config.py for architecture details.
    # Dropout rates are set to 0 for inference.
    model = BiLSTMCnn1D(
        input_size=config.NUM_FEATURES,
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
        cnn_strides=config.CNN_STRIDES,
        cnn_padding=config.CNN_PADDING,
        cnn_activation=config.CNN_ACTIVATION,
        cnn_dropout_rate=0.0, # No dropout during inference
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        num_lstm_layers=config.NUM_LSTM_LAYERS,
        lstm_dropout_rate=0.0, # No dropout during inference
        bidirectional_lstm=config.BIDIRECTIONAL_LSTM,
        linear_dropout_rate=0.0 # No dropout during inference
    ).to(config.DEVICE)
        
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    if scaler is None and config.NORMALIZATION_METHOD is not None:
        print("Warning: Scaler not found in checkpoint. Predictions may be inaccurate.")

    # File processing logic (same as before)
    if os.path.isdir(inference_input_path):
        files_to_process = []
        for subj_folder in glob.glob(os.path.join(inference_input_path, config.SUBJECT_DIRS_PATTERN)):
            if os.path.isdir(subj_folder):
                files_to_process.extend(glob.glob(os.path.join(subj_folder, config.TRIAL_FILE_PATTERN)))
    elif os.path.isfile(inference_input_path) and inference_input_path.endswith(".csv"):
        files_to_process = [inference_input_path]
    else:
        print(f"Error: {inference_input_path} is not a valid CSV file or directory of CSVs based on config patterns.")
        return

    if not files_to_process:
        print(f"No suitable CSV files found for inference in '{inference_input_path}' matching pattern '{config.TRIAL_FILE_PATTERN}'.")
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
        
        # Ensure all expected feature columns are present in original_df before selecting
        cols_to_select = [config.TIME_COLUMN]
        for fc in config.FEATURE_COLUMNS:
            if fc in original_df.columns:
                cols_to_select.append(fc)
            else:
                print(f"Warning: Feature column '{fc}' not in original_df for {csv_filepath}. It won't be in the output.")
        
        output_df = original_df[cols_to_select].copy()
        output_df['predicted_phase_model'] = trial_predicted_phases
        
        true_phases_available = False
        if config.TARGET_COLUMN in original_df.columns:
            output_df['true_phase_gui'] = original_df[config.TARGET_COLUMN]
            true_phases_available = True
            valid_preds_mask = (trial_predicted_phases != -1)
            if np.any(valid_preds_mask):
                aggregated_true_phases.extend(original_df[config.TARGET_COLUMN][valid_preds_mask].values)
                aggregated_pred_phases.extend(trial_predicted_phases[valid_preds_mask])
        
        base_filename = os.path.basename(csv_filepath)
        name_part_for_output = base_filename.replace("_corrected.csv", "")
        output_filename = f"{name_part_for_output}_BiLSTMCnn_predictions.csv"
        output_filepath = os.path.join(output_csv_dir, output_filename)
        output_df.to_csv(output_filepath, index=False)
        print(f"Saved model predictions for {base_filename} to {output_filepath}")

        plot_title_suffix = f" ({name_part_for_output})"
        plot_trial_num = None # Not applicable for direct inference run usually

        if true_phases_available:
            utils.plot_model_predictions_vs_true_phases(
                timestamps=original_df[config.TIME_COLUMN].values,
                true_phases=original_df[config.TARGET_COLUMN].values,
                predicted_phases=trial_predicted_phases,
                title_suffix=plot_title_suffix, trial_num=plot_trial_num
            )
        else:
             utils.plot_model_predictions_vs_true_phases(
                timestamps=original_df[config.TIME_COLUMN].values,
                true_phases=None, 
                predicted_phases=trial_predicted_phases,
                title_suffix=plot_title_suffix + " (No True Phases)", trial_num=plot_trial_num
            )

        raw_imu_file_for_plot = find_corresponding_raw_file(csv_filepath)
        if raw_imu_file_for_plot and gal:
            utils.plot_model_predictions_vs_angular_velocity(
                timestamps_csv=original_df[config.TIME_COLUMN].values,
                predicted_phases_csv=trial_predicted_phases,
                raw_imu_filepath=raw_imu_file_for_plot,
                title_suffix=plot_title_suffix, trial_num=plot_trial_num
            )
        else:
            print(f"Skipping angular velocity plot for {base_filename}: .raw file not found or gaitseg_utils not available.")

    if aggregated_true_phases and aggregated_pred_phases:
        print("\n--- Overall Inference Metrics (BiLSTMCnn1D) ---")
        metrics = utils.calculate_metrics(aggregated_true_phases, aggregated_pred_phases)
        print(metrics)
        class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
        true_eval = [p for p in aggregated_true_phases if p in range(config.NUM_CLASSES)]
        pred_eval = [p for idx, p in enumerate(aggregated_pred_phases) if aggregated_true_phases[idx] in range(config.NUM_CLASSES)]
        if true_eval and pred_eval:
             utils.plot_confusion_matrix_custom(true_eval, pred_eval, class_names, title="Overall Inference CM (BiLSTMCnn1D)")
        else:
            print("Not enough valid class data for overall confusion matrix after filtering.")
    else:
        print("\nNo true labels found across inference files for overall metrics calculation.")
    print("\n--- Inference Finished ---")

if __name__ == '__main__':
    inference_source_dir = config.BASE_DATA_DIR
    output_prediction_dir = os.path.join(config.OUTPUT_DIR, "inference_BiLSTMCnn1D_results")
    
    model_filename_to_load = config.DEFAULT_MODEL_PATH 
    if not os.path.exists(model_filename_to_load):
         # Try to find a fold0 model if DEFAULT_MODEL_PATH doesn't exist (common in CV setup)
        fold0_model_path = os.path.join(os.path.dirname(config.DEFAULT_MODEL_PATH), "best_model_fold0.pth")
        if os.path.exists(fold0_model_path):
            model_filename_to_load = fold0_model_path
            print(f"DEFAULT_MODEL_PATH not found, using best model from fold 0: {model_filename_to_load}")
        else:
            print(f"FATAL: Default model path {config.DEFAULT_MODEL_PATH} and fold0 model not found. Inference cannot proceed.")
            sys.exit(1) # Exit if no model found

    if not os.path.isdir(inference_source_dir) or not glob.glob(os.path.join(inference_source_dir, config.SUBJECT_DIRS_PATTERN, config.TRIAL_FILE_PATTERN)):
        print(f"Warning: Inference source directory '{inference_source_dir}' is empty or does not contain files matching pattern '{config.SUBJECT_DIRS_PATTERN}/{config.TRIAL_FILE_PATTERN}'.")
    else:
         run_inference(
            model_path=model_filename_to_load, 
            inference_input_path=inference_source_dir,
            output_csv_dir=output_prediction_dir
        )