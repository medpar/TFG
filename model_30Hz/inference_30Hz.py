# model_30Hz/inference_30Hz.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob

# Import from the new 30Hz files
import config_30Hz as config
from model_30Hz import GaitLSTM30Hz
import utils_30Hz as utils
# Import the downsampling function from the 30Hz data loader
from data_loader_30Hz import downsample_dataframe

# --- FIX FOR IMPORTING SIBLING DIRECTORY ---
import sys
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
# --- END FIX ---

try:
    from gaitseg import gaitseg_utils as gal
except ImportError:
    print("Warning: Could not import 'gaitseg_utils'.")
    gal = None


class InferenceDataset30Hz(Dataset):
    """Dataset for 30Hz inference data from video models."""
    def __init__(self, filepath, sequence_length, feature_cols, scaler=None):
        self.filepath = filepath
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.sequences_data = []
        self.all_trial_dfs = {}
        self._load_and_create_sequences()

    def _load_and_create_sequences(self):
        try:
            df = pd.read_csv(self.filepath)
            if df.empty:
                print(f"Warning: File {self.filepath} is empty.")
                return
        except Exception as e:
            print(f"Error reading {self.filepath}: {e}")
            return

        self.all_trial_dfs[self.filepath] = df.copy()
        df_to_process = df.copy()

        if not all(col in df_to_process.columns for col in self.feature_cols):
            print(f"Warning: Missing required feature columns in {self.filepath}. Required: {self.feature_cols}, Available: {list(df_to_process.columns)}. Skipping.")
            return

        if self.scaler:
            df_to_process.loc[:, self.feature_cols] = self.scaler.transform(df_to_process[self.feature_cols])
        
        features_scaled = df_to_process[self.feature_cols].values
        num_samples = len(df_to_process)
        if num_samples < self.sequence_length:
            print(f"Warning: Not enough samples in {self.filepath} for one sequence. Skipping.")
            return
            
        for i in range(num_samples - self.sequence_length + 1):
            feature_seq = features_scaled[i : i + self.sequence_length]
            self.sequences_data.append({
                "feature_seq": feature_seq,
                "original_start_index": i,
                "filepath": self.filepath
            })

    def __len__(self): return len(self.sequences_data)
    def __getitem__(self, idx):
        data = self.sequences_data[idx]
        return torch.tensor(data["feature_seq"], dtype=torch.float32), data["original_start_index"], data["filepath"]
    def get_original_df(self, filepath): return self.all_trial_dfs.get(filepath)

def predict(model, data_loader, device):
    model.eval()
    file_predictions = {}
    with torch.no_grad():
        for features_batch, start_indices_batch, filepaths_batch in tqdm(data_loader, desc="Inferencing"):
            features_batch = features_batch.to(device)
            outputs_batch = model(features_batch)
            _, predicted_classes_batch = torch.max(outputs_batch, 2)
            predicted_classes_batch, start_indices_batch = predicted_classes_batch.cpu().numpy(), start_indices_batch.cpu().numpy()
            for i in range(features_batch.size(0)):
                filepath, seq_preds, seq_start_idx = filepaths_batch[i], predicted_classes_batch[i], start_indices_batch[i]
                if filepath not in file_predictions: file_predictions[filepath] = {}
                for step_in_seq, pred_class in enumerate(seq_preds):
                    file_predictions[filepath][seq_start_idx + step_in_seq] = pred_class
    return file_predictions

def find_corresponding_raw_file(video_angle_csv_path):
    if gal is None: return None
    base_name = os.path.basename(video_angle_csv_path)
    filename_stem = os.path.splitext(base_name)[0]
    parts = filename_stem.split('_')
    if len(parts) < 3:
        print(f"Could not parse subject/trial from CSV stem: {filename_stem}")
        return None
    subject_id = parts[0]
    
    try:
        if hasattr(gal, 'root_dir') and os.path.isdir(gal.root_dir):
            raw_search_dir_base = gal.root_dir
        else:
            # Fallback if gal.root_dir is not set
            vidimu_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(video_angle_csv_path))))
            raw_search_dir_base = os.path.join(vidimu_root_guess, "benchmark", "jointangles", "jointangles_imus")
        
        raw_file_path = os.path.join(raw_search_dir_base, subject_id, f"{filename_stem}.raw")
        if os.path.exists(raw_file_path):
            print(f"  Found corresponding .raw file: {raw_file_path}")
            return raw_file_path
        else:
            print(f"  Could not find corresponding .raw file at expected path: {raw_file_path}")
            return None
    except Exception as e:
        print(f"Error finding corresponding raw file: {e}")
        return None

def find_corresponding_ground_truth_file(video_angle_csv_path):
    """Finds the ground truth file in gaitseg_corrected for a given inference file."""
    # Example input: /path/to/S41_A01_T02.csv
    base_name = os.path.basename(video_angle_csv_path)

    # --- FIX STARTS HERE ---
    # Get the filename without extension, e.g., 'S41_A01_T02'
    core_name = os.path.splitext(base_name)[0]
    
    subject_id = core_name.split('_')[0]
    
    # Construct the correct ground truth filename, e.g., 'S41_A01_T02_corrected.csv'
    gt_filename = f"{core_name}_corrected.csv"
    # --- FIX ENDS HERE ---
    
    # Use the TRAIN_DATA_DIR from config, which points to gaitseg_corrected
    gt_filepath = os.path.join(config.TRAIN_DATA_DIR, subject_id, gt_filename)
    
    if os.path.exists(gt_filepath):
        return gt_filepath
    else:
        print(f"  Warning: Ground truth file not found at expected path: {gt_filepath}")
        return None

def run_inference_30Hz(model_path, inference_input_path, output_dir):
    print(f"--- Starting 30Hz Model Inference ---")
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    if checkpoint is None: print(f"Failed to load checkpoint from {model_path}"); return

    ch_hyperparams = checkpoint.get('hyperparameters') or {}
    lstm_hidden_size = ch_hyperparams.get('lstm_hidden_size', config.LSTM_HIDDEN_SIZE)
    num_lstm_layers = ch_hyperparams.get('num_lstm_layers', config.NUM_LSTM_LAYERS)
    
    model = GaitLSTM30Hz(
        input_size=config.NUM_FEATURES, hidden_size=lstm_hidden_size,
        num_layers=num_lstm_layers, num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=0, linear_dropout=0
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    if scaler is None and config.NORMALIZATION_METHOD is not None:
        print("Warning: Scaler not found in checkpoint.")

    files_to_process = glob.glob(os.path.join(inference_input_path, config.SUBJECT_DIRS_PATTERN, config.INFERENCE_TRIAL_PATTERN), recursive=True)
    if not files_to_process:
        print(f"No files found in {inference_input_path} matching pattern '{config.INFERENCE_TRIAL_PATTERN}'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    utils.OUTPUT_PLOTS_DIR = os.path.join(output_dir, "output_plots")
    os.makedirs(utils.OUTPUT_PLOTS_DIR, exist_ok=True)

    aggregated_true_phases = []
    aggregated_pred_phases = []

    for csv_filepath in files_to_process:
        print(f"\nProcessing file: {os.path.basename(csv_filepath)}")
        inference_dataset = InferenceDataset30Hz(
            filepath=csv_filepath, sequence_length=config.SEQUENCE_LENGTH,
            feature_cols=config.FEATURE_COLUMNS, scaler=scaler
        )
        if len(inference_dataset) == 0: continue

        inference_loader = DataLoader(inference_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)
        raw_predictions_map = predict(model, inference_loader, config.DEVICE)
        
        original_df = inference_dataset.get_original_df(csv_filepath)
        if original_df is None: continue

        num_original_samples = len(original_df)
        predicted_phases = np.full(num_original_samples, -1, dtype=int)
        if csv_filepath in raw_predictions_map:
            for idx, pred_class in raw_predictions_map[csv_filepath].items():
                if 0 <= idx < num_original_samples: predicted_phases[idx] = pred_class
        
        ground_truth_phases_aligned = None
        gt_filepath = find_corresponding_ground_truth_file(csv_filepath)
        if gt_filepath:
            print(f"  Found ground truth file: {os.path.basename(gt_filepath)}")
            df_gt_50hz = pd.read_csv(gt_filepath)
            df_gt_30hz = downsample_dataframe(df_gt_50hz, original_hz=50.0, target_hz=config.SAMPLING_RATE)
            
            min_len = min(len(original_df), len(df_gt_30hz))
            ground_truth_phases_aligned = df_gt_30hz[config.TARGET_COLUMN].values[:min_len]
            predicted_phases = predicted_phases[:min_len]
            original_df = original_df.iloc[:min_len]
            
            valid_preds_mask = (predicted_phases != -1)
            if np.any(valid_preds_mask):
                aggregated_true_phases.extend(ground_truth_phases_aligned[valid_preds_mask])
                aggregated_pred_phases.extend(predicted_phases[valid_preds_mask])
        else:
            print("  Ground truth file not found, cannot plot true phases or individual CM.")

        output_df = original_df.copy()
        output_df['predicted_phase'] = predicted_phases
        if ground_truth_phases_aligned is not None:
             output_df['true_phase_aligned'] = ground_truth_phases_aligned

        base_filename = os.path.basename(csv_filepath)
        output_filename = os.path.splitext(base_filename)[0] + "_gait_phases.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        output_df.to_csv(output_filepath, index=False)
        print(f"  -> Saved predictions to: {output_filepath}")

        plot_title_suffix = f" ({base_filename})"
        utils.plot_model_predictions_vs_true_phases(
            timestamps=original_df[config.TIME_COLUMN].values,
            true_phases=ground_truth_phases_aligned,
            predicted_phases=predicted_phases,
            title_suffix=plot_title_suffix
        )
        
        raw_file_for_plot = find_corresponding_raw_file(csv_filepath)
        if raw_file_for_plot:
            utils.plot_model_predictions_vs_angular_velocity(
                timestamps_csv=original_df[config.TIME_COLUMN].values,
                predicted_phases_csv=predicted_phases,
                raw_imu_filepath=raw_file_for_plot,
                title_suffix=plot_title_suffix
            )

    if aggregated_true_phases and aggregated_pred_phases:
        print("\n--- Overall Inference Metrics ---")
        metrics = utils.calculate_metrics(aggregated_true_phases, aggregated_pred_phases)
        print(metrics)
        class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
        utils.plot_confusion_matrix_custom(aggregated_true_phases, aggregated_pred_phases, 
                                           class_names, title="Overall Inference Confusion Matrix")
    else:
        print("\nNo ground truth phases were found to calculate overall metrics.")

    print("\n--- 30Hz Inference Finished ---")


if __name__ == '__main__':
    inference_source_dir = config.INFERENCE_DATA_DIR 
    output_dir = config.INFERENCE_OUTPUT_DIR
    model_path = config.DEFAULT_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"FATAL: Trained model not found at {model_path}. Please run train_30Hz.py first.")
    elif not os.path.isdir(inference_source_dir):
        print(f"FATAL: Inference data directory not found at {inference_source_dir}.")
    else:
         run_inference_30Hz(
            model_path=model_path, 
            inference_input_path=inference_source_dir,
            output_dir=output_dir
        )