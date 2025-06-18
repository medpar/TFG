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

# Attempt to make gaitseg_utils accessible
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
GAITSEG_DIR = os.path.join(GRANDPARENT_DIR, "gaitseg")
if GAITSEG_DIR not in sys.path: sys.path.append(GAITSEG_DIR)
try:
    import gaitseg_utils as gal
except ImportError:
    print("Warning: gaitseg_utils not found. Plotting vs angular velocity will be limited.")
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

        # Check for required feature columns
        if not all(col in df_to_process.columns for col in self.feature_cols):
            print(f"Warning: Missing required feature columns in {self.filepath}. Skipping.")
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
    # ... (No changes needed)
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
    # ... (finds the 50Hz .raw file corresponding to a 30Hz video angle file)
    if gal is None: return None
    base_name = os.path.basename(video_angle_csv_path)
    # Assumes filename like 'S40_A01_T01_all_angles.csv' or similar
    parts = base_name.split('_')
    if len(parts) < 3: return None
    subject_id, activity_id, trial_id = parts[0], parts[1], parts[2]
    raw_filename_stem = f"{subject_id}_{activity_id}_{trial_id}"
    
    # Heuristic to find the original IMU-labeled data path
    # This assumes the 'gaitseg_corrected' folder is a sibling to where the video angles are
    try:
        vidimu_root_guess = os.path.dirname(os.path.dirname(config.INFERENCE_DATA_DIR))
        raw_data_dir = os.path.join(vidimu_root_guess, "gaitseg_corrected", subject_id) # Path to where original IMU data is
        raw_file_path = os.path.join(raw_data_dir, f"{raw_filename_stem}_corrected.csv") # Check for the corrected CSV first
        
        # We need the .raw file, which is likely in a different structure
        # Let's rebuild the path to the benchmark IMU folder
        imu_benchmark_dir = os.path.join(vidimu_root_guess, "benchmark", "jointangles", "jointangles_imus")
        final_raw_file_path = os.path.join(imu_benchmark_dir, subject_id, f"ik_{raw_filename_stem}.mot") # Actually .mot for raw angles
        raw_imu_path = os.path.join(vidimu_root_guess, "benchmark", "IMU_data", subject_id, f"{raw_filename_stem}.raw") # Path to true raw IMU
        
        if os.path.exists(raw_imu_path):
            return raw_imu_path
        else:
            print(f"Could not find corresponding .raw file at expected path: {raw_imu_path}")
            return None
    except Exception as e:
        print(f"Error finding corresponding raw file: {e}")
        return None


def run_inference_30Hz(model_path, inference_input_path, output_dir):
    print(f"--- Starting 30Hz Model Inference ---")
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    if checkpoint is None: print(f"Failed to load checkpoint from {model_path}"); return

    model = GaitLSTM30Hz(
        input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=0, linear_dropout=0
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    scaler = checkpoint.get('scaler')
    if scaler is None and config.NORMALIZATION_METHOD is not None:
        print("Warning: Scaler not found in checkpoint.")

    files_to_process = glob.glob(os.path.join(inference_input_path, config.SUBJECT_DIRS_PATTERN, config.INFERENCE_TRIAL_PATTERN), recursive=True)
    if not files_to_process:
        print(f"No files found in {inference_input_path} matching pattern.")
        return

    os.makedirs(output_dir, exist_ok=True)
    utils.OUTPUT_PLOTS_DIR = os.path.join(output_dir, "output_plots")
    os.makedirs(utils.OUTPUT_PLOTS_DIR, exist_ok=True)

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
        
        output_df = original_df.copy()
        output_df['predicted_phase'] = predicted_phases
        
        base_filename = os.path.basename(csv_filepath)
        output_filename = os.path.splitext(base_filename)[0] + "_gait_phases.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        output_df.to_csv(output_filepath, index=False)
        print(f"  -> Saved predictions to: {output_filepath}")

        # Plotting predicted phases vs joint angles
        utils.plot_model_predictions_vs_true_phases(
            timestamps=original_df[config.TIME_COLUMN].values,
            true_phases=None, # No true phases in this data
            predicted_phases=predicted_phases,
            title_suffix=f" ({base_filename})"
        )
        
        # Plotting vs angular velocity
        # Note: This requires finding the original 50Hz raw IMU file to get ang vel.
        # find_corresponding_raw_file logic may need to be robust.
        raw_file_for_plot = find_corresponding_raw_file(csv_filepath)
        if raw_file_for_plot:
            utils.plot_model_predictions_vs_angular_velocity(
                timestamps_csv=original_df[config.TIME_COLUMN].values,
                predicted_phases_csv=predicted_phases,
                raw_imu_filepath=raw_file_for_plot,
                title_suffix=f" ({base_filename})"
            )

    print("\n--- 30Hz Inference Finished ---")


if __name__ == '__main__':
    # Use the specific paths from the 30Hz config
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