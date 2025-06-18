# model_30Hz/data_loader_30Hz.py
import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
import config_30Hz as config # Use the 30Hz config

def get_subject_trial_files(base_dir, subject_pattern, trial_pattern):
    subject_trial_files = {}
    subject_dirs = sorted(glob.glob(os.path.join(base_dir, subject_pattern)))
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        trial_files = sorted(glob.glob(os.path.join(subj_dir, trial_pattern)))
        if trial_files:
            subject_trial_files[subject_id] = trial_files
    print(f"Found {len(subject_trial_files)} subjects with trial data in '{base_dir}'.")
    return subject_trial_files

def create_sequences(data_df, sequence_length, feature_cols, target_col, time_col=None):
    sequences = []
    features = data_df[feature_cols].values
    targets = data_df[target_col].values
    times = data_df[time_col].values if time_col and time_col in data_df.columns else None
    num_samples = len(data_df)
    for i in range(num_samples - sequence_length + 1):
        feature_seq = features[i : i + sequence_length]
        target_seq = targets[i : i + sequence_length]
        if times is not None:
            time_seq = times[i : i + sequence_length]
            sequences.append((feature_seq, target_seq, time_seq))
        else:
            sequences.append((feature_seq, target_seq))
    return sequences

def downsample_dataframe(df, original_hz, target_hz):
    """
    Downsamples a DataFrame containing time-series data.
    - Features and time are resampled using linear interpolation.
    - Phase (target) is resampled using nearest-neighbor to preserve labels.
    """
    if df.empty:
        return df

    original_len = len(df)
    duration = original_len / original_hz
    target_len = int(duration * target_hz)

    if target_len == 0:
        return pd.DataFrame(columns=df.columns)

    # 1. Resample continuous data (features and time)
    downsampled_continuous = pd.DataFrame()
    
    # Create new time vector at the target frequency
    new_time = np.linspace(df[config.TIME_COLUMN].iloc[0], df[config.TIME_COLUMN].iloc[-1], num=target_len)
    downsampled_continuous[config.TIME_COLUMN] = new_time
    
    for col in config.FEATURE_COLUMNS:
        downsampled_continuous[col] = resample(df[col], target_len)

    # 2. Resample categorical data (phase) using nearest-neighbor interpolation
    phase_interpolator = interp1d(
        df[config.TIME_COLUMN], 
        df[config.TARGET_COLUMN], 
        kind='nearest', 
        bounds_error=False, 
        fill_value=(df[config.TARGET_COLUMN].iloc[0], df[config.TARGET_COLUMN].iloc[-1])
    )
    downsampled_continuous[config.TARGET_COLUMN] = phase_interpolator(new_time).astype(int)

    return downsampled_continuous


class GaitPhaseDataset30Hz(Dataset):
    def __init__(self, trial_filepaths, sequence_length, feature_cols, target_col, 
                 time_col=None, scaler=None, fit_scaler=False, calculate_class_weights=False):
        self.trial_filepaths = trial_filepaths
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.time_col = time_col
        self.scaler = scaler
        self.calculate_class_weights = calculate_class_weights

        self.sequences = []
        self.all_features_for_scaling = []
        self.class_weights = None

        self._load_and_process_data(fit_scaler)

    def _load_and_process_data(self, fit_scaler):
        print(f"Loading and processing data for {len(self.trial_filepaths)} trials for 30Hz model...")
        
        all_dfs_processed = [] 
        all_target_labels_for_weight_calc = [] 

        for file_path in self.trial_filepaths:
            try:
                df_50hz = pd.read_csv(file_path)
                if not df_50hz.empty and all(col in df_50hz.columns for col in self.feature_cols + [self.target_col]):
                    # --- KEY CHANGE: DOWNSAMPLE DATA FROM 50Hz to 30Hz ---
                    print(f"  Downsampling {os.path.basename(file_path)} from 50Hz to {config.SAMPLING_RATE}Hz...")
                    df_30hz = downsample_dataframe(df_50hz, original_hz=50.0, target_hz=config.SAMPLING_RATE)
                    
                    if df_30hz.empty:
                        print(f"  Skipping file after downsampling, resulted in 0 samples.")
                        continue

                    if self.calculate_class_weights:
                        all_target_labels_for_weight_calc.extend(df_30hz[self.target_col].values)

                    all_dfs_processed.append(df_30hz)
                    if fit_scaler:
                        self.all_features_for_scaling.append(df_30hz[self.feature_cols].values)
            except Exception as e:
                print(f"Warning: Could not read or process {file_path}: {e}")
        
        if not all_dfs_processed:
            print("Warning: No valid data files were processed.")
            return

        if fit_scaler and self.all_features_for_scaling:
            combined_features = np.concatenate(self.all_features_for_scaling, axis=0)
            if config.NORMALIZATION_METHOD == "standard": self.scaler = StandardScaler()
            elif config.NORMALIZATION_METHOD == "minmax": self.scaler = MinMaxScaler()
            if self.scaler:
                print(f"Fitting scaler on downsampled 30Hz data ({combined_features.shape[0]} samples).")
                self.scaler.fit(combined_features)
        
        for df in all_dfs_processed:
            if self.scaler:
                df.loc[:, self.feature_cols] = self.scaler.transform(df[self.feature_cols].values)
            
            trial_sequences = create_sequences(df, self.sequence_length, self.feature_cols, self.target_col, self.time_col)
            self.sequences.extend(trial_sequences)
            
        print(f"Created {len(self.sequences)} sequences in total (at 30Hz).")

        if self.calculate_class_weights and all_target_labels_for_weight_calc:
            # ... (class weight calculation logic remains the same) ...
            all_labels_np = np.array(all_target_labels_for_weight_calc)
            valid_unique_classes = [cls for cls in np.unique(all_labels_np) if 0 <= cls < config.NUM_CLASSES]
            if not valid_unique_classes:
                self.class_weights = np.ones(config.NUM_CLASSES)
            else:
                weights = compute_class_weight('balanced', classes=np.array(valid_unique_classes), y=all_labels_np[np.isin(all_labels_np, valid_unique_classes)])
                self.class_weights = np.ones(config.NUM_CLASSES, dtype=float) 
                class_to_weight_map = dict(zip(valid_unique_classes, weights))
                for i in range(config.NUM_CLASSES):
                    self.class_weights[i] = class_to_weight_map.get(i, 1.0) # Default to 1.0 if class is missing
            print(f"Calculated class weights for 30Hz data: {self.class_weights}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feature_seq, target_seq, *time_seq_tuple = self.sequences[idx]
        feature_tensor = torch.tensor(feature_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        if time_seq_tuple:
            return feature_tensor, target_tensor, time_seq_tuple[0]
        else:
            return feature_tensor, target_tensor

    def get_scaler(self):
        return self.scaler

    def get_class_weights(self):
        return self.class_weights


def get_data_loaders_30Hz(fold_num=None, subject_trial_files=None):
    if subject_trial_files is None:
        # Load the 50Hz data for training
        subject_trial_files = get_subject_trial_files(
            config.TRAIN_DATA_DIR, 
            config.SUBJECT_DIRS_PATTERN, 
            config.TRAIN_TRIAL_PATTERN
        )
    subject_ids = sorted(list(subject_trial_files.keys()))
    if not subject_ids:
        raise ValueError("No subjects found in training data directory.")

    train_files, val_files, test_files = [], [], []
    scaler = None
    class_weights = None

    if config.K_FOLDS > 1:
        gkf = GroupKFold(n_splits=config.K_FOLDS)
        unique_subj_ids_arr = np.array(subject_ids)
        train_idx, val_idx = list(gkf.split(unique_subj_ids_arr, groups=unique_subj_ids_arr))[fold_num]
        train_subj_ids = [unique_subj_ids_arr[i] for i in train_idx]
        val_subj_ids = [unique_subj_ids_arr[i] for i in val_idx]
        print(f"Fold {fold_num}: {len(train_subj_ids)} train subjects, {len(val_subj_ids)} val subjects.")
        for subj_id in train_subj_ids: train_files.extend(subject_trial_files[subj_id])
        for subj_id in val_subj_ids: val_files.extend(subject_trial_files[subj_id])
        
        # Use the new 30Hz Dataset
        train_dataset = GaitPhaseDataset30Hz(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, 
                                            fit_scaler=True, calculate_class_weights=config.USE_WEIGHTED_LOSS)
        scaler = train_dataset.get_scaler()
        if config.USE_WEIGHTED_LOSS: class_weights = train_dataset.get_class_weights()
        val_dataset = GaitPhaseDataset30Hz(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler)
    else: # Single train/val/test split
        if config.TEST_SPLIT_RATIO > 0:
            train_val_subj_ids, test_subj_ids_list = train_test_split(subject_ids, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_SEED)
            for subj_id in test_subj_ids_list: test_files.extend(subject_trial_files[subj_id])
        else: train_val_subj_ids = subject_ids
        train_subj_ids_list, val_subj_ids_list = train_test_split(train_val_subj_ids, test_size=config.VALIDATION_SPLIT_RATIO / (1 - config.TEST_SPLIT_RATIO if config.TEST_SPLIT_RATIO < 1 else 1), random_state=config.RANDOM_SEED)
        for subj_id in train_subj_ids_list: train_files.extend(subject_trial_files[subj_id])
        for subj_id in val_subj_ids_list: val_files.extend(subject_trial_files[subj_id])

        train_dataset = GaitPhaseDataset30Hz(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, 
                                            fit_scaler=True, calculate_class_weights=config.USE_WEIGHTED_LOSS)
        scaler = train_dataset.get_scaler()
        if config.USE_WEIGHTED_LOSS: class_weights = train_dataset.get_class_weights()
        val_dataset = GaitPhaseDataset30Hz(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler)
        test_loader = None
        if test_files:
            test_dataset = GaitPhaseDataset30Hz(test_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler, time_col=config.TIME_COLUMN)
            test_loader = DataLoader(test_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    if config.K_FOLDS > 1:
        return train_loader, val_loader, scaler, class_weights
    else:
        return train_loader, val_loader, test_loader, scaler, class_weights