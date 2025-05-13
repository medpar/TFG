# data_loader.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GroupKFold
import torch
from torch.utils.data import Dataset, DataLoader
import config

def get_subject_trial_files(base_dir, subject_pattern, trial_pattern):
    """
    Scans the base directory for subject folders and their trial files.
    Returns a dictionary: {subject_id: [list_of_trial_filepaths]}
    """
    subject_trial_files = {}
    subject_dirs = sorted(glob.glob(os.path.join(base_dir, subject_pattern)))
    
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        trial_files = sorted(glob.glob(os.path.join(subj_dir, trial_pattern)))
        if trial_files:
            subject_trial_files[subject_id] = trial_files
            
    print(f"Found {len(subject_trial_files)} subjects with trial data.")
    return subject_trial_files

def create_sequences(data_df, sequence_length, feature_cols, target_col, time_col=None):
    """
    Creates sequences from a single trial DataFrame.
    Returns: List of (feature_sequence, target_sequence, time_sequence (optional))
    """
    sequences = []
    features = data_df[feature_cols].values
    targets = data_df[target_col].values
    times = data_df[time_col].values if time_col and time_col in data_df.columns else None
    
    num_samples = len(data_df)
    for i in range(num_samples - sequence_length + 1):
        feature_seq = features[i : i + sequence_length]
        target_seq = targets[i : i + sequence_length] # Target for each timestep in sequence
        
        if times is not None:
            time_seq = times[i : i + sequence_length]
            sequences.append((feature_seq, target_seq, time_seq))
        else:
            sequences.append((feature_seq, target_seq))
            
    return sequences

class GaitPhaseDataset(Dataset):
    def __init__(self, trial_filepaths, sequence_length, feature_cols, target_col, time_col=None, scaler=None, fit_scaler=False):
        self.trial_filepaths = trial_filepaths
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.time_col = time_col
        self.scaler = scaler
        
        self.sequences = []
        self.all_features_for_scaling = [] # Used only if fit_scaler is True

        self._load_and_process_data(fit_scaler)

    def _load_and_process_data(self, fit_scaler):
        print(f"Loading and processing data for {len(self.trial_filepaths)} trials...")
        
        all_dfs = []
        for file_path in self.trial_filepaths:
            try:
                df = pd.read_csv(file_path)
                if not df.empty and all(col in df.columns for col in self.feature_cols + [self.target_col]):
                    # Ensure target is integer type
                    df[self.target_col] = df[self.target_col].astype(int)
                    all_dfs.append(df)
                    if fit_scaler:
                        self.all_features_for_scaling.append(df[self.feature_cols].values)
            except Exception as e:
                print(f"Warning: Could not read or process {file_path}: {e}")
        
        if not all_dfs:
            print("Warning: No valid data files found or processed.")
            return

        if fit_scaler and self.all_features_for_scaling:
            combined_features = np.concatenate(self.all_features_for_scaling, axis=0)
            if config.NORMALIZATION_METHOD == "standard":
                self.scaler = StandardScaler()
            elif config.NORMALIZATION_METHOD == "minmax":
                self.scaler = MinMaxScaler()
            
            if self.scaler:
                print(f"Fitting scaler on {combined_features.shape[0]} samples from {len(self.trial_filepaths)} trials.")
                self.scaler.fit(combined_features)
        
        for df in all_dfs:
            if self.scaler:
                df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
            
            trial_sequences = create_sequences(df, self.sequence_length, self.feature_cols, self.target_col, self.time_col)
            self.sequences.extend(trial_sequences)
            
        print(f"Created {len(self.sequences)} sequences in total.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feature_seq, target_seq, *time_seq_tuple = self.sequences[idx]
        
        feature_tensor = torch.tensor(feature_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.long) # CrossEntropyLoss expects long
        
        if time_seq_tuple: # If time_col was provided
            time_seq = time_seq_tuple[0]
            # Time is usually not directly fed to model but useful for inference output
            return feature_tensor, target_tensor, time_seq 
        else:
            return feature_tensor, target_tensor

    def get_scaler(self):
        return self.scaler


def get_data_loaders(fold_num=None, subject_trial_files=None):
    """
    Prepares data loaders for training, validation, and (optionally) testing.
    Handles subject-wise splitting and cross-validation.
    """
    if subject_trial_files is None:
        subject_trial_files = get_subject_trial_files(
            config.BASE_DATA_DIR, 
            config.SUBJECT_DIRS_PATTERN, 
            config.TRIAL_FILE_PATTERN
        )

    subject_ids = sorted(list(subject_trial_files.keys()))
    if not subject_ids:
        raise ValueError("No subjects found. Check data directory and patterns in config.py.")

    train_subj_ids, val_subj_ids, test_subj_ids = [], [], []
    train_files, val_files, test_files = [], [], []
    scaler = None

    if config.K_FOLDS > 1:
        print(f"Using {config.K_FOLDS}-Fold Cross-Validation.")
        if fold_num is None:
            raise ValueError("fold_num must be specified for cross-validation.")
        
        gkf = GroupKFold(n_splits=config.K_FOLDS)
        # Create groups array (same group for all trials of a subject)
        groups = []
        all_files_for_cv_meta = [] # Store (filepath, subject_id) for group assignment
        for subj_id in subject_ids:
            for trial_file in subject_trial_files[subj_id]:
                all_files_for_cv_meta.append(subj_id) # Group by subject_id

        # Convert subject_ids to an array for gkf.split
        unique_subj_ids_arr = np.array(subject_ids)

        current_fold = 0
        for train_idx, val_idx in gkf.split(unique_subj_ids_arr, groups=unique_subj_ids_arr): # X is subjects, groups are subjects
            if current_fold == fold_num:
                train_subj_ids = [unique_subj_ids_arr[i] for i in train_idx]
                val_subj_ids = [unique_subj_ids_arr[i] for i in val_idx]
                break
            current_fold += 1
        
        print(f"Fold {fold_num}: {len(train_subj_ids)} train subjects, {len(val_subj_ids)} val subjects.")

        for subj_id in train_subj_ids: train_files.extend(subject_trial_files[subj_id])
        for subj_id in val_subj_ids: val_files.extend(subject_trial_files[subj_id])
        
        # For CV, scaler is fit on each training fold
        train_dataset = GaitPhaseDataset(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, fit_scaler=True)
        scaler = train_dataset.get_scaler()
        val_dataset = GaitPhaseDataset(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler)

    else: # Single train/val/test split
        print("Using single train/val/test split.")
        # First, split into train+val and test
        if config.TEST_SPLIT_RATIO > 0:
            train_val_subj_ids, test_subj_ids = train_test_split(
                subject_ids, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_SEED
            )
            for subj_id in test_subj_ids: test_files.extend(subject_trial_files[subj_id])
            print(f"Test set: {len(test_subj_ids)} subjects.")
        else:
            train_val_subj_ids = subject_ids
            print("No test set specified by TEST_SPLIT_RATIO=0.")

        # Split train+val into train and validation
        train_subj_ids, val_subj_ids = train_test_split(
            train_val_subj_ids, test_size=config.VALIDATION_SPLIT_RATIO / (1 - config.TEST_SPLIT_RATIO), 
            random_state=config.RANDOM_SEED
        ) # Adjust val ratio because it's from train_val_pool
        
        for subj_id in train_subj_ids: train_files.extend(subject_trial_files[subj_id])
        for subj_id in val_subj_ids: val_files.extend(subject_trial_files[subj_id])
        
        print(f"Train set: {len(train_subj_ids)} subjects.")
        print(f"Validation set: {len(val_subj_ids)} subjects.")

        train_dataset = GaitPhaseDataset(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, fit_scaler=True)
        scaler = train_dataset.get_scaler()
        val_dataset = GaitPhaseDataset(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler)
        
        if test_files:
            test_dataset = GaitPhaseDataset(test_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler, time_col=config.TIME_COLUMN) # Include time for inference output
            test_loader = DataLoader(test_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)
        else:
            test_loader = None

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    if config.K_FOLDS > 1:
        return train_loader, val_loader, scaler # No separate test_loader in CV loop directly
    else:
        return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    print("Testing data loader...")
    # Example of how to use:
    subject_data = get_subject_trial_files(config.BASE_DATA_DIR, config.SUBJECT_DIRS_PATTERN, config.TRIAL_FILE_PATTERN)
    
    if config.K_FOLDS > 1:
        print(f"\n--- Cross-Validation Mode (Example: Fold 0) ---")
        train_loader, val_loader, scaler = get_data_loaders(fold_num=0, subject_trial_files=subject_data)
        print(f"Train loader: {len(train_loader.dataset)} sequences")
        print(f"Validation loader: {len(val_loader.dataset)} sequences")
        if scaler: print("Scaler fitted on training data of fold 0.")
    else:
        print(f"\n--- Single Split Mode ---")
        train_loader, val_loader, test_loader, scaler = get_data_loaders(subject_trial_files=subject_data)
        print(f"Train loader: {len(train_loader.dataset)} sequences")
        print(f"Validation loader: {len(val_loader.dataset)} sequences")
        if test_loader: print(f"Test loader: {len(test_loader.dataset)} sequences")
        if scaler: print("Scaler fitted on training data.")

    # Check a batch
    if len(train_loader) > 0:
        features_batch, targets_batch = next(iter(train_loader))
        print(f"\nSample batch shapes:")
        print(f"Features: {features_batch.shape}") # Expected: [batch_size, seq_len, num_features]
        print(f"Targets: {targets_batch.shape}")   # Expected: [batch_size, seq_len]
        print(f"Feature example: {features_batch[0,0,:]}") # First feature vector of first sequence in batch
        print(f"Target example (first sequence): {targets_batch[0,:]}")