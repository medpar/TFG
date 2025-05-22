# model/data_loader.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GroupKFold
import torch
from torch.utils.data import Dataset, DataLoader
import config # Make sure config.py is in the same directory or accessible

# This function is defined at the module level
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

    print(f"Found {len(subject_trial_files)} subjects with trial data matching pattern '{trial_pattern}' in '{base_dir}'.")
    if not subject_trial_files:
        print(f"Warning: No subject trial files found. Check BASE_DATA_DIR, SUBJECT_DIRS_PATTERN, and TRIAL_FILE_PATTERN in config.py.")
        print(f"  BASE_DATA_DIR: {config.BASE_DATA_DIR}")
        print(f"  SUBJECT_DIRS_PATTERN: {config.SUBJECT_DIRS_PATTERN}")
        print(f"  TRIAL_FILE_PATTERN: {config.TRIAL_FILE_PATTERN}")

    return subject_trial_files

def create_sequences(data_df, sequence_length, feature_cols, target_col, time_col=None):
    """
    Creates sequences from a single trial DataFrame.
    Returns: List of (feature_sequence, target_sequence, time_sequence (optional))
    """
    sequences = []
    # Ensure feature_cols are present before attempting to access .values
    if not all(col in data_df.columns for col in feature_cols):
        print(f"Warning: Missing one or more feature columns ({feature_cols}) in DataFrame. Available: {data_df.columns.tolist()}. Skipping sequence creation for this df.")
        return sequences

    features = data_df[feature_cols].values
    targets = data_df[target_col].values
    times = data_df[time_col].values if time_col and time_col in data_df.columns else None

    num_samples = len(data_df)
    if num_samples < sequence_length: # Check if df is long enough
        # print(f"Warning: DataFrame length ({num_samples}) is less than sequence_length ({sequence_length}). No sequences created.")
        return sequences

    for i in range(num_samples - sequence_length + 1):
        feature_seq = features[i : i + sequence_length]
        target_seq = targets[i : i + sequence_length]

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
        if fit_scaler: # Initialize only if fitting
            self.all_features_for_scaling = []

        self._load_and_process_data(fit_scaler)

    def _load_and_process_data(self, fit_scaler):
        print(f"GaitPhaseDataset: Loading and processing data for {len(self.trial_filepaths)} trials...")

        all_dfs_for_sequencing = [] # Store DataFrames that are valid for sequencing
        
        # If fitting scaler, collect all features first
        if fit_scaler:
            features_to_fit_scaler = []
            for file_path in self.trial_filepaths:
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty and all(col in df.columns for col in self.feature_cols + [self.target_col]):
                        features_to_fit_scaler.append(df[self.feature_cols].values)
                    # No need to append to all_dfs_for_sequencing yet, do it in the next loop
                except Exception as e:
                    print(f"Warning: Could not read or process {file_path} during scaler fitting pass: {e}")

            if features_to_fit_scaler:
                combined_features_np = np.concatenate(features_to_fit_scaler, axis=0)
                if config.NORMALIZATION_METHOD == "standard":
                    self.scaler = StandardScaler()
                elif config.NORMALIZATION_METHOD == "minmax":
                    self.scaler = MinMaxScaler()
                
                if self.scaler:
                    print(f"Fitting scaler on {combined_features_np.shape[0]} samples from {len(features_to_fit_scaler)} feature sets.")
                    self.scaler.fit(combined_features_np)
            else:
                print("Warning: No features found to fit the scaler.")

        # Second pass: load, (optionally scale if scaler exists), and create sequences
        for file_path in self.trial_filepaths:
            try:
                df = pd.read_csv(file_path)
                # Basic checks for necessary columns
                if df.empty or not all(col in df.columns for col in self.feature_cols + [self.target_col]):
                    print(f"Warning: Skipping file {file_path} due to missing columns or empty dataframe.")
                    continue
                
                df[self.target_col] = df[self.target_col].astype(int)

                if self.scaler:
                    # Ensure only existing columns are scaled and DataFrame structure is maintained
                    df_features_scaled_np = self.scaler.transform(df[self.feature_cols].values)
                    df[self.feature_cols] = df_features_scaled_np # Assign scaled numpy array back
                
                all_dfs_for_sequencing.append(df) # Add processed df for sequence creation
            except Exception as e:
                print(f"Warning: Could not read or process {file_path} during main processing pass: {e}")
        
        if not all_dfs_for_sequencing:
            print("Warning: No valid data files found or processed for sequence creation.")
            return

        for df_processed in all_dfs_for_sequencing:
            trial_sequences = create_sequences(df_processed, self.sequence_length, self.feature_cols, self.target_col, self.time_col)
            self.sequences.extend(trial_sequences)

        print(f"Created {len(self.sequences)} sequences in total.")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Check if sequences were actually created
        if not self.sequences:
            raise IndexError("Dataset is empty, no sequences available.")
            
        sequence_item = self.sequences[idx]
        
        # Handle if sequence_item doesn't have time_seq (if time_col was None)
        if len(sequence_item) == 3:
            feature_seq, target_seq, time_seq = sequence_item
        elif len(sequence_item) == 2:
            feature_seq, target_seq = sequence_item
            time_seq = None # Or handle as an error if time is always expected
        else:
            raise ValueError(f"Unexpected sequence item format: {sequence_item}")

        feature_tensor = torch.tensor(feature_seq, dtype=torch.float32)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)

        if time_seq is not None:
            return feature_tensor, target_tensor, time_seq
        else:
            return feature_tensor, target_tensor


    def get_scaler(self):
        return self.scaler

# This function is also defined at the module level
def get_data_loaders(fold_num=None, subject_trial_files=None):
    """
    Prepares data loaders for training, validation, and (optionally) testing.
    Handles subject-wise splitting and cross-validation.
    """
    if subject_trial_files is None:
        # THIS IS THE CALL: it calls the function defined above in this same file.
        subject_trial_files = get_subject_trial_files(
            config.BASE_DATA_DIR,
            config.SUBJECT_DIRS_PATTERN,
            config.TRIAL_FILE_PATTERN
        )

    subject_ids = sorted(list(subject_trial_files.keys()))
    if not subject_ids:
        # This error is critical and means no data files were found by get_subject_trial_files
        raise ValueError("No subjects found with trial data. Check data directory and patterns in config.py, "
                         "and ensure get_subject_trial_files is working correctly.")

    train_subj_ids, val_subj_ids, test_subj_ids = [], [], []
    train_files, val_files, test_files = [], [], []
    scaler = None # Scaler will be fit on training data

    if config.K_FOLDS > 1:
        print(f"Using {config.K_FOLDS}-Fold Cross-Validation.")
        if fold_num is None:
            raise ValueError("fold_num must be specified for cross-validation.")

        gkf = GroupKFold(n_splits=config.K_FOLDS)
        unique_subj_ids_arr = np.array(subject_ids)

        # Check if there are enough subjects for the number of folds
        if len(unique_subj_ids_arr) < config.K_FOLDS:
            raise ValueError(f"Number of subjects ({len(unique_subj_ids_arr)}) is less than K_FOLDS ({config.K_FOLDS})."
                             " Reduce K_FOLDS or add more subjects.")

        current_fold_count = 0
        # gkf.split expects X to be data, but we only need indices for subjects, so pass unique_subj_ids_arr
        # groups should be the subject IDs themselves to ensure subject-wise split
        for train_idx, val_idx in gkf.split(X=unique_subj_ids_arr, y=None, groups=unique_subj_ids_arr):
            if current_fold_count == fold_num:
                train_subj_ids = [unique_subj_ids_arr[i] for i in train_idx]
                val_subj_ids = [unique_subj_ids_arr[i] for i in val_idx]
                break
            current_fold_count += 1
        
        if not train_subj_ids or not val_subj_ids:
             raise ValueError(f"Fold {fold_num} resulted in empty train or validation subject IDs. Check K_FOLDS and subject data.")

        print(f"Fold {fold_num}: {len(train_subj_ids)} train subjects, {len(val_subj_ids)} val subjects.")

        for subj_id in train_subj_ids: train_files.extend(subject_trial_files.get(subj_id, []))
        for subj_id in val_subj_ids: val_files.extend(subject_trial_files.get(subj_id, []))
        
        if not train_files:
            raise ValueError(f"No training files found for fold {fold_num}. Check subject IDs and file paths.")
        
        train_dataset = GaitPhaseDataset(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, fit_scaler=True)
        scaler = train_dataset.get_scaler() # Get the fitted scaler

        if not val_files:
            print(f"Warning: No validation files found for fold {fold_num}. Validation loader will be empty.")
            val_dataset = GaitPhaseDataset([], config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, scaler=scaler) # Empty dataset
        else:
            val_dataset = GaitPhaseDataset(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, scaler=scaler)


    else: # Single train/val/test split
        print("Using single train/val/test split.")
        train_val_subj_ids = subject_ids
        test_subj_ids_final = [] # Initialize

        if config.TEST_SPLIT_RATIO > 0 and len(subject_ids) * config.TEST_SPLIT_RATIO >= 1:
            train_val_subj_ids, test_subj_ids_final = train_test_split(
                subject_ids, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_SEED, shuffle=True
            )
            for subj_id in test_subj_ids_final: test_files.extend(subject_trial_files.get(subj_id, []))
            print(f"Test set: {len(test_subj_ids_final)} subjects.")
        else:
            print("No test set created (TEST_SPLIT_RATIO is 0 or not enough subjects).")

        # Calculate validation split ratio relative to the train_val_pool
        if (1 - config.TEST_SPLIT_RATIO) <= 0: # Avoid division by zero if test_split is 100%
            effective_val_ratio = config.VALIDATION_SPLIT_RATIO
        else:
            effective_val_ratio = config.VALIDATION_SPLIT_RATIO / (1 - config.TEST_SPLIT_RATIO) if (1-config.TEST_SPLIT_RATIO) > 0 else 0


        if effective_val_ratio > 0 and len(train_val_subj_ids) * effective_val_ratio >=1 :
            train_subj_ids, val_subj_ids = train_test_split(
                train_val_subj_ids, test_size=effective_val_ratio,
                random_state=config.RANDOM_SEED, shuffle=True
            )
        else: # Not enough data for validation split, or ratio is 0
            train_subj_ids = train_val_subj_ids
            val_subj_ids = [] # No validation subjects
            print("No validation set created (VALIDATION_SPLIT_RATIO is 0 or not enough subjects in train_val pool).")

        for subj_id in train_subj_ids: train_files.extend(subject_trial_files.get(subj_id, []))
        for subj_id in val_subj_ids: val_files.extend(subject_trial_files.get(subj_id, []))
        
        print(f"Train set: {len(train_subj_ids)} subjects, {len(train_files)} files.")
        print(f"Validation set: {len(val_subj_ids)} subjects, {len(val_files)} files.")

        if not train_files:
            raise ValueError("No training files found for single split. Check subject IDs and file paths.")

        train_dataset = GaitPhaseDataset(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, fit_scaler=True)
        scaler = train_dataset.get_scaler()
        
        if not val_files:
            print("Warning: Validation files list is empty. Validation loader will be empty.")
            val_dataset = GaitPhaseDataset([], config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, scaler=scaler)
        else:
            val_dataset = GaitPhaseDataset(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, scaler=scaler)
        
        test_loader = None
        if test_files:
            test_dataset = GaitPhaseDataset(test_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, config.TIME_COLUMN, scaler=scaler)
            if len(test_dataset) > 0:
                 test_loader = DataLoader(test_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True) # num_workers=0 for MPS sometimes
        
    # Create DataLoaders
    # Set num_workers=0 for MPS if you encounter issues with multiprocessing
    num_workers_setting = 0 if config.DEVICE == "mps" else 4 

    if len(train_dataset) == 0:
        print("Warning: Training dataset is empty. Train loader will be empty.")
        # Optuna might prune this trial if it can't train.
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True) # Still create, might be handled by caller
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers_setting, pin_memory=True)

    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Validation loader will be empty.")
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers_setting, pin_memory=True)

    if config.K_FOLDS > 1:
        return train_loader, val_loader, scaler
    else:
        return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    print("Testing data loader directly...")
    print(f"Config BASE_DATA_DIR: {config.BASE_DATA_DIR}")
    print(f"Config TRIAL_FILE_PATTERN: {config.TRIAL_FILE_PATTERN}")
    
    # Test get_subject_trial_files directly
    print("\n--- Testing get_subject_trial_files ---")
    subject_data = get_subject_trial_files(config.BASE_DATA_DIR, config.SUBJECT_DIRS_PATTERN, config.TRIAL_FILE_PATTERN)
    if not subject_data:
        print("get_subject_trial_files returned no data. Exiting test.")
    else:
        print(f"Found data for {len(subject_data)} subjects.")
        first_subj_key = list(subject_data.keys())[0]
        print(f"Example files for subject {first_subj_key}: {subject_data[first_subj_key][:2]}") # Print first 2 files

        if config.K_FOLDS > 1:
            print(f"\n--- Testing CV Mode (Fold 0) ---")
            try:
                train_loader_test, val_loader_test, scaler_test = get_data_loaders(fold_num=0, subject_trial_files=subject_data)
                if train_loader_test and len(train_loader_test.dataset) > 0:
                    print(f"Train loader (CV): {len(train_loader_test.dataset)} sequences")
                else:
                    print("Train loader (CV) is empty or None.")
                if val_loader_test and len(val_loader_test.dataset) > 0:
                     print(f"Validation loader (CV): {len(val_loader_test.dataset)} sequences")
                else:
                    print("Validation loader (CV) is empty or None.")

                if scaler_test: print("Scaler fitted on training data of fold 0.")
            except Exception as e:
                print(f"Error during CV mode test: {e}")
                import traceback
                traceback.print_exc()

        else: # Single split mode
            print(f"\n--- Testing Single Split Mode ---")
            try:
                train_loader_test, val_loader_test, test_loader_test, scaler_test = get_data_loaders(subject_trial_files=subject_data)
                if train_loader_test and len(train_loader_test.dataset) > 0:
                    print(f"Train loader (Single Split): {len(train_loader_test.dataset)} sequences")
                else:
                    print("Train loader (Single Split) is empty or None.")

                if val_loader_test and len(val_loader_test.dataset) > 0:
                    print(f"Validation loader (Single Split): {len(val_loader_test.dataset)} sequences")
                else:
                    print("Validation loader (Single Split) is empty or None.")

                if test_loader_test and len(test_loader_test.dataset) > 0:
                    print(f"Test loader (Single Split): {len(test_loader_test.dataset)} sequences")
                else:
                    print("Test loader (Single Split) is empty or None.")
                if scaler_test: print("Scaler fitted on training data (Single Split).")
            except Exception as e:
                print(f"Error during single split mode test: {e}")
                import traceback
                traceback.print_exc()


        # Check a batch from train_loader_test if it's not None and has data
        if 'train_loader_test' in locals() and train_loader_test and len(train_loader_test.dataset) > 0:
            print("\n--- Checking a sample batch ---")
            try:
                # Check if time_col is used for this dataset (based on how GaitPhaseDataset is called in get_data_loaders)
                # get_data_loaders calls GaitPhaseDataset with config.TIME_COLUMN
                if config.TIME_COLUMN:
                    features_batch_test, targets_batch_test, time_batch_test = next(iter(train_loader_test))
                    print(f"Time batch shape: {time_batch_test.shape if hasattr(time_batch_test, 'shape') else 'N/A'}")
                else:
                    features_batch_test, targets_batch_test = next(iter(train_loader_test))
                
                print(f"Features batch shape: {features_batch_test.shape}")
                print(f"Targets batch shape: {targets_batch_test.shape}")
                print(f"Feature example (first sequence, first step): {features_batch_test[0,0,:]}")
                print(f"Target example (first sequence): {targets_batch_test[0,:]}")
            except Exception as e:
                print(f"Error checking a batch: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSkipping batch check as train_loader_test is empty or None.")