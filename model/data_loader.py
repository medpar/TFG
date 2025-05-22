# model/data_loader.py
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
import config

# get_subject_trial_files and create_sequences remain the same

def get_subject_trial_files(base_dir, subject_pattern, trial_pattern):
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


class GaitPhaseDataset(Dataset):
    def __init__(self, trial_filepaths, sequence_length, feature_cols, target_col, 
                 time_col=None, scaler=None, fit_scaler=False, calculate_class_weights=False): # Added calculate_class_weights
        self.trial_filepaths = trial_filepaths
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.time_col = time_col
        self.scaler = scaler
        self.calculate_class_weights = calculate_class_weights # Store the flag

        self.sequences = []
        self.all_features_for_scaling = []
        self.class_counts = np.zeros(config.NUM_CLASSES, dtype=int) # Initialize class counts
        self.class_weights = None # Will store computed weights

        self._load_and_process_data(fit_scaler)

    def _load_and_process_data(self, fit_scaler):
        print(f"Loading and processing data for {len(self.trial_filepaths)} trials...")
        
        all_dfs_for_sequencing = [] # Store DataFrames after potential scaling
        all_target_labels_for_weight_calc = [] # Store all target labels for weight calculation

        for file_path in self.trial_filepaths:
            try:
                df = pd.read_csv(file_path)
                if not df.empty and all(col in df.columns for col in self.feature_cols + [self.target_col]):
                    df[self.target_col] = df[self.target_col].astype(int)
                    
                    # Collect all target labels if weights need to be calculated
                    if self.calculate_class_weights:
                        all_target_labels_for_weight_calc.extend(df[self.target_col].values)

                    all_dfs_for_sequencing.append(df) # Add original df for now, scaling applied later
                    if fit_scaler:
                        self.all_features_for_scaling.append(df[self.feature_cols].values)
            except Exception as e:
                print(f"Warning: Could not read or process {file_path}: {e}")
        
        if not all_dfs_for_sequencing:
            print("Warning: No valid data files found or processed.")
            return

        if fit_scaler and self.all_features_for_scaling:
            combined_features = np.concatenate(self.all_features_for_scaling, axis=0)
            if config.NORMALIZATION_METHOD == "standard": self.scaler = StandardScaler()
            elif config.NORMALIZATION_METHOD == "minmax": self.scaler = MinMaxScaler()
            if self.scaler:
                print(f"Fitting scaler on {combined_features.shape[0]} samples.")
                self.scaler.fit(combined_features)
        
        # Apply scaling and create sequences
        for df in all_dfs_for_sequencing:
            df_processed = df.copy() # Process a copy
            if self.scaler:
                df_processed[self.feature_cols] = self.scaler.transform(df_processed[self.feature_cols].values) # Use .values for numpy array
            
            trial_sequences = create_sequences(df_processed, self.sequence_length, self.feature_cols, self.target_col, self.time_col)
            self.sequences.extend(trial_sequences)
            
        print(f"Created {len(self.sequences)} sequences in total.")

        # Calculate class weights if requested (typically for the training dataset)
        if self.calculate_class_weights and all_target_labels_for_weight_calc:
            all_labels_np = np.array(all_target_labels_for_weight_calc)
            unique_classes = np.unique(all_labels_np)
            
            # Ensure all NUM_CLASSES are represented, even if some are missing in this specific dataset split
            # This is important if a class is extremely rare and might be absent in a small fold.
            present_classes = np.arange(config.NUM_CLASSES) 
            
            # Filter unique_classes to only those that are valid class indices
            valid_unique_classes = [cls for cls in unique_classes if 0 <= cls < config.NUM_CLASSES]
            if not valid_unique_classes: # If no valid classes, weights are uniform (or handle error)
                print("Warning: No valid classes found for weight calculation. Using uniform weights.")
                self.class_weights = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
            else:
                # Compute weights only for classes present in the data and valid
                weights = compute_class_weight('balanced', classes=np.array(valid_unique_classes), y=all_labels_np[np.isin(all_labels_np, valid_unique_classes)])
                
                # Assign computed weights to the full class_weights array
                # Initialize with a default (e.g., 1.0 or average if a class is missing)
                self.class_weights = np.ones(config.NUM_CLASSES, dtype=float) 
                class_to_weight_map = dict(zip(valid_unique_classes, weights))
                for i in range(config.NUM_CLASSES):
                    if i in class_to_weight_map:
                        self.class_weights[i] = class_to_weight_map[i]
                    else:
                        # If a class is completely missing, assign a neutral or high weight
                        # Assigning a high weight might be risky if it's truly absent.
                        # Assigning a weight of 1.0 is neutral. Or average of others.
                        print(f"Warning: Class {i} not present in this dataset split for weight calculation. Assigning weight 1.0.")
                        self.class_weights[i] = 1.0 


                print(f"Calculated class weights: {self.class_weights}")
        elif self.calculate_class_weights:
            print("Warning: 'calculate_class_weights' was True, but no target labels were collected.")


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


def get_data_loaders(fold_num=None, subject_trial_files=None):
    # ... (splitting logic remains largely the same) ...
    if subject_trial_files is None:
        subject_trial_files = get_subject_trial_files(
            config.BASE_DATA_DIR, 
            config.SUBJECT_DIRS_PATTERN, 
            config.TRIAL_FILE_PATTERN
        )
    subject_ids = sorted(list(subject_trial_files.keys()))
    if not subject_ids:
        raise ValueError("No subjects found.")

    train_files, val_files, test_files = [], [], []
    scaler = None
    class_weights = None # Initialize class_weights

    if config.K_FOLDS > 1:
        # ... (CV splitting logic) ...
        gkf = GroupKFold(n_splits=config.K_FOLDS)
        unique_subj_ids_arr = np.array(subject_ids)
        current_fold = 0
        train_subj_ids, val_subj_ids = [], []
        for train_idx, val_idx in gkf.split(unique_subj_ids_arr, groups=unique_subj_ids_arr):
            if current_fold == fold_num:
                train_subj_ids = [unique_subj_ids_arr[i] for i in train_idx]
                val_subj_ids = [unique_subj_ids_arr[i] for i in val_idx]
                break
            current_fold += 1
        print(f"Fold {fold_num}: {len(train_subj_ids)} train subjects, {len(val_subj_ids)} val subjects.")
        for subj_id in train_subj_ids: train_files.extend(subject_trial_files[subj_id])
        for subj_id in val_subj_ids: val_files.extend(subject_trial_files[subj_id])
        
        train_dataset = GaitPhaseDataset(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, 
                                         fit_scaler=True, calculate_class_weights=config.USE_WEIGHTED_LOSS) # Calculate weights on train set
        scaler = train_dataset.get_scaler()
        if config.USE_WEIGHTED_LOSS:
            class_weights = train_dataset.get_class_weights()
        val_dataset = GaitPhaseDataset(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler)
    else: # Single train/val/test split
        # ... (single split logic) ...
        if config.TEST_SPLIT_RATIO > 0:
            train_val_subj_ids, test_subj_ids_list = train_test_split(subject_ids, test_size=config.TEST_SPLIT_RATIO, random_state=config.RANDOM_SEED)
            for subj_id in test_subj_ids_list: test_files.extend(subject_trial_files[subj_id])
        else:
            train_val_subj_ids = subject_ids
        train_subj_ids_list, val_subj_ids_list = train_test_split(train_val_subj_ids, test_size=config.VALIDATION_SPLIT_RATIO / (1 - config.TEST_SPLIT_RATIO if config.TEST_SPLIT_RATIO < 1 else 1), random_state=config.RANDOM_SEED)
        for subj_id in train_subj_ids_list: train_files.extend(subject_trial_files[subj_id])
        for subj_id in val_subj_ids_list: val_files.extend(subject_trial_files[subj_id])

        train_dataset = GaitPhaseDataset(train_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, 
                                         fit_scaler=True, calculate_class_weights=config.USE_WEIGHTED_LOSS)
        scaler = train_dataset.get_scaler()
        if config.USE_WEIGHTED_LOSS:
            class_weights = train_dataset.get_class_weights()
        val_dataset = GaitPhaseDataset(val_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler)
        test_loader = None
        if test_files:
            test_dataset = GaitPhaseDataset(test_files, config.SEQUENCE_LENGTH, config.FEATURE_COLUMNS, config.TARGET_COLUMN, scaler=scaler, time_col=config.TIME_COLUMN)
            test_loader = DataLoader(test_dataset, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False) # num_workers=0 for MPS often
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    if config.K_FOLDS > 1:
        return train_loader, val_loader, scaler, class_weights
    else:
        return train_loader, val_loader, test_loader, scaler, class_weights

# ... (if __name__ == '__main__': block remains the same, but calls to get_data_loaders will need to handle new return value)
if __name__ == '__main__':
    print("Testing data loader...")
    if config.K_FOLDS > 1:
        train_loader, val_loader, scaler, weights = get_data_loaders(fold_num=0)
        if weights is not None: print(f"Class weights for fold 0: {weights}")
    else:
        train_loader, val_loader, test_loader, scaler, weights = get_data_loaders()
        if weights is not None: print(f"Class weights for single split: {weights}")

    if len(train_loader) > 0:
        features_batch, targets_batch = next(iter(train_loader))
        print(f"Features: {features_batch.shape}, Targets: {targets_batch.shape}")