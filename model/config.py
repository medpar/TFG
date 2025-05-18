# config.py
import torch
import os

# --- Data Configuration ---
BASE_DATA_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected"
OUTPUT_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/model/" 
SUBJECT_DIRS_PATTERN = "S*"                         
TRIAL_FILE_PATTERN = "S*_A01_T*_corrected.csv"     

FEATURE_COLUMNS = ['knee_angle_l', 'knee_angle_r'] 
TARGET_COLUMN = 'phase'
TIME_COLUMN = 'time'
NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_CLASSES = 3                                     # 0: Swing, 1: Stance, 2: Turn 

# --- Preprocessing ---
SEQUENCE_LENGTH = 100  # e.g., 100 timesteps (2 seconds at 50Hz)
NORMALIZATION_METHOD = "standard" # "standard" or "minmax" or None

# --- Model Hyperparameters ---
MODEL_TYPE = "LSTM" # Could be "BiLSTM" or other variants if you extend
LSTM_HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3 # Dropout between LSTM layers if num_layers > 1
BIDIRECTIONAL_LSTM = True   # We use BiLSTM
LINEAR_DROPOUT = 0.4 # Dropout before the final classification layer

# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100 # Max epochs; early stopping will be used
WEIGHT_DECAY = 1e-5 # For AdamW optimizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cross-Validation & Splitting (subject-wise) ---
K_FOLDS = 1 # Number of folds for cross-validation. Set to 1 for a single train/val/test split.
TEST_SPLIT_RATIO = 0.15 # Proportion of subjects for the final test set (if K_FOLDS=1)
VALIDATION_SPLIT_RATIO = 0.15 # Proportion of subjects for validation (if K_FOLDS=1, taken from training set)
RANDOM_SEED = 42

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 0.001 # Minimum change to qualify as an improvement

# --- Inference ---
INFERENCE_BATCH_SIZE = 128
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_fold0.pth")

# --- Plotting ---
PLOT_MAX_SAMPLES_INFERENCE = 2000 # Max samples to plot for inference visualization

# Ajustes para entrenar rápido para debug
# NUM_EPOCHS = 20
# K_FOLDS = 3


os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    print(f"Configuration loaded.")
    print(f"Device: {DEVICE}")
    print(f"Base Data Directory: {BASE_DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")