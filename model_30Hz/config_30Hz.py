# model_30Hz/config_30Hz.py
import torch
import os

# --- Sampling Rate Configuration ---
# This is the target sampling rate for this model.
SAMPLING_RATE = 30  # Hz

# --- Data Configuration ---
# For TRAINING, we use 50Hz IMU-labeled data and downsample it.
TRAIN_DATA_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected"
TRAIN_TRIAL_PATTERN = "S*_A01_T*_corrected.csv"

# For INFERENCE, we use the 30Hz data generated from video models.
INFERENCE_DATA_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/jointangles_motionagformer_dataset/"
INFERENCE_TRIAL_PATTERN = "*.csv" # Or a more specific pattern if needed, e.g., "*_all_angles.csv"

# --- Output Directories ---
# A new base output directory for the 30Hz model's artifacts
BASE_OUTPUT_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/tests_model_30Hz/"
# Subdirectories for training and inference artifacts
TRAIN_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "train") 
INFERENCE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "inference") 

SUBJECT_DIRS_PATTERN = "S*"                         

# --- Feature & Target Configuration ---
FEATURE_COLUMNS = ['knee_angle_l', 'knee_angle_r'] 
TARGET_COLUMN = 'phase'
TIME_COLUMN = 'time'
NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_CLASSES = 3  # 0: Swing, 1: Stance, 2: Turn 

# --- Preprocessing ---
SEQUENCE_LENGTH = 75                   # TAMAÑO VENTANA (This now corresponds to 75 / 30Hz = 2.5 seconds)
NORMALIZATION_METHOD = "standard" 

# --- Loss Configuration ---
USE_WEIGHTED_LOSS = True 

# --- Optimization Metric ---
OPTIMIZE_METRIC = 'f1' 
OPTIMIZE_METRIC_FOR_OPTUNA = 'f1'

# --- Model Hyperparameters (Kept from your optimized 50Hz version) ---
MODEL_TYPE = "BiLSTM"
LSTM_HIDDEN_SIZE = 128       
NUM_LSTM_LAYERS = 1  
LSTM_DROPOUT = 0.5          
BIDIRECTIONAL_LSTM = True   
LINEAR_DROPOUT = 0.5       

# --- Training Hyperparameters (Kept from your optimized 50Hz version) ---
LEARNING_RATE = 0.0001757600399129188  
BATCH_SIZE = 64
NUM_EPOCHS = 100 
WEIGHT_DECAY = 0.00010510373125205576
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Cross-Validation & Splitting ---
K_FOLDS = 3
RANDOM_SEED = 42

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 20 
EARLY_STOPPING_DELTA = 0.0005 

# --- Inference ---
INFERENCE_BATCH_SIZE = 64
# Path to the trained model. This will be created by train_30Hz.py
DEFAULT_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, "best_model_fold0.pth") 

# --- Plotting ---
PLOT_MAX_SAMPLES_INFERENCE = 20000 

# Create directories if they don't exist
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

if K_FOLDS <= 1:
    TEST_SPLIT_RATIO = 0.15 
    VALIDATION_SPLIT_RATIO = 0.15
else:
    TEST_SPLIT_RATIO = None
    VALIDATION_SPLIT_RATIO = None

if __name__ == '__main__':
    print(f"--- 30Hz Model Configuration Loaded ---")
    print(f"Target Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"Training Data Directory (to be downsampled): {TRAIN_DATA_DIR}")
    print(f"Inference Data Directory (already 30Hz): {INFERENCE_DATA_DIR}")
    print(f"Training Output Directory: {TRAIN_OUTPUT_DIR}")
    print(f"Inference Output Directory: {INFERENCE_OUTPUT_DIR}")