# model/config.py
import torch
import os

# --- Data Configuration ---
BASE_DATA_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected"

# --- Output Directories ---
# OUTPUT_DIR is for training artifacts (checkpoints, training plots, etc.)
OUTPUT_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/train/" 
# INFERENCE_OUTPUT_DIR is for inference results (prediction CSVs, inference plots, etc.)
INFERENCE_OUTPUT_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/inference/" 

SUBJECT_DIRS_PATTERN = "S*"                         
TRIAL_FILE_PATTERN = "S*_A01_T*_corrected.csv"     

FEATURE_COLUMNS = ['knee_angle_l', 'knee_angle_r'] 
TARGET_COLUMN = 'phase'
TIME_COLUMN = 'time'
NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_CLASSES = 3                                     # 0: Swing, 1: Stance, 2: Turn 

# --- Preprocessing ---
SEQUENCE_LENGTH = 75                   # TAMAÑO VENTANA
NORMALIZATION_METHOD = "standard" 

# --- Loss Configuration ---
USE_WEIGHTED_LOSS = True 

# --- Optimization Metric (for Optuna and Early Stopping/Best Model) ---
# Options: 'loss' (minimize validation loss) or 'f1' (maximize validation F1 score)
OPTIMIZE_METRIC = 'f1' 
# For Optuna: Optuna reports this metric to its pruner.
# If OPTIMIZE_METRIC is 'f1', Optuna's study direction should be 'maximize'.
# If OPTIMIZE_METRIC is 'loss', Optuna's study direction should be 'minimize'.
OPTIMIZE_METRIC_FOR_OPTUNA = 'f1' # This tells train_loop what to report to Optuna pruner

# --- Model Hyperparameters ---
MODEL_TYPE = "BiLSTM"
LSTM_HIDDEN_SIZE = 128       
NUM_LSTM_LAYERS = 1  
LSTM_DROPOUT = 0.5          
BIDIRECTIONAL_LSTM = True   
LINEAR_DROPOUT = 0.5       

# --- Training Hyperparameters ---
LEARNING_RATE = 0.0001757600399129188  
BATCH_SIZE = 64
NUM_EPOCHS = 100 
WEIGHT_DECAY = 0.00010510373125205576
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Cross-Validation & Splitting ---
K_FOLDS = 5 
RANDOM_SEED = 42

# --- Early Stopping ---
# EARLY_STOPPING_PATIENCE and EARLY_STOPPING_DELTA will now apply to OPTIMIZE_METRIC
EARLY_STOPPING_PATIENCE = 20 
EARLY_STOPPING_DELTA = 0.0005 # For F1, a small change is significant. For loss, can be larger. Adjust if optimizing loss.

# --- Inference ---
INFERENCE_BATCH_SIZE = 64
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_fold0.pth") 

# --- Plotting ---
PLOT_MAX_SAMPLES_INFERENCE = 20000 

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)


if K_FOLDS <= 1:
    TEST_SPLIT_RATIO = 0.15 
    VALIDATION_SPLIT_RATIO = 0.15
else:
    TEST_SPLIT_RATIO = None
    VALIDATION_SPLIT_RATIO = None

if __name__ == '__main__':
    print(f"Configuration loaded.")
    print(f"Optimizing for: {OPTIMIZE_METRIC}")
    print(f"Using Weighted Loss: {USE_WEIGHTED_LOSS}")
    print(f"Training Output Directory: {OUTPUT_DIR}")
    print(f"Inference Output Directory: {INFERENCE_OUTPUT_DIR}")