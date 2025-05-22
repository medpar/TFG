# model/config.py
import torch
import os

# --- Data Configuration ---
BASE_DATA_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected"
OUTPUT_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/tests_BiLSTM-CNN" # New output dir for this test
SUBJECT_DIRS_PATTERN = "S*"                         
TRIAL_FILE_PATTERN = "S*_A01_T*_corrected.csv"     

FEATURE_COLUMNS = ['knee_angle_l', 'knee_angle_r'] 
TARGET_COLUMN = 'phase'
TIME_COLUMN = 'time'
NUM_FEATURES = len(FEATURE_COLUMNS)
NUM_CLASSES = 3                                     # 0: Swing, 1: Stance, 2: Turn 

# --- Preprocessing ---
SEQUENCE_LENGTH = 100  # Kept the same for now, but could be a tuning parameter
NORMALIZATION_METHOD = "standard" 

# --- Model Hyperparameters ---
MODEL_TYPE = "BiLSTMCnn1D"

# CNN Specific Hyperparameters - REDUCED COMPLEXITY
# Fewer filters, especially in later layers. Simpler kernels.
CNN_OUT_CHANNELS = [16, 48] # Reduced from [32, 64]. Start with fewer feature maps.
CNN_KERNEL_SIZES = [7, 3]   # Slightly smaller kernels.
CNN_STRIDES = [1, 1]        
CNN_PADDING = 'same'        
CNN_ACTIVATION = "relu"     
CNN_DROPOUT = 0.1           # Increased CNN dropout

# BiLSTM Specific Hyperparameters - REDUCED COMPLEXITY
LSTM_HIDDEN_SIZE = 32       # Reduced from 128. Fewer LSTM units.
NUM_LSTM_LAYERS = 1         # Reduced from 2. One BiLSTM layer is often enough after CNN.
LSTM_DROPOUT = 0.3            # Not applicable for NUM_LSTM_LAYERS = 1. If > 1, use 0.4 or 0.5
BIDIRECTIONAL_LSTM = True   

# Classifier Head - INCREASED REGULARIZATION
LINEAR_DROPOUT = 0.6        # Increased dropout before the final layer.

# --- Training Hyperparameters ---
LEARNING_RATE = 0.0007      # Slightly reduced learning rate.
BATCH_SIZE = 64             # Kept the same, but smaller batches (e.g., 32) can sometimes add noise that helps regularization.
NUM_EPOCHS = 100            # Early stopping will hopefully prevent running all epochs.
WEIGHT_DECAY = 3.3e-5         # Slightly increased L2 regularization.

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --- Cross-Validation & Splitting (subject-wise) ---
K_FOLDS = 3 
TEST_SPLIT_RATIO = 0.15 
VALIDATION_SPLIT_RATIO = 0.15 
RANDOM_SEED = 42

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 16 # Slightly increased patience to see if it finds a better minimum later
EARLY_STOPPING_DELTA = 0.001 # Minimum change to qualify as an improvement (can be made smaller if loss fluctuates a lot)

# --- Inference ---
INFERENCE_BATCH_SIZE = 128
DEFAULT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_fold0.pth") 

# --- Plotting ---
PLOT_MAX_SAMPLES_INFERENCE = 20000 

# Create the new output directory
os.makedirs(OUTPUT_DIR, exist_ok=True) 

if __name__ == '__main__':
    print(f"Configuration loaded for reducing overfitting.")
    print(f"Device: {DEVICE}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Base Data Directory: {BASE_DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"CNN Channels: {CNN_OUT_CHANNELS}, Kernels: {CNN_KERNEL_SIZES}, Dropout: {CNN_DROPOUT}")
    print(f"LSTM Hidden: {LSTM_HIDDEN_SIZE}, Layers: {NUM_LSTM_LAYERS}")
    print(f"Linear Dropout: {LINEAR_DROPOUT}")
    print(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")