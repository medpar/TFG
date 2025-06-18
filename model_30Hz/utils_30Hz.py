# model_30Hz/utils_30Hz.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import config_30Hz as config # Use 30Hz config
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GAITSEG_DIR = os.path.join(PARENT_DIR, "gaitseg")   

try:
    import gaitseg_utils as gal
except ImportError:
    gal = None

OUTPUT_PLOTS_DIR = os.path.join(config.TRAIN_OUTPUT_DIR, "output_plots")
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# All functions (plot_training_history, calculate_metrics, etc.) are identical to your
# previous utils.py, but now they import from config_30Hz.
# For brevity, I'll show one function to demonstrate the pattern.

def plot_training_history(history, fold_num=None, trial_num=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    title_parts = []
    if trial_num is not None: title_parts.append(f"Trial {trial_num}")
    if fold_num is not None: title_parts.append(f"Fold {fold_num}")
    title_prefix = " ".join(title_parts) + " (30Hz Model) " if title_parts else "(30Hz Model) "

    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title(f'{title_prefix}Training & Validation Loss')
    axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Loss'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(history['train_acc'], label='Train Accuracy')
    axs[1].plot(history['val_acc'], label='Validation Accuracy')
    axs[1].set_title(f'{title_prefix}Training & Validation Accuracy')
    axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Accuracy'); axs[1].legend(); axs[1].grid(True)

    plt.tight_layout()
    plot_filename_base = f"{title_prefix.strip().replace(' ', '_').lower()}_training_history.png"
    global OUTPUT_PLOTS_DIR
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base))
    print(f"Saved training history plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)}")
    plt.close(fig)

def calculate_metrics(y_true_flat, y_pred_flat, average='weighted'):
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

def plot_confusion_matrix_custom(y_true_flat, y_pred_flat, class_names, title='Confusion Matrix', fold_num=None, trial_num=None):
    if not y_true_flat or not y_pred_flat: return
    valid_labels = np.arange(len(class_names))
    y_true_filtered = [l for l in y_true_flat if l in valid_labels]
    y_pred_filtered = [p for i, p in enumerate(y_pred_flat) if y_true_flat[i] in valid_labels]
    if not y_true_filtered or not y_pred_filtered: return
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=valid_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    title_parts = []
    if trial_num is not None: title_parts.append(f"Trial {trial_num}")
    if fold_num is not None: title_parts.append(f"Fold {fold_num}")
    title_prefix = " ".join(title_parts) + " (30Hz) " if title_parts else "(30Hz) "
    ax.set_title(f'{title_prefix}{title}')
    plot_filename_base = f"{title_prefix}{title.replace(' ', '_').lower()}_confusion_matrix.png".replace(" ", "_").lower()
    global OUTPUT_PLOTS_DIR
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base))
    print(f"Saved confusion matrix to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)}")
    plt.close(fig)

def save_checkpoint(state, is_best, filename="checkpoint.pth", best_filename="best_model.pth", output_dir=None):
    if output_dir is None: output_dir = config.TRAIN_OUTPUT_DIR
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, best_filename)
        torch.save(state, best_filepath)
        print(f"Saved new best 30Hz model to {best_filepath} (Epoch {state.get('epoch', 'N/A')})")

def load_checkpoint(filepath, model, optimizer=None, device=config.DEVICE):
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at {filepath}"); return None
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 0)})")
    return checkpoint

def plot_model_predictions_vs_true_phases(timestamps, true_phases, predicted_phases, title_suffix="", trial_num=None):
    if len(timestamps) == 0: return
    n = min(config.PLOT_MAX_SAMPLES_INFERENCE, len(timestamps))
    time_plot, predicted_phases_plot = timestamps[:n], predicted_phases[:n]
    true_phases_plot = true_phases[:n] if true_phases is not None and len(true_phases) > 0 else None
    fig, ax = plt.subplots(figsize=(15, 5)); ax.set_xlabel('Time (s)'); ax.set_ylabel('Gait Phase')
    if true_phases_plot is not None: ax.plot(time_plot, true_phases_plot, color='tab:blue', linestyle='-', marker='.', label='True Phases (GUI)')
    ax.plot(time_plot, predicted_phases_plot, color='tab:red', linestyle='--', marker='x', label='Predicted Phases (Model)'); ax.set_yticks(np.arange(-1, config.NUM_CLASSES +1)); ax.legend(); ax.grid(True, linestyle=':')
    full_title = f"Model Predicted Gait Phases vs True {title_suffix}"; plt.title(full_title); plt.tight_layout()
    plot_filename_base = f"model_vs_true_phases{title_suffix.replace(' ', '_').lower()}.png"
    global OUTPUT_PLOTS_DIR; plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)); plt.close(fig)

def plot_model_predictions_vs_angular_velocity(timestamps_csv, predicted_phases_csv, raw_imu_filepath, title_suffix="", trial_num=None):
    if gal is None: return
    t_plot_master = timestamps_csv
    results_from_raw = gal.compute_velocity_and_events(raw_imu_filepath)
    t_imu_raw, omega_imu_raw = (results_from_raw[0], results_from_raw[2]) if results_from_raw and results_from_raw[0] is not None else (None, None)
    omega_plot_aligned = np.full_like(t_plot_master, np.nan, dtype=float)
    if t_imu_raw is not None and len(t_imu_raw) > 1:
        from scipy.interpolate import interp1d
        omega_interpolator = interp1d(t_imu_raw, omega_imu_raw, kind='linear', bounds_error=False, fill_value=np.nan)
        omega_plot_aligned = omega_interpolator(t_plot_master)
    n_plot = min(config.PLOT_MAX_SAMPLES_INFERENCE, len(t_plot_master))
    t_plot_master_sub, omega_plot_aligned_sub, predicted_phases_csv_sub = t_plot_master[:n_plot], omega_plot_aligned[:n_plot], predicted_phases_csv[:n_plot]
    fig, ax1 = plt.subplots(figsize=(15, 6)); ax1.plot(t_plot_master_sub, omega_plot_aligned_sub, alpha=0.7, label='Ang. Vel. (IMU)'); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Angular Velocity (rad/s)')
    phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
    legend_phases_added = set()
    for ph_val, color in phase_colors_map.items():
        mask = (predicted_phases_csv_sub == ph_val)
        if not np.any(mask): continue
        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int)); starts, ends = np.where(diff_mask == 1)[0], np.where(diff_mask == -1)[0]
        for s, e in zip(starts, ends):
            if s < e:
                label = f"Phase {ph_val} (Model)" if ph_val not in legend_phases_added else ""
                ax1.axvspan(t_plot_master_sub[s], t_plot_master_sub[e-1] + (1/config.SAMPLING_RATE), color=color, alpha=0.45, label=label)
                if label: legend_phases_added.add(ph_val)
    plt.title(f"Model Predicted Phases (30Hz) vs IMU Angular Velocity {title_suffix}"); ax1.legend(); plt.grid(True); plt.tight_layout()
    plot_filename_base = f"model_phases_vs_ang_vel{title_suffix.replace(' ', '_').lower()}.png"
    global OUTPUT_PLOTS_DIR; plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)); plt.close(fig)