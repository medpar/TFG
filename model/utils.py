# model/utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import config
import os

# Updated import for gaitseg_utils from parent directory
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GAITSEG_DIR = os.path.join(PARENT_DIR, "gaitseg")

# OUTPUT_PLOTS_DIR will now be set per trial by hyperparameter_tuning.py
# Default if utils is run standalone or directly from train.py without tuning
OUTPUT_PLOTS_DIR = os.path.join(config.OUTPUT_DIR, "output_plots_default") 
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


if GAITSEG_DIR not in sys.path:
    sys.path.append(GAITSEG_DIR)

try:
    import gaitseg_utils as gal
except ImportError:
    print("Warning: gaitseg_utils.py not found in gaitseg directory. Some plotting functions in utils.py might not work.")
    gal = None


def plot_training_history(history, fold_num=None, trial_num=None): # Added trial_num
    """Plots training and validation loss and accuracy."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    title_parts = []
    if trial_num is not None: title_parts.append(f"Trial {trial_num}")
    if fold_num is not None: title_parts.append(f"Fold {fold_num}")
    title_prefix = " ".join(title_parts) + " " if title_parts else ""


    # Plot Loss
    axs[0].plot(history['train_loss'], label='Train Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title(f'{title_prefix}Training & Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(history['train_acc'], label='Train Accuracy')
    axs[1].plot(history['val_acc'], label='Validation Accuracy')
    axs[1].set_title(f'{title_prefix}Training & Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plot_filename_base = f"{title_prefix}training_history.png".replace(" ", "_").lower()
    # Ensure OUTPUT_PLOTS_DIR is used (which should be set by the calling script like tuning.py)
    global OUTPUT_PLOTS_DIR # Access the potentially modified global
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base))
    print(f"Saved training history plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)}")
    plt.close(fig)


def calculate_metrics(y_true_flat, y_pred_flat, average='weighted'):
    # ... (no changes needed) ...
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average=average, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics

def plot_confusion_matrix_custom(y_true_flat, y_pred_flat, class_names, title='Confusion Matrix', fold_num=None, trial_num=None): # Added trial_num
    """Plots the confusion matrix."""
    if not y_true_flat or not y_pred_flat or len(y_true_flat) == 0 or len(y_pred_flat) == 0:
        print(f"Skipping confusion matrix for '{title}': No data.")
        return
    
    # Ensure labels for confusion_matrix are within the expected range of class_names
    valid_labels = np.arange(len(class_names))
    # Filter out any labels not in valid_labels if they exist (e.g. -1 if not a class)
    # This is important if y_true_flat/y_pred_flat might contain values like -1 (unclassified)
    # that are not part of the confusion matrix classes.
    y_true_filtered = [l for l in y_true_flat if l in valid_labels]
    y_pred_filtered = [p for i, p in enumerate(y_pred_flat) if y_true_flat[i] in valid_labels] # Keep predictions corresponding to valid true labels

    if not y_true_filtered or not y_pred_filtered:
        print(f"Skipping confusion matrix for '{title}': No valid class data after filtering.")
        return

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=valid_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    
    title_parts = []
    if trial_num is not None: title_parts.append(f"Trial {trial_num}")
    if fold_num is not None: title_parts.append(f"Fold {fold_num}")
    title_prefix = " ".join(title_parts) + " " if title_parts else ""
    ax.set_title(f'{title_prefix}{title}')
    
    plot_filename_base = f"{title_prefix}{title.replace(' ', '_').lower()}_confusion_matrix.png".replace(" ", "_").lower()
    global OUTPUT_PLOTS_DIR
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base))
    print(f"Saved confusion matrix to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)}")
    plt.close(fig)

def save_checkpoint(state, is_best, filename="checkpoint.pth", best_filename="best_model.pth", output_dir=None): # Allow output_dir override
    """Saves model checkpoint."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR # Default if not provided

    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, best_filename)
        torch.save(state, best_filepath)
        print(f"Saved new best model to {best_filepath} (Epoch {state.get('epoch', 'N/A')})")

def load_checkpoint(filepath, model, optimizer=None, device=config.DEVICE):
    # ... (no changes needed) ...
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at {filepath}")
        return None # Return just one None
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print(f"Loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 0)}, Best Val Metric: {checkpoint.get('best_val_metric', float('inf')):.4f})")
    return checkpoint # Return the whole checkpoint dictionary

# ... (plot_model_predictions_vs_true_phases and plot_model_predictions_vs_angular_velocity remain the same for now) ...
# You might want to add trial_num to their filenames if called from tuning script directly.

def plot_model_predictions_vs_true_phases(timestamps, true_phases, predicted_phases, title_suffix="", trial_num=None):
    if len(timestamps) == 0:
        print(f"No data to plot for true vs predicted phases ({title_suffix}).")
        return

    n = min(config.PLOT_MAX_SAMPLES_INFERENCE, len(timestamps))
    
    time_plot = timestamps[:n]
    true_phases_plot = true_phases[:n] if true_phases is not None and len(true_phases) > 0 else None
    predicted_phases_plot = predicted_phases[:n]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gait Phase')
    
    if true_phases_plot is not None:
        ax.plot(time_plot, true_phases_plot, color='tab:blue', linestyle='-', marker='.', markersize=5, alpha=0.8, label='True Phases (GUI)')
    ax.plot(time_plot, predicted_phases_plot, color='tab:red', linestyle='--', marker='x', markersize=5, alpha=0.8, label='Predicted Phases (Model)')
    
    ax.set_yticks(np.arange(-1, config.NUM_CLASSES +1)) 
    ax.set_yticklabels([f"Phase {i}" for i in np.arange(-1, config.NUM_CLASSES +1)])
    ax.legend()
    ax.grid(True, linestyle=':')
    
    full_title = f"Model Predicted Gait Phases vs True Phases {title_suffix}"
    if trial_num is not None: full_title = f"Trial {trial_num}: " + full_title
    plt.title(full_title)
    plt.tight_layout()
    
    plot_filename_base = f"model_vs_true_phases{title_suffix.replace(' ', '_').lower()}"
    if trial_num is not None: plot_filename_base = f"trial{trial_num}_" + plot_filename_base
    plot_filename_base += ".png"
    
    global OUTPUT_PLOTS_DIR
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base))
    print(f"Saved model vs true phases plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)}")
    plt.close(fig)


def plot_model_predictions_vs_angular_velocity(
    timestamps_csv, predicted_phases_csv, raw_imu_filepath, title_suffix="", trial_num=None
):
    if gal is None:
        print("gaitseg_utils not available. Skipping plot_model_predictions_vs_angular_velocity.")
        return
    if len(timestamps_csv) == 0:
        print(f"No CSV data to plot for angular velocity comparison ({title_suffix}).")
        return

    print(f"Processing .raw file for angular velocity: {raw_imu_filepath} (using gal.LEG: {gal.LEG if hasattr(gal, 'LEG') else 'N/A'})")
    results_from_raw = gal.compute_velocity_and_events(raw_imu_filepath)

    t_imu_raw, omega_imu_raw, ms_indices_imu, hs_indices_imu, to_indices_imu = None, None, [], [], []
    if results_from_raw is not None and results_from_raw[0] is not None:
        t_imu_raw, _, omega_imu_raw, \
        ms_indices_imu, _, hs_indices_imu, to_indices_imu, _ = results_from_raw

    t_plot_master = timestamps_csv
    omega_plot_aligned = np.full_like(t_plot_master, np.nan, dtype=float)
    if t_imu_raw is not None and omega_imu_raw is not None and len(t_imu_raw) > 0:
        from scipy.interpolate import interp1d # Ensure import
        if len(t_imu_raw) == 1:
            if len(t_plot_master) > 0:
                closest_idx_to_imu_time = np.argmin(np.abs(t_plot_master - t_imu_raw[0]))
                omega_plot_aligned[closest_idx_to_imu_time] = omega_imu_raw[0]
        else:
            sort_indices_imu = np.argsort(t_imu_raw)
            t_imu_raw_sorted, omega_imu_raw_sorted = t_imu_raw[sort_indices_imu], omega_imu_raw[sort_indices_imu]
            omega_interpolator = interp1d(
                t_imu_raw_sorted, omega_imu_raw_sorted,
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            omega_plot_aligned = omega_interpolator(t_plot_master)
    else:
        print(f"No valid IMU omega signal to interpolate for {title_suffix}.")

    event_marker_size = 6
    hs_event_times, hs_event_values, to_event_times, to_event_values, ms_event_times, ms_event_values = [],[],[],[],[],[]
    if t_imu_raw is not None and omega_imu_raw is not None:
        valid_hs_indices = [i for i in hs_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        if valid_hs_indices: hs_event_times, hs_event_values = t_imu_raw[valid_hs_indices], omega_imu_raw[valid_hs_indices]
        valid_to_indices = [i for i in to_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        if valid_to_indices: to_event_times, to_event_values = t_imu_raw[valid_to_indices], omega_imu_raw[valid_to_indices]
        valid_ms_indices = [i for i in ms_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        if valid_ms_indices: ms_event_times, ms_event_values = t_imu_raw[valid_ms_indices], omega_imu_raw[valid_ms_indices]

    n_plot = min(config.PLOT_MAX_SAMPLES_INFERENCE, len(t_plot_master))
    t_plot_master_sub, omega_plot_aligned_sub, predicted_phases_csv_sub = t_plot_master[:n_plot], omega_plot_aligned[:n_plot], predicted_phases_csv[:n_plot]

    fig, ax1 = plt.subplots(figsize=(15, 6))
    sensor_id_for_plot = gal.sensors[0] if hasattr(gal, 'LEG') and gal.LEG.upper() == 'L' else gal.sensors[1] if hasattr(gal, 'sensors') else "IMU"
    omega_color = gal.colors.get(sensor_id_for_plot, 'tab:grey') if hasattr(gal, 'colors') else 'tab:grey'
    ax1.plot(t_plot_master_sub, omega_plot_aligned_sub, color=omega_color, alpha=0.7, linewidth=1.5, label=f'Ang. Vel. ({sensor_id_for_plot})')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Angular Velocity (rad/s)', color='black'); ax1.tick_params(axis='y', labelcolor='black')

    phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
    phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
    legend_phases_added = set()
    for ph_val, color_val_phase in phase_colors_map.items():
        mask = (predicted_phases_csv_sub == ph_val)
        if not np.any(mask): continue
        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts, ends = np.where(diff_mask == 1)[0], np.where(diff_mask == -1)[0]
        for seg_start, seg_end in zip(starts, ends):
            if seg_start < seg_end and seg_start < len(t_plot_master_sub):
                actual_seg_end_idx = min(seg_end, len(t_plot_master_sub))
                if actual_seg_end_idx <= seg_start: continue
                label_to_use_phase = None
                if ph_val not in legend_phases_added:
                    label_to_use_phase = phase_legend_labels_map.get(ph_val); legend_phases_added.add(ph_val)
                time_start_span, time_end_span = t_plot_master_sub[seg_start], t_plot_master_sub[actual_seg_end_idx-1] + (gal.dt/2 if hasattr(gal, 'dt') else 0.01)
                ax1.axvspan(time_start_span, time_end_span, color=color_val_phase, alpha=0.45, label=label_to_use_phase, zorder=-1)
    
    if len(ms_event_times) > 0: ax1.plot(ms_event_times, ms_event_values, 'o', color='red', markersize=event_marker_size, alpha=0.8, label='Mid-Swing')
    if len(hs_event_times) > 0: ax1.plot(hs_event_times, hs_event_values, 'o', color='magenta', markersize=event_marker_size, alpha=0.8, label='Heel Strike')
    if len(to_event_times) > 0: ax1.plot(to_event_times, to_event_values, 'o', color='cyan', markersize=event_marker_size, alpha=0.8, label='Toe Off')

    full_title = f"Model Predicted Phases vs. IMU Angular Velocity {title_suffix}"
    if trial_num is not None: full_title = f"Trial {trial_num}: " + full_title
    plt.title(full_title); ax1.grid(True, linestyle=':', alpha=0.5); ax1.legend(loc='best', fontsize='small'); plt.tight_layout()
    
    plot_filename_base = f"model_phases_vs_ang_vel{title_suffix.replace(' ', '_').lower()}"
    if trial_num is not None: plot_filename_base = f"trial{trial_num}_" + plot_filename_base
    plot_filename_base += ".png"

    global OUTPUT_PLOTS_DIR
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base))
    print(f"Saved model phases vs angular velocity plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename_base)}")
    plt.close(fig)