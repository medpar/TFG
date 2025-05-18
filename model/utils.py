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

# Add the gaitseg directory (sibling to model) to sys.path for importing gaitseg_utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GAITSEG_DIR = os.path.join(PARENT_DIR, "gaitseg")
OUTPUT_PLOTS_DIR = os.path.join(config.OUTPUT_DIR, "output_plots")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

if GAITSEG_DIR not in sys.path:
    sys.path.append(GAITSEG_DIR)

try:
    import gaitseg_utils as gal
except ImportError:
    print("Warning: gaitseg_utils.py not found in gaitseg directory. Some plotting functions in utils.py might not work.")
    gal = None


def plot_training_history(history, fold_num=None):
    """Plots training and validation loss and accuracy."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    title_prefix = f"Fold {fold_num} " if fold_num is not None else ""

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
    plot_filename = f"{title_prefix}training_history.png".replace(" ", "_").lower()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename))
    print(f"Saved training history plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename)}")
    plt.close(fig)


def calculate_metrics(y_true_flat, y_pred_flat, average='weighted'):
    """Calculates and returns classification metrics."""
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

def plot_confusion_matrix_custom(y_true_flat, y_pred_flat, class_names, title='Confusion Matrix', fold_num=None):
    """Plots the confusion matrix."""
    if not y_true_flat or not y_pred_flat:
        print(f"Skipping confusion matrix for '{title}': No data.")
        return
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    title_prefix = f"Fold {fold_num} " if fold_num is not None else ""
    ax.set_title(f'{title_prefix}{title}')
    
    plot_filename = f"{title_prefix}{title.replace(' ', '_').lower()}_confusion_matrix.png"
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename))
    print(f"Saved confusion matrix to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename)}")
    plt.close(fig)

def save_checkpoint(state, is_best, filename="checkpoint.pth", best_filename="best_model.pth", output_dir=config.OUTPUT_DIR):
    """Saves model checkpoint."""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, best_filename)
        torch.save(state, best_filepath)
        print(f"Saved new best model to {best_filepath}")

def load_checkpoint(filepath, model, optimizer=None, device=config.DEVICE):
    """Loads model checkpoint."""
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at {filepath}")
        return None, None # Return epoch and best_val_metric as None
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print(f"Loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 0)}, Best Val Metric: {checkpoint.get('best_val_metric', float('inf')):.4f})")
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_metric', float('inf'))


def plot_model_predictions_vs_true_phases(timestamps, true_phases, predicted_phases, title_suffix=""):
    """
    Plots true vs. predicted phases over time.
    """
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
    
    ax.set_yticks(np.arange(-1, config.NUM_CLASSES +1)) # Show unclassified (-1) if present
    ax.set_yticklabels([f"Phase {i}" for i in np.arange(-1, config.NUM_CLASSES +1)])
    ax.legend()
    ax.grid(True, linestyle=':')
    
    plt.title(f"Model Predicted Gait Phases vs True Phases {title_suffix}")
    plt.tight_layout()
    
    plot_filename = f"model_vs_true_phases{title_suffix.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename))
    print(f"Saved model vs true phases plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename)}")
    plt.close(fig)


def plot_model_predictions_vs_angular_velocity(
    timestamps_csv, predicted_phases_csv, raw_imu_filepath, title_suffix=""
):
    """
    Plots estimated gait phases (from model output CSV) vs. angular velocity from the raw IMU file.
    """
    if gal is None:
        print("gaitseg_utils not available. Skipping plot_model_predictions_vs_angular_velocity.")
        return
        
    if len(timestamps_csv) == 0:
        print(f"No CSV data to plot for angular velocity comparison ({title_suffix}).")
        return

    # 1. Get Angular Velocity and Events from .raw file
    # Ensure global LEG in gal is set appropriately if needed, or pass it to compute_velocity_and_events
    # For simplicity, assuming gal.LEG is 'l' or 'r' as configured in gaitseg_utils.py
    print(f"Processing .raw file for angular velocity: {raw_imu_filepath} (using gal.LEG: {gal.LEG if hasattr(gal, 'LEG') else 'N/A'})")
    results_from_raw = gal.compute_velocity_and_events(raw_imu_filepath)

    if results_from_raw is None or results_from_raw[0] is None:
        print(f"Could not process .raw file {raw_imu_filepath}. Skipping ang. vel. plot.")
        t_imu_raw, omega_imu_raw, ms_indices_imu, hs_indices_imu, to_indices_imu = None, None, [], [], []
    else:
        t_imu_raw, _, omega_imu_raw, \
        ms_indices_imu, _, hs_indices_imu, to_indices_imu, _ = results_from_raw

    # 2. Prepare data for plotting
    t_plot_master = timestamps_csv  # Master time axis for the plot from prediction CSV

    # Interpolate omega_imu_raw onto t_plot_master for aligned plotting
    omega_plot_aligned = np.full_like(t_plot_master, np.nan, dtype=float)
    if t_imu_raw is not None and omega_imu_raw is not None and len(t_imu_raw) > 0:
        if len(t_imu_raw) == 1:
            if len(t_plot_master) > 0:
                closest_idx_to_imu_time = np.argmin(np.abs(t_plot_master - t_imu_raw[0]))
                omega_plot_aligned[closest_idx_to_imu_time] = omega_imu_raw[0]
        else:
            sort_indices_imu = np.argsort(t_imu_raw)
            t_imu_raw_sorted = t_imu_raw[sort_indices_imu]
            omega_imu_raw_sorted = omega_imu_raw[sort_indices_imu]
            
            omega_interpolator = torch.from_numpy # Using interp1d from scipy not torch
            from scipy.interpolate import interp1d
            omega_interpolator = interp1d(
                t_imu_raw_sorted, omega_imu_raw_sorted,
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            omega_plot_aligned = omega_interpolator(t_plot_master)
    else:
        print(f"No valid IMU omega signal to interpolate for {title_suffix}.")

    # Event times and values (from original IMU processing)
    event_marker_size = 6
    hs_event_times, hs_event_values = [], []
    to_event_times, to_event_values = [], []
    ms_event_times, ms_event_values = [], []

    if t_imu_raw is not None and omega_imu_raw is not None:
        valid_hs_indices = [i for i in hs_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        hs_event_times = t_imu_raw[valid_hs_indices]
        hs_event_values = omega_imu_raw[valid_hs_indices]

        valid_to_indices = [i for i in to_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        to_event_times = t_imu_raw[valid_to_indices]
        to_event_values = omega_imu_raw[valid_to_indices]

        valid_ms_indices = [i for i in ms_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        ms_event_times = t_imu_raw[valid_ms_indices]
        ms_event_values = omega_imu_raw[valid_ms_indices]

    # 3. Plotting
    n_plot = min(config.PLOT_MAX_SAMPLES_INFERENCE, len(t_plot_master))
    t_plot_master_sub = t_plot_master[:n_plot]
    omega_plot_aligned_sub = omega_plot_aligned[:n_plot]
    predicted_phases_csv_sub = predicted_phases_csv[:n_plot]


    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    sensor_id_for_plot = gal.sensors[0] if hasattr(gal, 'LEG') and gal.LEG.upper() == 'L' else gal.sensors[1] if hasattr(gal, 'sensors') else "IMU"
    omega_color = gal.colors.get(sensor_id_for_plot, 'tab:grey') if hasattr(gal, 'colors') else 'tab:grey'

    ax1.plot(t_plot_master_sub, omega_plot_aligned_sub, color=omega_color, alpha=0.7, linewidth=1.5, label=f'Ang. Vel. ({sensor_id_for_plot})')
    ax1.set_xlabel('Time (s) - from Prediction CSV')
    ax1.set_ylabel('Angular Velocity (rad/s)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Plot phase regions from model predictions
    phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'} # Match GUI
    phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
    legend_phases_added = set()

    for ph_val, color_val_phase in phase_colors_map.items():
        mask = (predicted_phases_csv_sub == ph_val)
        if not np.any(mask): continue

        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]

        for seg_start, seg_end in zip(starts, ends):
            if seg_start < seg_end and seg_start < len(t_plot_master_sub):
                actual_seg_end_idx = min(seg_end, len(t_plot_master_sub))
                if actual_seg_end_idx <= seg_start: continue
                
                label_to_use_phase = None
                if ph_val not in legend_phases_added:
                    label_to_use_phase = phase_legend_labels_map.get(ph_val) + " (Model)"
                    legend_phases_added.add(ph_val)
                
                time_start_span = t_plot_master_sub[seg_start]
                time_end_span = t_plot_master_sub[actual_seg_end_idx-1] + (gal.dt/2 if hasattr(gal, 'dt') else 0.01)
                
                ax1.axvspan(time_start_span, time_end_span,
                            color=color_val_phase, alpha=0.45, label=label_to_use_phase, zorder=-1)
    
    # Plot event markers (from raw IMU processing)
    if len(ms_event_times) > 0:
        ax1.plot(ms_event_times, ms_event_values, 'o', color='red', markersize=event_marker_size, alpha=0.8, label='Mid-Swing (raw IMU)')
    if len(hs_event_times) > 0:
        ax1.plot(hs_event_times, hs_event_values, 'o', color='magenta', markersize=event_marker_size, alpha=0.8, label='Heel Strike (raw IMU)')
    if len(to_event_times) > 0:
        ax1.plot(to_event_times, to_event_values, 'o', color='cyan', markersize=event_marker_size, alpha=0.8, label='Toe Off (raw IMU)')

    plt.title(f"Model Predicted Phases vs. IMU Angular Velocity {title_suffix}")
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    plot_filename = f"model_phases_vs_ang_vel{title_suffix.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, plot_filename))
    print(f"Saved model phases vs angular velocity plot to {os.path.join(OUTPUT_PLOTS_DIR, plot_filename)}")
    plt.close(fig)


# Old plot_predictions_vs_actual - can be removed or kept if a simpler feature plot is desired
# def plot_predictions_vs_actual(timestamps, features, true_phases, predicted_phases, num_samples_to_plot=config.PLOT_MAX_SAMPLES_INFERENCE, title="Gait Phase Prediction"):
#     """
#     Plots true vs. predicted phases along with one of the input features.
#     Assumes timestamps, features, true_phases, predicted_phases are 1D numpy arrays or lists.
#     """
#     if len(timestamps) == 0:
#         print("No data to plot for predictions vs actual.")
#         return

#     n = min(num_samples_to_plot, len(timestamps))
    
#     time_plot = timestamps[:n]
#     feature_to_plot = features[:n, 0] if features.ndim > 1 else features[:n] 
#     true_phases_plot = true_phases[:n] if true_phases is not None and len(true_phases) > 0 else None
#     predicted_phases_plot = predicted_phases[:n]

#     fig, ax1 = plt.subplots(figsize=(15, 6))

#     color = 'tab:grey'
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel(config.FEATURE_COLUMNS[0], color=color)
#     ax1.plot(time_plot, feature_to_plot, color=color, alpha=0.6, label=config.FEATURE_COLUMNS[0])
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()
#     color_true = 'tab:blue'
#     color_pred = 'tab:red'
#     # ax2.set_ylabel('Phase', color=color_true)
    
#     if true_phases_plot is not None:
#         ax2.plot(time_plot, true_phases_plot, color=color_true, linestyle='-', marker='.', markersize=4, alpha=0.7, label='True Phases')
    
#     ax2.plot(time_plot, predicted_phases_plot, color=color_pred, linestyle='--', marker='x', markersize=4, alpha=0.7, label='Predicted Phases')
    
#     ax2.tick_params(axis='y') # Use default color for ticks if both true/pred are shown
#     ax2.set_yticks(np.arange(config.NUM_CLASSES)) 
#     ax2.set_yticklabels([f"Phase {i}" for i in range(config.NUM_CLASSES)])
#     ax2.set_ylabel('Phase Label')


#     fig.tight_layout() 
#     plt.title(title)
    
#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
#     plot_filename = f"{title.replace(' ', '_').lower()}_prediction_vs_feature.png"
#     plt.savefig(os.path.join(config.OUTPUT_DIR, plot_filename))
#     print(f"Saved prediction vs feature plot to {os.path.join(config.OUTPUT_DIR, plot_filename)}")
#     plt.close(fig)


if __name__ == '__main__':
    print("Testing utils...")
    # Dummy history for plot_training_history
    history = {
        'train_loss': np.random.rand(10) * 2, 'val_loss': np.random.rand(10) * 2 + 0.1,
        'train_acc': np.random.rand(10) * 0.3 + 0.5, 'val_acc': np.random.rand(10) * 0.3 + 0.45
    }
    plot_training_history(history, fold_num=0)

    # Dummy data for confusion matrix and metrics
    y_true = np.random.randint(0, config.NUM_CLASSES, 100)
    y_pred = np.random.randint(0, config.NUM_CLASSES, 100)
    class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
    
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Calculated metrics: {metrics}")
    plot_confusion_matrix_custom(y_true, y_pred, class_names, title="Dummy CM", fold_num=0)

    # Dummy data for plot_model_predictions_vs_true_phases
    timestamps_dummy = np.linspace(0, 10, 100)
    true_phases_dummy = np.random.randint(0, config.NUM_CLASSES, 100)
    pred_phases_dummy = np.random.randint(0, config.NUM_CLASSES, 100)
    plot_model_predictions_vs_true_phases(timestamps_dummy, true_phases_dummy, pred_phases_dummy, title_suffix="Dummy")

    # For plot_model_predictions_vs_angular_velocity, you'd need a dummy .raw file
    # and ensure gaitseg_utils is correctly imported.
    print("Utils testing complete. Angular velocity plot requires data and gaitseg_utils.")