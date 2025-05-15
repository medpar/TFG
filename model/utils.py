# utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
import config
import os

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
    plt.savefig(os.path.join(config.OUTPUT_DIR, plot_filename))
    print(f"Saved training history plot to {os.path.join(config.OUTPUT_DIR, plot_filename)}")
    #plt.show()


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
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    title_prefix = f"Fold {fold_num} " if fold_num is not None else ""
    ax.set_title(f'{title_prefix}{title}')
    
    plot_filename = f"{title_prefix}confusion_matrix.png".replace(" ", "_").lower()
    plt.savefig(os.path.join(config.OUTPUT_DIR, plot_filename))
    print(f"Saved confusion matrix to {os.path.join(config.OUTPUT_DIR, plot_filename)}")
    #plt.show()

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
        return None
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    print(f"Loaded checkpoint from {filepath}")
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_metric', float('inf'))


def plot_predictions_vs_actual(timestamps, features, true_phases, predicted_phases, num_samples_to_plot=config.PLOT_MAX_SAMPLES_INFERENCE, title="Gait Phase Prediction"):
    """
    Plots true vs. predicted phases along with one of the input features.
    Assumes timestamps, features, true_phases, predicted_phases are 1D numpy arrays or lists.
    """
    if len(timestamps) == 0:
        print("No data to plot for predictions vs actual.")
        return

    n = min(num_samples_to_plot, len(timestamps))
    
    time_plot = timestamps[:n]
    feature_to_plot = features[:n, 0] # Plot the first feature (e.g., knee_angle_l)
    true_phases_plot = true_phases[:n]
    predicted_phases_plot = predicted_phases[:n]

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Plot feature on primary y-axis
    color = 'tab:grey'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(config.FEATURE_COLUMNS[0], color=color)
    ax1.plot(time_plot, feature_to_plot, color=color, alpha=0.6, label=config.FEATURE_COLUMNS[0])
    ax1.tick_params(axis='y', labelcolor=color)

    # Create secondary y-axis for phases
    ax2 = ax1.twinx()
    color_true = 'tab:blue'
    color_pred = 'tab:red'
    ax2.set_ylabel('Phase', color=color_true) # We'll use blue for true phases
    
    # Plot true phases as steps or distinct markers
    ax2.plot(time_plot, true_phases_plot, color=color_true, linestyle='-', marker='.', markersize=4, alpha=0.7, label='True Phases')
    # Plot predicted phases
    ax2.plot(time_plot, predicted_phases_plot, color=color_pred, linestyle='--', marker='x', markersize=4, alpha=0.7, label='Predicted Phases')
    
    ax2.tick_params(axis='y', labelcolor=color_true)
    ax2.set_yticks(np.arange(config.NUM_CLASSES)) # Assuming phases are 0, 1, 2
    ax2.set_yticklabels([f"Phase {i}" for i in range(config.NUM_CLASSES)])

    fig.tight_layout() # Otherwise the right y-label is slightly clipped
    plt.title(title)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plot_filename = f"prediction_vs_actual.png"
    plt.savefig(os.path.join(config.OUTPUT_DIR, plot_filename))
    print(f"Saved prediction vs actual plot to {os.path.join(config.OUTPUT_DIR, plot_filename)}")
    #plt.show()


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
    plot_confusion_matrix_custom(y_true, y_pred, class_names, fold_num=0)

    # Dummy data for plot_predictions_vs_actual
    timestamps_dummy = np.linspace(0, 10, 100)
    features_dummy = np.random.rand(100, config.NUM_FEATURES)
    plot_predictions_vs_actual(timestamps_dummy, features_dummy, y_true, y_pred)