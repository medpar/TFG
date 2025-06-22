# generate_final_results.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def plot_df_as_table(df, title="", save_path=None, filename="table.png"):
    """
    Renders a pandas DataFrame as a table image and saves it.
    """
    fig, ax = plt.subplots(figsize=(8, 2 + len(df) * 0.5)) # Adjust size based on number of rows
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) # Adjust scale for better padding

    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0: # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('royalblue')
        if j == -1: # Index column
            cell.set_text_props(weight='bold')

    plt.title(title, fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Table image saved to {full_path}")
    
    plt.close(fig)


def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculates overall and per-class metrics.
    Returns two dictionaries: one for overall metrics and one for per-class metrics.
    """
    valid_indices = (y_pred != -1)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    # --- FIX: Ensure the correct order of metrics ---
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    
    overall_metrics = {
        'Accuracy': accuracy,
        'Weighted Precision': precision_w,
        'Weighted Recall': recall_w,
        'Weighted F1-Score': f1_w,
    }

    precision_p, recall_p, f1_p, support_p = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.arange(len(class_names)), zero_division=0
    )
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            # Note: Accuracy is not a per-class metric, so we don't include it here.
            'Precision': precision_p[i],
            'Recall': recall_p[i],
            'F1-Score': f1_p[i],
            'Support': support_p[i]
        }
        
    return overall_metrics, per_class_metrics

def process_inference_folder(folder_path):
    """
    Reads all prediction CSVs in a folder, aggregates true and predicted labels,
    and calculates metrics.
    """
    all_true_phases = []
    all_pred_phases = []
    
    prediction_files = glob.glob(os.path.join(folder_path, "*_gait_phases.csv"))
    if not prediction_files:
        print(f"Warning: No '*_gait_phases.csv' files found in {folder_path}")
        return None, None

    for f in prediction_files:
        try:
            df = pd.read_csv(f)
            if 'true_phase_aligned' in df.columns and 'predicted_phase' in df.columns:
                df.dropna(subset=['true_phase_aligned', 'predicted_phase'], inplace=True)
                all_true_phases.extend(df['true_phase_aligned'].astype(int).values)
                all_pred_phases.extend(df['predicted_phase'].astype(int).values)
        except Exception as e:
            print(f"Could not process file {f}: {e}")
            
    if not all_true_phases:
        print(f"Warning: No valid ground truth data found in any files in {folder_path}")
        return None, None

    return np.array(all_true_phases), np.array(all_pred_phases)


def create_master_performance_table(all_results, save_path=None):
    """
    Creates, prints, and saves a formatted table of overall performance metrics as CSV and image.
    """
    # --- FIX: Enforce column order ---
    metric_order = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score']
    
    df = pd.DataFrame(all_results)
    df = df.set_index('HPE Model')
    # Reorder columns according to your preference
    df = df[metric_order]
    df = df.round(4)

    print("\n--- Master Performance Table ---")
    print(df.to_markdown())

    if save_path:
        # Save as CSV
        csv_path = os.path.join(save_path, "master_performance_table.csv")
        df.to_csv(csv_path)
        print(f"\nMaster performance table saved to {csv_path}")
        
        # Save as Image
        plot_df_as_table(df, 
                         title="Master Performance Comparison", 
                         save_path=save_path, 
                         filename="master_performance_table.png")
    
    return df

def create_per_class_metrics_table(per_class_metrics, hpe_model_name, save_path=None):
    """
    Creates, prints, and saves a formatted table for per-class metrics as CSV and image.
    """
    # --- FIX: Enforce column order ---
    metric_order = ['Precision', 'Recall', 'F1-Score', 'Support']
    
    df = pd.DataFrame.from_dict(per_class_metrics, orient='index')
    # Reorder columns
    df = df[metric_order]
    df = df.round(4)
    # Ensure Support is an integer
    df['Support'] = df['Support'].astype(int)

    print(f"\n--- Per-Class Metrics for {hpe_model_name} ---")
    print(df.to_markdown())

    if save_path:
        filename_base = f"per_class_metrics_{hpe_model_name}"
        # Save as CSV
        csv_path = os.path.join(save_path, f"{filename_base}.csv")
        df.to_csv(csv_path)
        print(f"\nPer-class metrics for {hpe_model_name} saved to {csv_path}")

        # Save as Image
        plot_df_as_table(df, 
                         title=f"Per-Class Metrics: {hpe_model_name}",
                         save_path=save_path, 
                         filename=f"{filename_base}.png")
    
    return df

def plot_final_performance(master_df, save_path=None):
    """
    Creates and saves a bar chart comparing the F1-score and Accuracy for each HPE model.
    """
    master_df_reset = master_df.reset_index()
    
    n_models = len(master_df_reset)
    index = np.arange(n_models)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- FIX: Ensure the correct column names are used ---
    f1_col = 'Weighted F1-Score'
    acc_col = 'Accuracy'

    bars1 = ax.bar(index - bar_width/2, master_df_reset[f1_col], bar_width, label='Weighted F1-Score', color='royalblue')
    bars2 = ax.bar(index + bar_width/2, master_df_reset[acc_col], bar_width, label='Accuracy', color='skyblue')

    ax.set_xlabel('3D Pose Estimation Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Overall Gait Phase Prediction Performance Comparison', fontweight='bold', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(master_df_reset['HPE Model'])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    fig.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "final_performance_plot.png"), dpi=300)
        plt.savefig(os.path.join(save_path, "final_performance_plot.svg"))
        print(f"\nFinal performance plot saved to {save_path}")

    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    BASE_INFERENCE_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/model_30Hz/"
    HPE_MODELS = {
        "BodyTrack": "inference_bodytrack",
        "MMPose": "inference_mmpose",
        "MotionAGFormer": "inference_motionagformer",
        "MotionBERT": "inference_motionbert"
    }
    FINAL_RESULTS_OUTPUT_DIR = "/Users/mario/Documents/TFG_VIDIMU/VIDIMU/model_30Hz/final_results"
    os.makedirs(FINAL_RESULTS_OUTPUT_DIR, exist_ok=True)
    CLASS_NAMES = ['Stance', 'Swing', 'Turn']
    
    all_hpe_results = []
    
    # --- Main Loop ---
    for model_name, folder_name in HPE_MODELS.items():
        print(f"\n=================================================")
        print(f"Processing results for: {model_name}")
        print(f"=================================================")
        
        folder_path = os.path.join(BASE_INFERENCE_DIR, folder_name)
        
        y_true, y_pred = process_inference_folder(folder_path)
        
        if y_true is not None and y_pred is not None:
            overall_metrics, per_class_metrics = calculate_metrics(y_true, y_pred, CLASS_NAMES)
            result_entry = {'HPE Model': model_name, **overall_metrics}
            all_hpe_results.append(result_entry)
            create_per_class_metrics_table(per_class_metrics, model_name, save_path=FINAL_RESULTS_OUTPUT_DIR)
        else:
            print(f"Skipping metrics for {model_name} due to missing data.")

    # --- Generate Final Outputs ---
    if all_hpe_results:
        master_df = create_master_performance_table(all_hpe_results, save_path=FINAL_RESULTS_OUTPUT_DIR)
        plot_final_performance(master_df, save_path=FINAL_RESULTS_OUTPUT_DIR)
    else:
        print("\nNo results were processed. Cannot generate final tables or plots.")