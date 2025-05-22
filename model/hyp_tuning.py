# model/hyp_tuning.py
import os
import time
import shutil
import optuna
import torch
import numpy as np
import pandas as pd

import config # For default values and paths
import data_loader
from model import GaitLSTM
import train
import utils

# --- Optuna Study Configuration ---
N_TRIALS = 50  # Number of hyperparameter combinations to try
N_JOBS = 1     # Keep at 1 for MPS stability
STUDY_NAME = "hyp_tuning" # Updated study name
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db" # SQLite DB for study persistence

# --- Output Directory for All Tuning Runs ---
# This uses the OUTPUT_DIR from the main config.py as the base for tuning runs
TUNING_OUTPUT_BASE_DIR = os.path.join(config.OUTPUT_DIR, "optuna_tuning_runs", STUDY_NAME)
os.makedirs(TUNING_OUTPUT_BASE_DIR, exist_ok=True)

# --- Log file for hyperparameter results ---
HYPERPARAM_LOG_FILE = os.path.join(TUNING_OUTPUT_BASE_DIR, "hyperparameter_tuning_log.csv")

# Define the expected columns for the CSV log file
# This helps ensure consistency when appending new trial data.
EXPECTED_LOG_COLUMNS = [
    "trial_number", "value_to_optimize (target_metric)", "target_metric_name",
    "status_in_objective", "datetime_start", "duration_seconds_in_objective",
    # Tunable Hyperparameters (these must match the names used in trial.suggest_... calls)
    "lstm_hidden_size", "num_lstm_layers", "lstm_dropout",
    "linear_dropout", "weight_decay", "learning_rate",
    "sequence_length", "early_stopping_patience",
    # Fixed (non-tuned) parameters for this study, good to log for record-keeping
    "fixed_use_weighted_loss", "fixed_optimize_metric", "fixed_optimize_metric_for_optuna",
    # Additional Metrics from the best epoch of the trial
    "best_epoch_num",
    "best_epoch_train_loss", "best_epoch_train_acc",
    "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1"
]
# Initialize log file with headers if it doesn't exist
if not os.path.exists(HYPERPARAM_LOG_FILE):
    pd.DataFrame(columns=EXPECTED_LOG_COLUMNS).to_csv(HYPERPARAM_LOG_FILE, index=False)


def objective(trial: optuna.trial.Trial):
    objective_start_time = time.time()
    # --- 1. Suggest Hyperparameters (using your current ranges) ---
    suggested_lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 96, 128])
    suggested_num_lstm_layers = trial.suggest_categorical("num_lstm_layers", [1, 2]) # Allowing 1 or 2 layers
    
    if suggested_num_lstm_layers > 1:
        suggested_lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
    else:
        suggested_lstm_dropout = 0.0 # No LSTM dropout if only 1 layer

    suggested_linear_dropout = trial.suggest_float("linear_dropout", 0.3, 0.6, step=0.1)
    suggested_weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    suggested_learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 2e-3) # Adjusted from 5e-5, 5e-3
    suggested_sequence_length = trial.suggest_categorical("sequence_length", [75, 100, 125])
    suggested_early_stopping_patience = trial.suggest_int("early_stopping_patience", 10, 20)
    
    # --- Parameters Fixed for this Study (as per your request) ---
    fixed_use_weighted_loss = True
    # The following OPTIMIZE_METRIC flags will be set directly from this script's logic
    # ensuring this Optuna study optimizes for F1.
    fixed_optimize_metric = 'f1'
    fixed_optimize_metric_for_optuna = 'f1'


    # --- 2. Update Global Config Dynamically for this Trial ---
    original_config_values = {}
    def update_config_value(attr_name, new_value):
        if hasattr(config, attr_name):
            original_config_values[attr_name] = getattr(config, attr_name)
        else:
            original_config_values[attr_name] = "__NEWLY_ADDED_BY_OPTUNA__" # Sentinel for new attributes
        setattr(config, attr_name, new_value)

    # Apply suggested hyperparameters
    update_config_value("LSTM_HIDDEN_SIZE", suggested_lstm_hidden_size)
    update_config_value("NUM_LSTM_LAYERS", suggested_num_lstm_layers)
    update_config_value("LSTM_DROPOUT", suggested_lstm_dropout)
    update_config_value("LINEAR_DROPOUT", suggested_linear_dropout)
    update_config_value("WEIGHT_DECAY", suggested_weight_decay)
    update_config_value("LEARNING_RATE", suggested_learning_rate)
    update_config_value("SEQUENCE_LENGTH", suggested_sequence_length)
    update_config_value("EARLY_STOPPING_PATIENCE", suggested_early_stopping_patience)
    
    # Apply fixed/tuning-specific parameters
    update_config_value("NUM_EPOCHS", 75) # Max epochs for each tuning trial
    update_config_value("USE_WEIGHTED_LOSS", fixed_use_weighted_loss)
    update_config_value("OPTIMIZE_METRIC", fixed_optimize_metric)
    update_config_value("OPTIMIZE_METRIC_FOR_OPTUNA", fixed_optimize_metric_for_optuna)


    # --- 3. Create Unique Output Directory for this Trial's Artifacts ---
    trial_output_dir_name = f"trial_{trial.number}_{int(objective_start_time)}"
    trial_specific_output_dir = os.path.join(TUNING_OUTPUT_BASE_DIR, trial_output_dir_name)
    os.makedirs(trial_specific_output_dir, exist_ok=True)
    update_config_value("OUTPUT_DIR", trial_specific_output_dir) # Redirect train.py's output
    
    # Ensure utils.py saves plots to this trial's specific plot folder
    utils.OUTPUT_PLOTS_DIR = os.path.join(trial_specific_output_dir, "output_plots")
    os.makedirs(utils.OUTPUT_PLOTS_DIR, exist_ok=True)

    print(f"\n--- Optuna Trial {trial.number} ---")
    current_trial_params_for_print = trial.params.copy()
    current_trial_params_for_print["fixed_use_weighted_loss"] = fixed_use_weighted_loss
    current_trial_params_for_print["optimizing_metric_for_study"] = fixed_optimize_metric
    print(f"Hyperparameters: {current_trial_params_for_print}")
    print(f"Output Directory for this trial: {config.OUTPUT_DIR}")

    # Device setup
    if torch.backends.mps.is_available(): device_to_use = "mps"
    elif torch.cuda.is_available(): device_to_use = "cuda"
    else: device_to_use = "cpu"
    update_config_value("DEVICE", device_to_use)
    print(f"Using device: {config.DEVICE}")

    # --- 4. Run Training Loop ---
    fold_to_run_for_tuning = 0 # Using fold 0 for hyperparameter evaluation
    optuna_metric_value_to_return = 0.0 # Default for F1 (lower is worse) if pruning/error
    final_eval_metrics_for_log = {} 
    status_in_objective = "INITIATED"

    try:
        # train.main_train_loop should return: model, scaler, metric_for_optuna, all_final_metrics_dict
        if config.K_FOLDS > 1: # Ensure config.K_FOLDS is appropriate for your setup
            _, _, optuna_metric_value_to_return, final_eval_metrics_for_log = train.main_train_loop(
                fold_num=fold_to_run_for_tuning,
                optuna_trial=trial # Pass the Optuna trial object for pruning
            )
        else: # Single split scenario
            _, _, optuna_metric_value_to_return, final_eval_metrics_for_log = train.main_train_loop(
                optuna_trial=trial
            )
        status_in_objective = "COMPLETED_TRAINING_LOOP"
        print(f"Trial {trial.number} training loop finished. Optuna metric (Val F1): {optuna_metric_value_to_return:.4f}")

    except optuna.exceptions.TrialPruned:
        status_in_objective = "PRUNED_BY_OPTUNA"
        print(f"Trial {trial.number} was pruned by Optuna.")
        # optuna_metric_value_to_return remains 0.0 (bad F1)
        raise # Re-raise to let Optuna handle it and record state correctly

    except Exception as e:
        status_in_objective = f"ERROR_IN_TRAINING:_{type(e).__name__}"
        print(f"Unhandled error during Optuna trial {trial.number} training: {e}")
        import traceback
        traceback.print_exc()
        # optuna_metric_value_to_return remains 0.0

    finally:
        # Restore original config values to prevent interference between trials
        for attr_name, original_value in original_config_values.items():
            if original_value == "__NEWLY_ADDED_BY_OPTUNA__":
                if hasattr(config, attr_name): delattr(config, attr_name)
            elif hasattr(config, attr_name): # Check if it still exists before trying to set
                setattr(config, attr_name, original_value)
        
        # Restore default plot output directory for utils module
        # This uses the original OUTPUT_DIR that was set before this objective function ran
        original_main_output_dir = original_config_values.get("OUTPUT_DIR", "/tmp/model_outputs_default") # Fallback
        utils.OUTPUT_PLOTS_DIR = os.path.join(original_main_output_dir, "output_plots_default")

    objective_end_time = time.time()
    duration_in_objective = objective_end_time - objective_start_time
    
    # --- Log results to CSV ---
    log_entry_data = {
        "trial_number": trial.number,
        "value_to_optimize (target_metric)": optuna_metric_value_to_return if status_in_objective == "COMPLETED_TRAINING_LOOP" else 0.0,
        "target_metric_name": fixed_optimize_metric_for_optuna,
        "status_in_objective": status_in_objective,
        "datetime_start": trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if trial.datetime_start else "N/A",
        "duration_seconds_in_objective": duration_in_objective,
        "fixed_use_weighted_loss": fixed_use_weighted_loss,
        "fixed_optimize_metric": fixed_optimize_metric,
        "fixed_optimize_metric_for_optuna": fixed_optimize_metric_for_optuna
    }
    # Add all Optuna-suggested hyperparameters
    for key, value in trial.params.items():
        log_entry_data[key] = value
    
    # Add final evaluation metrics if the trial completed its training loop
    if status_in_objective == "COMPLETED_TRAINING_LOOP" and final_eval_metrics_for_log:
        for metric_key_suffix in ["num", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]:
            log_entry_data[f"best_epoch_{metric_key_suffix}"] = final_eval_metrics_for_log.get(f"best_epoch_{metric_key_suffix}", np.nan)
    else: # Fill with NaNs if not completed or metrics not available
        for metric_key_suffix in ["num", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]:
            log_entry_data[f"best_epoch_{metric_key_suffix}"] = np.nan
            
    # Create a DataFrame for the current entry, ensuring columns match the log file
    current_log_df = pd.DataFrame([log_entry_data])
    # Reorder/add missing columns to match EXPECTED_LOG_COLUMNS for robust appending
    df_to_append = pd.DataFrame(columns=EXPECTED_LOG_COLUMNS)
    for col in EXPECTED_LOG_COLUMNS:
        df_to_append.loc[0, col] = current_log_df.get(col, pd.Series(dtype='object')).iloc[0] if col in current_log_df else np.nan

    df_to_append.to_csv(HYPERPARAM_LOG_FILE, mode='a', header=False, index=False)

    # Return the F1 score (Optuna will maximize this)
    return optuna_metric_value_to_return


if __name__ == "__main__":
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # --- Ensure config reflects the fixed settings for THIS Optuna study ---
    # These will be used by the objective function if it needs to read them before overriding.
    # The objective function itself will also set these for each trial.
    _original_use_weighted_loss = config.USE_WEIGHTED_LOSS
    _original_optimize_metric = config.OPTIMIZE_METRIC
    _original_optimize_metric_for_optuna = config.OPTIMIZE_METRIC_FOR_OPTUNA

    setattr(config, "USE_WEIGHTED_LOSS", True)
    setattr(config, "OPTIMIZE_METRIC", 'f1')
    setattr(config, "OPTIMIZE_METRIC_FOR_OPTUNA", 'f1')
    print(f"Optuna study will use fixed USE_WEIGHTED_LOSS: {config.USE_WEIGHTED_LOSS}")
    print(f"Optuna study will optimize for metric: {config.OPTIMIZE_METRIC}")


    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=7) # More startup trials for TPE
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1) # Adjusted warmup
    
    study = optuna.create_study(
        study_name=STUDY_NAME, 
        storage=STORAGE_NAME, 
        direction="maximize",  # IMPORTANT: Set to maximize F1-score
        sampler=sampler, 
        pruner=pruner, 
        load_if_exists=True
    )

    print(f"Optuna study database: {STORAGE_NAME}")
    print(f"Study direction is set to: {study.direction.name}")
    finished_trials_count = len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]])
    print(f"Number of trials already processed in study: {finished_trials_count}")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=None)
    except KeyboardInterrupt: 
        print("Optuna optimization stopped by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"An unhandled exception occurred during study.optimize: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original config values that might have been globally set for the study
        setattr(config, "USE_WEIGHTED_LOSS", _original_use_weighted_loss)
        setattr(config, "OPTIMIZE_METRIC", _original_optimize_metric)
        setattr(config, "OPTIMIZE_METRIC_FOR_OPTUNA", _original_optimize_metric_for_optuna)
        print("Restored original config settings for USE_WEIGHTED_LOSS and OPTIMIZE_METRIC.")

    
    print("\n--- Optuna Hyperparameter Tuning Finished ---")
    # ... (rest of the summary printing, ensure it correctly refers to F1-score as the optimized value)
    all_study_trials = study.trials
    completed_trials = [t for t in all_study_trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in all_study_trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in all_study_trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"Total trials in study: {len(all_study_trials)}")
    print(f"  Successfully completed: {len(completed_trials)}")
    print(f"  Pruned: {len(pruned_trials)}")
    print(f"  Failed: {len(failed_trials)}")

    if completed_trials:
        try:
            best_trial_overall = study.best_trial # Optuna considers direction when giving best_trial
            print("\nBest trial (overall completed):")
            print(f"  Trial Number: {best_trial_overall.number}")
            print(f"  Value (Validation F1-score): {best_trial_overall.value:.4f}") 
            print("  Params (tuned by Optuna): ")
            for key, value in best_trial_overall.params.items():
                print(f"    {key}: {value}")
            # Confirm fixed parameters for this study
            print(f"    fixed_use_weighted_loss: True")
            print(f"    fixed_optimize_metric (for study): f1")
            
            best_params_file = os.path.join(TUNING_OUTPUT_BASE_DIR, "best_hyperparameters.txt")
            with open(best_params_file, "w") as f:
                f.write(f"Optuna Study Name: {STUDY_NAME}\n")
                f.write(f"Best trial number: {best_trial_overall.number}\n")
                f.write(f"Best validation F1-score (Optuna objective): {best_trial_overall.value:.4f}\n")
                f.write("Best hyperparameters (tuned by Optuna):\n")
                for key, value in best_trial_overall.params.items(): f.write(f"  {key}: {value}\n")
                f.write(f"  fixed_use_weighted_loss: True\n")
                f.write(f"  fixed_optimize_metric (for study): f1\n")
                
                df_log = pd.read_csv(HYPERPARAM_LOG_FILE)
                # Ensure trial_number is treated as int for matching if it was read as float from CSV
                df_log["trial_number"] = df_log["trial_number"].astype(int) 
                best_trial_log_entry = df_log[df_log["trial_number"] == best_trial_overall.number]

                if not best_trial_log_entry.empty:
                    f.write("\nDetailed Metrics for Best Trial (from CSV log):\n")
                    metrics_to_print_from_log = [
                        "best_epoch_num", "best_epoch_train_loss", "best_epoch_train_acc", 
                        "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1"
                    ]
                    for col in metrics_to_print_from_log:
                        if col in best_trial_log_entry.columns and pd.notna(best_trial_log_entry[col].iloc[0]):
                             f.write(f"  {col}: {best_trial_log_entry[col].iloc[0]}\n")
                
                best_model_fold_suffix = f"_fold{0}" if config.K_FOLDS > 1 else "" # Assuming fold 0
                trial_prefix_for_best = f"trial{best_trial_overall.number}_"
                best_model_filename_for_best_trial = f"{trial_prefix_for_best}best_model{best_model_fold_suffix}.pth"
                f.write(f"Best model file (relative to its trial folder under {TUNING_OUTPUT_BASE_DIR}): {os.path.join(f'trial_{best_trial_overall.number}_*', best_model_filename_for_best_trial)}\n") # Path hint
            print(f"Best hyperparameters and metrics saved to {best_params_file}")

        except ValueError: 
            print("No trials completed successfully, cannot determine the best trial from study.best_trial.")
        except Exception as e_summary:
            print(f"Error during best trial summary: {e_summary}")
    else: 
        print("No trials completed successfully. Cannot determine best trial.")

    print(f"\nAll trial attempts logged in: {HYPERPARAM_LOG_FILE}")
    print(f"Individual trial outputs are in subfolders under: {TUNING_OUTPUT_BASE_DIR}")