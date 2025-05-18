# model/hyperparameter_tuning.py
import os
import time
import shutil
import optuna
import torch
import numpy as np
import pandas as pd

import config
import data_loader
from model import GaitLSTM
import train
import utils

N_TRIALS = 5
N_JOBS = 1
STUDY_NAME = "hyp_tuning" # Changed study name for new log format
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"
TUNING_OUTPUT_BASE_DIR = os.path.join(config.OUTPUT_DIR, "hyp_tuning")
os.makedirs(TUNING_OUTPUT_BASE_DIR, exist_ok=True)
HYPERPARAM_LOG_FILE = os.path.join(TUNING_OUTPUT_BASE_DIR, "hyperparameter_log.csv")

# Initialize log file with new headers if it doesn't exist
EXPECTED_LOG_COLUMNS = [
    "trial_number", "value_to_optimize (best_val_loss)", "status_in_objective",
    "datetime_start", "duration_seconds_in_objective",
    # Tunable Hyperparameters (must match trial.params keys + any others you add)
    "lstm_hidden_size", "num_lstm_layers", "lstm_dropout",
    "linear_dropout", "weight_decay", "learning_rate",
    "sequence_length", "early_stopping_patience",
    # Additional Metrics from best epoch of the trial
    "best_epoch_num",
    "best_epoch_train_loss", "best_epoch_train_acc",
    "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1"
]
if not os.path.exists(HYPERPARAM_LOG_FILE):
    header_df = pd.DataFrame(columns=EXPECTED_LOG_COLUMNS)
    header_df.to_csv(HYPERPARAM_LOG_FILE, index=False)


def objective(trial: optuna.trial.Trial):
    objective_start_time = time.time()
    # --- 1. Suggest Hyperparameters ---
    suggested_lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 96, 128])
    suggested_num_lstm_layers = trial.suggest_categorical("num_lstm_layers", [1, 2])
    suggested_lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1) if suggested_num_lstm_layers > 1 else 0.0
    suggested_linear_dropout = trial.suggest_float("linear_dropout", 0.2, 0.6, step=0.1)
    suggested_weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    suggested_learning_rate = trial.suggest_loguniform("learning_rate", 5e-5, 5e-3)
    suggested_sequence_length = trial.suggest_categorical("sequence_length", [75, 100, 125])
    suggested_early_stopping_patience = trial.suggest_int("early_stopping_patience", 7, 12) # Adjusted range

    # --- 2. Update Global Config ---
    original_config_values = {}
    def update_config_value(attr_name, new_value):
        if hasattr(config, attr_name):
            original_config_values[attr_name] = getattr(config, attr_name)
        else:
            original_config_values[attr_name] = "__NEWLY_ADDED_BY_OPTUNA__"
        setattr(config, attr_name, new_value)

    update_config_value("LSTM_HIDDEN_SIZE", suggested_lstm_hidden_size)
    update_config_value("NUM_LSTM_LAYERS", suggested_num_lstm_layers)
    update_config_value("LSTM_DROPOUT", suggested_lstm_dropout)
    update_config_value("LINEAR_DROPOUT", suggested_linear_dropout)
    update_config_value("WEIGHT_DECAY", suggested_weight_decay)
    update_config_value("LEARNING_RATE", suggested_learning_rate)
    update_config_value("SEQUENCE_LENGTH", suggested_sequence_length)
    update_config_value("EARLY_STOPPING_PATIENCE", suggested_early_stopping_patience)
    update_config_value("NUM_EPOCHS", 50) # Max epochs for tuning trials; early stopping is key

    # --- 3. Create Unique Output Directory for this Trial ---
    trial_output_dir_name = f"trial_{trial.number}_{int(objective_start_time)}"
    trial_output_dir = os.path.join(TUNING_OUTPUT_BASE_DIR, trial_output_dir_name)
    os.makedirs(trial_output_dir, exist_ok=True)
    update_config_value("OUTPUT_DIR", trial_output_dir)
    utils.OUTPUT_PLOTS_DIR = os.path.join(trial_output_dir, "output_plots")
    os.makedirs(utils.OUTPUT_PLOTS_DIR, exist_ok=True)

    print(f"\n--- Optuna Trial {trial.number} ---")
    print(f"Hyperparameters: {trial.params}")
    print(f"Output Directory: {config.OUTPUT_DIR}")

    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    update_config_value("DEVICE", device)
    print(f"Using device: {config.DEVICE}")

    # --- 4. Run Training Loop ---
    fold_to_run_for_tuning = 0
    optuna_return_metric = float('inf') # This is what Optuna will minimize (best val_loss from training)
    final_eval_metrics_for_log = {} # To store the dict returned by main_train_loop
    status_in_objective = "INITIATED"

    try:
        if config.K_FOLDS > 1:
            _, _, optuna_return_metric, final_eval_metrics_for_log = train.main_train_loop(
                fold_num=fold_to_run_for_tuning,
                optuna_trial=trial
            )
        else:
            _, _, optuna_return_metric, final_eval_metrics_for_log = train.main_train_loop(
                optuna_trial=trial
            )
        
        status_in_objective = "COMPLETED_TRAINING_LOOP"
        print(f"Trial {trial.number} training loop finished. Optuna metric (best val_loss): {optuna_return_metric:.4f}")

    except optuna.exceptions.TrialPruned:
        status_in_objective = "PRUNED_BY_OPTUNA"
        print(f"Trial {trial.number} was pruned by Optuna.")
        # optuna_return_metric remains float('inf')
        raise 

    except Exception as e:
        status_in_objective = f"ERROR: {type(e).__name__}"
        print(f"Unhandled error during Optuna trial {trial.number} training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        for attr_name, original_value in original_config_values.items():
            if original_value == "__NEWLY_ADDED_BY_OPTUNA__":
                if hasattr(config, attr_name): delattr(config, attr_name)
            else:
                setattr(config, attr_name, original_value)
        original_main_output_dir = original_config_values.get("OUTPUT_DIR", config.OUTPUT_DIR)
        utils.OUTPUT_PLOTS_DIR = os.path.join(original_main_output_dir, "output_plots_default")

    objective_end_time = time.time()
    duration_in_objective = objective_end_time - objective_start_time

    # --- Log results to CSV ---
    log_entry_data = {
        "trial_number": trial.number,
        "value_to_optimize (best_val_loss)": optuna_return_metric if status_in_objective == "COMPLETED_TRAINING_LOOP" else np.nan,
        "status_in_objective": status_in_objective,
        "datetime_start": trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if trial.datetime_start else "N/A",
        "duration_seconds_in_objective": duration_in_objective,
    }
    # Add all suggested hyperparameters
    for key, value in trial.params.items():
        log_entry_data[key] = value
    
    # Add final evaluation metrics if the trial completed its training loop
    if status_in_objective == "COMPLETED_TRAINING_LOOP" and final_eval_metrics_for_log:
        log_entry_data["best_epoch_num"] = final_eval_metrics_for_log.get('best_epoch_num', np.nan)
        log_entry_data["best_epoch_train_loss"] = final_eval_metrics_for_log.get('best_epoch_train_loss', np.nan)
        log_entry_data["best_epoch_train_acc"] = final_eval_metrics_for_log.get('best_epoch_train_acc', np.nan)
        log_entry_data["best_epoch_val_loss"] = final_eval_metrics_for_log.get('best_epoch_val_loss', np.nan) # This should match optuna_return_metric
        log_entry_data["best_epoch_val_acc"] = final_eval_metrics_for_log.get('best_epoch_val_acc', np.nan)
        log_entry_data["best_epoch_val_f1"] = final_eval_metrics_for_log.get('best_epoch_val_f1', np.nan)
    else: # Fill with NaNs if not completed or metrics not available
        for metric_key in ["best_epoch_num", "best_epoch_train_loss", "best_epoch_train_acc", 
                           "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1"]:
            log_entry_data[metric_key] = np.nan
            
    log_df = pd.DataFrame([log_entry_data])
    
    # Align columns with the header before saving
    # This requires EXPECTED_LOG_COLUMNS to be defined globally or passed
    log_df_aligned = pd.DataFrame(columns=EXPECTED_LOG_COLUMNS) # Use the globally defined columns
    for col in EXPECTED_LOG_COLUMNS: # Ensure all expected columns are in log_entry_data
        if col not in log_df:
            log_df[col] = np.nan 
    log_df_aligned = pd.concat([log_df_aligned, log_df[EXPECTED_LOG_COLUMNS]], ignore_index=True)


    log_df_aligned.to_csv(HYPERPARAM_LOG_FILE, mode='a', header=False, index=False)

    return optuna_return_metric


if __name__ == "__main__":
    # ... (main execution block from previous version, no changes needed here) ...
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1) 
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    print(f"Optuna study database: {STORAGE_NAME}")
    finished_trials_count = len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]])
    print(f"Number of trials already processed in study: {finished_trials_count}")
    
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS, 
            n_jobs=N_JOBS,
            timeout=None,
        )
    except KeyboardInterrupt:
        print("Optuna optimization stopped by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"An unhandled exception occurred during study.optimize: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Optuna Hyperparameter Tuning Finished ---")
    
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
            best_trial_overall = study.best_trial
            print("\nBest trial (overall completed):")
            print(f"  Trial Number: {best_trial_overall.number}")
            print(f"  Value (Validation Loss): {best_trial_overall.value:.4f}") # This is optuna_return_metric
            print("  Params: ")
            for key, value in best_trial_overall.params.items():
                print(f"    {key}: {value}")
            
            best_params_file = os.path.join(TUNING_OUTPUT_BASE_DIR, "best_hyperparameters.txt")
            with open(best_params_file, "w") as f:
                f.write(f"Best trial number: {best_trial_overall.number}\n")
                f.write(f"Best validation loss (Optuna objective): {best_trial_overall.value:.4f}\n")
                f.write("Best hyperparameters:\n")
                for key, value in best_trial_overall.params.items():
                    f.write(f"  {key}: {value}\n")
                
                # Log additional metrics for the best trial if available from CSV (requires reading CSV)
                # Or, could try to get them from user_attrs if we set them
                df_log = pd.read_csv(HYPERPARAM_LOG_FILE)
                best_trial_log_entry = df_log[df_log["trial_number"] == best_trial_overall.number]
                if not best_trial_log_entry.empty:
                    f.write("\nMetrics for Best Trial (from CSV log):\n")
                    for col in ["best_epoch_num", "best_epoch_train_loss", "best_epoch_train_acc", 
                                "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1"]:
                        if col in best_trial_log_entry:
                             f.write(f"  {col}: {best_trial_log_entry[col].iloc[0]}\n")
                
                best_model_fold_suffix = f"_fold{0}" if config.K_FOLDS > 1 else ""
                trial_prefix_for_best = f"trial{best_trial_overall.number}_"
                best_model_filename_for_best_trial = f"{trial_prefix_for_best}best_model{best_model_fold_suffix}.pth"
                f.write(f"Best model file (relative to its trial folder): {best_model_filename_for_best_trial}\n")
            print(f"Best hyperparameters and metrics saved to {best_params_file}")

        except ValueError: 
            print("No trials completed successfully, cannot determine the best trial from study.best_trial.")
    else: 
        print("No trials completed successfully. Cannot determine best trial.")

    print(f"\nAll trial attempts logged in: {HYPERPARAM_LOG_FILE}")
    print(f"Individual trial outputs are in subfolders under: {TUNING_OUTPUT_BASE_DIR}")