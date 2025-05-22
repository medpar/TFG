# model/hyperparameter_tuning_bilstm_cnn.py
import os
import time
import shutil
import optuna
import torch
import numpy as np
import pandas as pd

import config # Optuna will modify this
import data_loader
# model.py now contains BiLSTMCnn1D, train.py will use it based on config
import train
import utils

# --- Optuna Study Configuration ---
N_TRIALS = 70
N_JOBS = 1
STUDY_NAME = "hyp_tuning_bilstm_cnn" # Updated study name for F1 optimization
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"
TUNING_OUTPUT_BASE_DIR = os.path.join(config.OUTPUT_DIR, "hyp_tuning") # New output
os.makedirs(TUNING_OUTPUT_BASE_DIR, exist_ok=True)
HYPERPARAM_LOG_FILE = os.path.join(TUNING_OUTPUT_BASE_DIR, "hyperparameter_log_bilstm_cnn.csv")

# --- Define Columns for the Log CSV ---
MAX_CNN_DEPTH = 2 
dynamic_cnn_param_keys = []
for i in range(MAX_CNN_DEPTH):
    dynamic_cnn_param_keys.extend([f"cnn_out_channels_filters_{i}", f"cnn_kernel_size_{i}", f"cnn_stride_{i}"])

EXPECTED_LOG_COLUMNS = [
    "trial_number", "value_to_optimize (best_val_f1)", "status_in_objective", # << CHANGED
    "datetime_start", "duration_seconds_in_objective",
] + dynamic_cnn_param_keys + [
    "cnn_dropout", "lstm_hidden_size", "num_lstm_layers", "lstm_dropout",
    "linear_dropout", "learning_rate", "weight_decay",
    "sequence_length", "early_stopping_patience",
    "best_epoch_num", "best_epoch_train_loss", "best_epoch_train_acc",
    "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1" # This will be the same as value_to_optimize
]

if not os.path.exists(HYPERPARAM_LOG_FILE):
    header_df = pd.DataFrame(columns=EXPECTED_LOG_COLUMNS)
    header_df.to_csv(HYPERPARAM_LOG_FILE, index=False)


def objective(trial: optuna.trial.Trial):
    objective_start_time = time.time()
    # --- 1. Suggest Hyperparameters for BiLSTMCnn1D ---
    # (Hyperparameter suggestion logic remains the same as your provided script)
    cnn_depth = MAX_CNN_DEPTH
    suggested_cnn_params = {}
    cnn_out_channels_list = []
    cnn_kernel_sizes_list = []
    cnn_strides_list = []

    for i in range(cnn_depth):
        filters = trial.suggest_categorical(f"cnn_out_channels_filters_{i}", [16, 32, 48, 64]) # Expanded slightly
        cnn_out_channels_list.append(filters)
        suggested_cnn_params[f"cnn_out_channels_filters_{i}"] = filters

        kernel = trial.suggest_categorical(f"cnn_kernel_size_{i}", [3, 5, 7])
        cnn_kernel_sizes_list.append(kernel)
        suggested_cnn_params[f"cnn_kernel_size_{i}"] = kernel
        
        stride = 1 
        cnn_strides_list.append(stride)
        suggested_cnn_params[f"cnn_stride_{i}"] = stride

    suggested_cnn_dropout = trial.suggest_float("cnn_dropout", 0.1, 0.5, step=0.1) # Wider range
    suggested_cnn_params["cnn_dropout"] = suggested_cnn_dropout

    suggested_lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 96, 128]) # Wider range
    suggested_num_lstm_layers = trial.suggest_categorical("num_lstm_layers", [1, 2])
    suggested_lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1) if suggested_num_lstm_layers > 1 else 0.0
    
    suggested_linear_dropout = trial.suggest_float("linear_dropout", 0.2, 0.7, step=0.1) # Wider range
    suggested_weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 5e-4) # Wider range
    suggested_learning_rate = trial.suggest_loguniform("learning_rate", 5e-5, 1e-3) 
    suggested_sequence_length = trial.suggest_categorical("sequence_length", [75, 100, 125])
    suggested_early_stopping_patience = trial.suggest_int("early_stopping_patience", 10, 20)

    # --- 2. Update Global Config ---
    original_config_values = {}
    def update_config_value(attr_name, new_value):
        if hasattr(config, attr_name):
            original_config_values[attr_name] = getattr(config, attr_name)
        else:
            original_config_values[attr_name] = "__NEWLY_ADDED_BY_OPTUNA__"
        setattr(config, attr_name, new_value)

    update_config_value("MODEL_TYPE", "BiLSTMCnn1D")
    update_config_value("CNN_OUT_CHANNELS", cnn_out_channels_list)
    update_config_value("CNN_KERNEL_SIZES", cnn_kernel_sizes_list)
    update_config_value("CNN_STRIDES", cnn_strides_list)
    update_config_value("CNN_DROPOUT", suggested_cnn_dropout)
    update_config_value("LSTM_HIDDEN_SIZE", suggested_lstm_hidden_size)
    update_config_value("NUM_LSTM_LAYERS", suggested_num_lstm_layers)
    update_config_value("LSTM_DROPOUT", suggested_lstm_dropout)
    update_config_value("LINEAR_DROPOUT", suggested_linear_dropout)
    update_config_value("WEIGHT_DECAY", suggested_weight_decay)
    update_config_value("LEARNING_RATE", suggested_learning_rate)
    update_config_value("SEQUENCE_LENGTH", suggested_sequence_length)
    update_config_value("EARLY_STOPPING_PATIENCE", suggested_early_stopping_patience)
    update_config_value("NUM_EPOCHS", 75) 

    # --- 3. Create Unique Output Directory for this Trial ---
    trial_output_dir_name = f"trial_{trial.number}_{int(objective_start_time)}"
    trial_output_dir = os.path.join(TUNING_OUTPUT_BASE_DIR, trial_output_dir_name)
    os.makedirs(trial_output_dir, exist_ok=True)
    update_config_value("OUTPUT_DIR", trial_output_dir)
    utils.OUTPUT_PLOTS_DIR = os.path.join(trial_output_dir, "output_plots")
    os.makedirs(utils.OUTPUT_PLOTS_DIR, exist_ok=True)

    print(f"\n--- Optuna Trial {trial.number} (BiLSTMCnn1D for F1-Score) ---")
    print(f"Hyperparameters: {trial.params}")
    print(f"  CNN_OUT_CHANNELS: {cnn_out_channels_list}")
    # ... (other print statements)

    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    update_config_value("DEVICE", device)
    print(f"Using device: {config.DEVICE}")

    # --- 4. Run Training Loop ---
    fold_to_run_for_tuning = 0
    # optuna_return_metric will now be F1-score (to be maximized)
    # If an error occurs, or pruned, we want a very low F1-score
    optuna_return_metric = -1.0 # Default for failed/pruned trials (F1 is between 0 and 1)
    final_eval_metrics_for_log = {}
    status_in_objective = "INITIATED"

    try:
        # main_train_loop returns: final_model, scaler, best_val_loss, final_metrics_dict
        _, _, _, final_eval_metrics_for_log = train.main_train_loop(
            fold_num=fold_to_run_for_tuning if config.K_FOLDS > 1 else None,
            optuna_trial=trial
        )
        
        # We want to MAXIMIZE F1-score. Optuna's default direction is minimize.
        # So, we will return the F1-score and set study direction to "maximize".
        optuna_return_metric = final_eval_metrics_for_log.get('best_epoch_val_f1', -1.0) # Default to -1 if not found
        if optuna_return_metric is np.nan: optuna_return_metric = -1.0 # Handle NaN case

        status_in_objective = "COMPLETED_TRAINING_LOOP"
        print(f"Trial {trial.number} training loop finished. Optuna metric (best_epoch_val_f1): {optuna_return_metric:.4f}")

    except optuna.exceptions.TrialPruned:
        status_in_objective = "PRUNED_BY_OPTUNA"
        print(f"Trial {trial.number} was pruned by Optuna.")
        # optuna_return_metric remains -1.0
        raise 
    except Exception as e:
        status_in_objective = f"ERROR: {type(e).__name__}"
        print(f"Unhandled error during Optuna trial {trial.number} training: {e}")
        import traceback
        traceback.print_exc()
        # optuna_return_metric remains -1.0
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
        "value_to_optimize (best_val_f1)": optuna_return_metric if status_in_objective == "COMPLETED_TRAINING_LOOP" else np.nan, # << CHANGED
        "status_in_objective": status_in_objective,
        "datetime_start": trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if trial.datetime_start else "N/A",
        "duration_seconds_in_objective": duration_in_objective,
    }
    for key, value in trial.params.items():
        log_entry_data[key] = value
    
    if status_in_objective == "COMPLETED_TRAINING_LOOP" and final_eval_metrics_for_log:
        log_entry_data.update(final_eval_metrics_for_log) # This already contains best_epoch_val_f1 etc.
    else:
        for metric_key in ["best_epoch_num", "best_epoch_train_loss", "best_epoch_train_acc",
                           "best_epoch_val_loss", "best_epoch_val_acc", "best_epoch_val_f1"]:
            log_entry_data[metric_key] = np.nan
            
    log_df = pd.DataFrame([log_entry_data])
    df_aligned_for_csv = pd.DataFrame(columns=EXPECTED_LOG_COLUMNS)
    for col in EXPECTED_LOG_COLUMNS:
        if col in log_df.columns:
            df_aligned_for_csv[col] = log_df[col]
        else:
            df_aligned_for_csv[col] = np.nan
            
    df_aligned_for_csv.to_csv(HYPERPARAM_LOG_FILE, mode='a', header=False, index=False)

    return optuna_return_metric # This is now best_epoch_val_f1 (higher is better)


if __name__ == "__main__":
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED, n_startup_trials=15, multivariate=True) # Increased startup trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=7, n_warmup_steps=15, interval_steps=1) # Adjusted pruner
    
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_NAME, 
        direction="maximize",  # <<<<<<<<<<<<<<<<<<<<<<<<<<<< CHANGED TO MAXIMIZE F1-SCORE
        sampler=sampler, pruner=pruner, load_if_exists=True
    )

    print(f"Optuna study database: {STORAGE_NAME}")
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
    
    print("\n--- Optuna Hyperparameter Tuning Finished (BiLSTMCnn1D for F1-Score) ---")
    
    all_study_trials = study.trials
    completed_trials = [t for t in all_study_trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Total trials in study: {len(all_study_trials)}")
    print(f"  Successfully completed: {len(completed_trials)}")

    if completed_trials:
        try:
            best_trial_overall = study.best_trial
            print("\nBest trial (overall completed - maximizing F1-score):")
            print(f"  Trial Number: {best_trial_overall.number}")
            print(f"  Value (Validation F1-score): {best_trial_overall.value:.4f}") # << CHANGED
            print("  Params: ")
            for key, value in best_trial_overall.params.items():
                print(f"    {key}: {value}")
            
            best_params_file = os.path.join(TUNING_OUTPUT_BASE_DIR, "best_hyperparameters_bilstm_cnn_f1.txt")
            with open(best_params_file, "w") as f:
                f.write(f"Best trial number: {best_trial_overall.number}\n")
                f.write(f"Best validation F1-score (Optuna objective): {best_trial_overall.value:.4f}\n") # << CHANGED
                f.write("Best hyperparameters:\n")
                for key, value in best_trial_overall.params.items():
                    f.write(f"  {key}: {value}\n")
                
                df_log = pd.read_csv(HYPERPARAM_LOG_FILE)
                best_trial_log_entry = df_log[df_log["trial_number"] == best_trial_overall.number]
                if not best_trial_log_entry.empty:
                    f.write("\nMetrics for Best Trial (from CSV log):\n")
                    for col in EXPECTED_LOG_COLUMNS: # Log all relevant columns from the best trial's log entry
                        if col in best_trial_log_entry.columns and col not in best_trial_overall.params and \
                           col not in ["trial_number", "value_to_optimize (best_val_f1)", "status_in_objective", "datetime_start", "duration_seconds_in_objective"]:
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