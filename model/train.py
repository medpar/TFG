# model/train.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score # For per-epoch accuracy
import optuna # Import optuna for TrialPruned exception

import config
import data_loader
from model import GaitLSTM
import utils

# train_epoch and validate_epoch remain the same

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_targets_flat = []
    all_predictions_flat = []

    for features, targets in tqdm(data_loader, desc="Training", leave=False):
        features = features.to(device)
        targets = targets.to(device) 

        optimizer.zero_grad()
        outputs = model(features) 
        
        loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted_classes = torch.max(outputs, 2) 
        all_targets_flat.extend(targets.view(-1).cpu().numpy())
        all_predictions_flat.extend(predicted_classes.view(-1).cpu().numpy())

    avg_loss = epoch_loss / len(data_loader)
    accuracy = accuracy_score(all_targets_flat, all_predictions_flat)
    return avg_loss, accuracy

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_targets_flat = []
    all_predictions_flat = []

    with torch.no_grad():
        for features, targets in tqdm(data_loader, desc="Validating", leave=False):
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
            
            epoch_loss += loss.item()

            _, predicted_classes = torch.max(outputs, 2)
            all_targets_flat.extend(targets.view(-1).cpu().numpy())
            all_predictions_flat.extend(predicted_classes.view(-1).cpu().numpy())
            
    avg_loss = epoch_loss / len(data_loader)
    accuracy = accuracy_score(all_targets_flat, all_predictions_flat)
    metrics = utils.calculate_metrics(all_targets_flat, all_predictions_flat) # This returns a dict
    
    # Add individual metrics to the return for easier access if needed
    return avg_loss, accuracy, metrics, all_targets_flat, all_predictions_flat


def main_train_loop(fold_num=None, optuna_trial=None):
    # ... (setup code remains the same: print statements, data loading, model, criterion, optimizer) ...
    print(f"\n--- Starting Training Loop {'for Fold ' + str(fold_num) if fold_num is not None else ''} ---")
    if optuna_trial:
        print(f"--- Optuna Trial: {optuna_trial.number} ---")
    print(f"Using device: {config.DEVICE}")

    if config.K_FOLDS > 1:
        if fold_num is None: raise ValueError("fold_num required for CV.")
        train_loader, val_loader, scaler = data_loader.get_data_loaders(fold_num=fold_num)
    else:
        train_loader, val_loader, _, scaler = data_loader.get_data_loaders() # Get test_loader as _
    
    if not train_loader or not val_loader:
        print("Error: Train or Validation loader is None. Aborting.")
        if optuna_trial: raise optuna.TrialPruned()
        return None, None, float('inf'), {} # Added empty dict for final_metrics

    model = GaitLSTM(
        input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=config.LSTM_DROPOUT,
        linear_dropout=config.LINEAR_DROPOUT
    ).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_metric = float('inf')
    best_epoch_num = 0 # To store the epoch of the best model
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []} # Added val_f1 to history
    
    fold_model_name_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    trial_prefix = f"trial{optuna_trial.number}_" if optuna_trial else ""
    checkpoint_filename = f"{trial_prefix}checkpoint{fold_model_name_suffix}.pth"
    best_model_filename = f"{trial_prefix}best_model{fold_model_name_suffix}.pth"

    final_metrics_for_log = {} # To store metrics of the best model

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        # validate_epoch now returns: avg_loss, accuracy, metrics_dict, _, _
        val_loss, val_acc, val_metrics_dict, _, _ = validate_epoch(model, val_loader, criterion, config.DEVICE)
        val_f1 = val_metrics_dict['f1_score'] # Get F1 from the returned dict
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{epoch_duration:.2f}s] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)


        current_val_metric = val_loss
        if current_val_metric < best_val_metric - config.EARLY_STOPPING_DELTA: # Check delta
            best_val_metric = current_val_metric
            best_epoch_num = epoch + 1
            epochs_no_improve = 0
            # Store metrics of this best epoch for potential logging
            # These are from the *current* epoch that resulted in the best val_loss
            final_metrics_for_log = {
                'best_epoch_train_loss': train_loss,
                'best_epoch_train_acc': train_acc,
                'best_epoch_val_loss': val_loss,
                'best_epoch_val_acc': val_acc,
                'best_epoch_val_f1': val_f1,
                'best_epoch_num': best_epoch_num
            }
            utils.save_checkpoint({
                'epoch': best_epoch_num,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'scaler': scaler,
                'hyperparameters': optuna_trial.params if optuna_trial else None,
                'final_metrics': final_metrics_for_log # Save these metrics too
            }, is_best=True, filename=checkpoint_filename, best_filename=best_model_filename, output_dir=config.OUTPUT_DIR)
        else:
            epochs_no_improve += 1

        if optuna_trial:
            optuna_trial.report(current_val_metric, epoch)
            if optuna_trial.should_prune():
                print(f"Optuna Trial {optuna_trial.number} pruned at epoch {epoch+1}.")
                utils.plot_training_history(history, fold_num=fold_num, trial_num=optuna_trial.number)
                raise optuna.TrialPruned()

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement at epoch {epoch+1}.")
            break
            
    print("Training finished for this run/fold.")
    utils.plot_training_history(history, fold_num=fold_num, trial_num=optuna_trial.number if optuna_trial else None)

    # --- Final Evaluation using the saved best model state ---
    best_model_path = os.path.join(config.OUTPUT_DIR, best_model_filename)
    metrics_from_best_model_eval = {} # For storing final eval metrics

    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation...")
        eval_model = GaitLSTM( # Create a new instance for evaluation
            input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
            bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=0, # No dropout for eval
            linear_dropout=0 # No dropout for eval
        ).to(config.DEVICE)
        
        checkpoint = utils.load_checkpoint(best_model_path, eval_model, optimizer=None) # Returns full checkpoint
        
        if checkpoint is not None:
            # If we saved metrics in checkpoint, use them, otherwise re-evaluate
            if 'final_metrics' in checkpoint:
                 metrics_from_best_model_eval = checkpoint['final_metrics']
                 # Ensure all expected keys are present, fill with NaN if not, for consistency
                 metrics_from_best_model_eval.setdefault('best_epoch_train_loss', np.nan)
                 metrics_from_best_model_eval.setdefault('best_epoch_train_acc', np.nan)
                 metrics_from_best_model_eval.setdefault('best_epoch_val_loss', checkpoint.get('best_val_metric', np.nan)) # val_loss is best_val_metric
                 metrics_from_best_model_eval.setdefault('best_epoch_val_acc', np.nan)
                 metrics_from_best_model_eval.setdefault('best_epoch_val_f1', np.nan)
                 metrics_from_best_model_eval.setdefault('best_epoch_num', checkpoint.get('epoch',0))

                 print("\n--- Final Validation Set Evaluation (using loaded best model state) ---")
                 # Re-evaluate to be sure, or if not all metrics were saved in 'final_metrics'
                 val_loss_eval, val_acc_eval, val_metrics_dict_eval, val_targets_flat, val_preds_flat = validate_epoch(eval_model, val_loader, criterion, config.DEVICE)
                 print(f"Eval Best Model - Val Loss: {val_loss_eval:.4f}, Val Acc: {val_acc_eval:.4f}, Val F1: {val_metrics_dict_eval['f1_score']:.4f}")
                 
                 # Update metrics_from_best_model_eval with potentially more accurate re-evaluation
                 metrics_from_best_model_eval['best_epoch_val_loss'] = val_loss_eval
                 metrics_from_best_model_eval['best_epoch_val_acc'] = val_acc_eval
                 metrics_from_best_model_eval['best_epoch_val_f1'] = val_metrics_dict_eval['f1_score']
                 # We don't have train loss/acc from this specific re-evaluation, so keep from checkpoint
                 metrics_from_best_model_eval['best_epoch_train_loss'] = metrics_from_best_model_eval.get('best_epoch_train_loss', history['train_loss'][best_epoch_num-1] if best_epoch_num > 0 and len(history['train_loss']) >= best_epoch_num else np.nan)
                 metrics_from_best_model_eval['best_epoch_train_acc'] = metrics_from_best_model_eval.get('best_epoch_train_acc', history['train_acc'][best_epoch_num-1] if best_epoch_num > 0 and len(history['train_acc']) >= best_epoch_num else np.nan)


                 class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
                 utils.plot_confusion_matrix_custom(val_targets_flat, val_preds_flat, class_names, 
                                                    title="Validation Confusion Matrix (Best Model)", fold_num=fold_num, 
                                                    trial_num=optuna_trial.number if optuna_trial else None)
            else:
                print("Warning: 'final_metrics' not found in checkpoint. Re-evaluating...")
                # Fallback to just re-evaluating if 'final_metrics' key is missing
                val_loss_eval, val_acc_eval, val_metrics_dict_eval, _, _ = validate_epoch(eval_model, val_loader, criterion, config.DEVICE)
                metrics_from_best_model_eval = {
                    'best_epoch_val_loss': val_loss_eval,
                    'best_epoch_val_acc': val_acc_eval,
                    'best_epoch_val_f1': val_metrics_dict_eval['f1_score'],
                    'best_epoch_num': checkpoint.get('epoch',0),
                    'best_epoch_train_loss': history['train_loss'][checkpoint.get('epoch',1)-1] if checkpoint.get('epoch',0) > 0 else np.nan, # Approx
                    'best_epoch_train_acc': history['train_acc'][checkpoint.get('epoch',1)-1] if checkpoint.get('epoch',0) > 0 else np.nan,  # Approx
                }
        else: # Checkpoint could not be loaded
            print("Could not load best model checkpoint. Metrics will be from last model state.")
            # Use metrics from the very last epoch of training if best model didn't load
            # This is a fallback and might not represent the "best" performance
            if history['val_loss']: # Check if history has any data
                metrics_from_best_model_eval = {
                    'best_epoch_train_loss': history['train_loss'][-1],
                    'best_epoch_train_acc': history['train_acc'][-1],
                    'best_epoch_val_loss': history['val_loss'][-1],
                    'best_epoch_val_acc': history['val_acc'][-1],
                    'best_epoch_val_f1': history['val_f1'][-1] if history['val_f1'] else np.nan,
                    'best_epoch_num': config.NUM_EPOCHS # Or actual last epoch if early stopping
                }
            else: # No history, e.g. if training failed very early
                 metrics_from_best_model_eval = {k: np.nan for k in ['best_epoch_train_loss', 'best_epoch_train_acc', 'best_epoch_val_loss', 'best_epoch_val_acc', 'best_epoch_val_f1', 'best_epoch_num']}


    else: # Best model path does not exist
        print(f"Best model file {best_model_path} not found. No final evaluation metrics to report for this specific path.")
        # Fill with NaNs if no best model was saved/found
        metrics_from_best_model_eval = {k: np.nan for k in ['best_epoch_train_loss', 'best_epoch_train_acc', 'best_epoch_val_loss', 'best_epoch_val_acc', 'best_epoch_val_f1', 'best_epoch_num']}


    # The 'model' variable here is the one from the end of the training loop,
    # 'eval_model' is the one loaded from the best checkpoint (if successful).
    # We return 'eval_model' if loaded, otherwise the last state 'model'.
    final_model_to_return = eval_model if 'eval_model' in locals() and checkpoint is not None else model
    
    # best_val_metric is the primary metric Optuna uses (lowest val_loss during training)
    return final_model_to_return, scaler, best_val_metric, metrics_from_best_model_eval


if __name__ == '__main__':
    # ... (main execution block remains the same) ...
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    elif config.DEVICE == "mps":
        pass 

    if config.K_FOLDS > 1:
        print(f"Starting {config.K_FOLDS}-Fold Cross-Validation Training...")
        for i in range(config.K_FOLDS):
            print(f"\n===== FOLD {i+1}/{config.K_FOLDS} =====")
            _, _, _, _ = main_train_loop(fold_num=i) 
        print("\nCross-validation finished.")
    else:
        print("Starting Single Train/Validation/Test Training...")
        main_train_loop()