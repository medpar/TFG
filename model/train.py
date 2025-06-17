# model/train.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import optuna

import config
import data_loader
from model import GaitLSTM
import utils

# train_epoch and validate_epoch remain the same
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_targets_flat, all_predictions_flat = [], []
    for features, targets in tqdm(data_loader, desc="Training", leave=False):
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted_classes = torch.max(outputs, 2)
        all_targets_flat.extend(targets.view(-1).cpu().numpy())
        all_predictions_flat.extend(predicted_classes.view(-1).cpu().numpy())
    return epoch_loss / len(data_loader), accuracy_score(all_targets_flat, all_predictions_flat)

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_targets_flat, all_predictions_flat = [], []
    with torch.no_grad():
        for features, targets in tqdm(data_loader, desc="Validating", leave=False):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
            epoch_loss += loss.item()
            _, predicted_classes = torch.max(outputs, 2)
            all_targets_flat.extend(targets.view(-1).cpu().numpy())
            all_predictions_flat.extend(predicted_classes.view(-1).cpu().numpy())
    avg_loss = epoch_loss / len(data_loader)
    accuracy = accuracy_score(all_targets_flat, all_predictions_flat)
    metrics_dict = utils.calculate_metrics(all_targets_flat, all_predictions_flat, average='weighted') # Ensure 'weighted' or 'macro' as needed
    return avg_loss, accuracy, metrics_dict, all_targets_flat, all_predictions_flat


def main_train_loop(fold_num=None, optuna_trial=None):
    print(f"\n--- Starting Training Loop {'for Fold ' + str(fold_num) if fold_num is not None else ''} ---")
    if optuna_trial: print(f"--- Optuna Trial: {optuna_trial.number} ---")
    print(f"Using device: {config.DEVICE}")

    class_weights_tensor = None
    if config.K_FOLDS > 1:
        if fold_num is None: raise ValueError("fold_num required for CV.")
        train_loader, val_loader, scaler, class_weights = data_loader.get_data_loaders(fold_num=fold_num)
    else:
        # get_data_loaders for single split now returns 5 values (including test_loader and weights)
        train_loader, val_loader, _, scaler, class_weights = data_loader.get_data_loaders()
    
    if config.USE_WEIGHTED_LOSS and class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(config.DEVICE)
        print(f"Using class weights for loss: {class_weights_tensor.cpu().numpy()}")
    
    if not train_loader or not val_loader:
        print("Error: Train or Validation loader is None. Aborting.")
        if optuna_trial: raise optuna.TrialPruned()
        # Return structure: model, scaler, optuna_metric_to_return, full_metrics_dict
        return None, None, 0.0 if config.OPTIMIZE_METRIC == 'f1' else float('inf'), {} # Default for bad F1 or high loss

    model = GaitLSTM(
        input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=config.LSTM_DROPOUT,
        linear_dropout=config.LINEAR_DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) if config.USE_WEIGHTED_LOSS and class_weights_tensor is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Initialize based on what we are optimizing (F1: lower is worse, Loss: higher is worse)
    best_val_objective_metric = 0.0 if config.OPTIMIZE_METRIC == 'f1' else float('inf') 
    best_epoch_num = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    fold_model_name_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    trial_prefix = f"trial{optuna_trial.number}_" if optuna_trial else ""
    checkpoint_filename = f"{trial_prefix}checkpoint{fold_model_name_suffix}.pth"
    best_model_filename = f"{trial_prefix}best_model{fold_model_name_suffix}.pth"
    
    final_metrics_for_log = {}

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_metrics_dict, _, _ = validate_epoch(model, val_loader, criterion, config.DEVICE)
        val_f1 = val_metrics_dict['f1_score'] # Assuming 'f1_score' is weighted/macro as desired
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{epoch_duration:.2f}s] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Determine current metric for Optuna reporting and early stopping/best model saving
        current_report_metric = val_f1 if config.OPTIMIZE_METRIC == 'f1' else val_loss
        
        is_best = False
        if config.OPTIMIZE_METRIC == 'f1':
            if current_report_metric > best_val_objective_metric + config.EARLY_STOPPING_DELTA: # F1 improves if higher
                is_best = True
        else: # Optimizing for loss (minimize)
            if current_report_metric < best_val_objective_metric - config.EARLY_STOPPING_DELTA: # Loss improves if lower
                is_best = True

        if is_best:
            best_val_objective_metric = current_report_metric
            best_epoch_num = epoch + 1
            epochs_no_improve = 0
            final_metrics_for_log = {
                'best_epoch_train_loss': train_loss, 'best_epoch_train_acc': train_acc,
                'best_epoch_val_loss': val_loss, 'best_epoch_val_acc': val_acc,
                'best_epoch_val_f1': val_f1, 'best_epoch_num': best_epoch_num
            }
            utils.save_checkpoint({
                'epoch': best_epoch_num, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'best_val_objective_metric': best_val_objective_metric, # Save the actual best metric value
                'optimized_metric_name': config.OPTIMIZE_METRIC,         # Save which metric was optimized
                'scaler': scaler, 'hyperparameters': optuna_trial.params if optuna_trial else None,
                'final_metrics': final_metrics_for_log,
                'class_weights_used': class_weights_tensor.cpu().numpy() if class_weights_tensor is not None else None
            }, is_best=True, filename=checkpoint_filename, best_filename=best_model_filename, output_dir=config.OUTPUT_DIR)
        else:
            epochs_no_improve += 1

        if optuna_trial:
            optuna_trial.report(val_f1 if config.OPTIMIZE_METRIC_FOR_OPTUNA == 'f1' else val_loss, epoch)
            if optuna_trial.should_prune():
                print(f"Optuna Trial {optuna_trial.number} pruned at epoch {epoch+1}.")
                utils.plot_training_history(history, fold_num=fold_num, trial_num=optuna_trial.number)
                raise optuna.TrialPruned()

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (metric: {config.OPTIMIZE_METRIC}).")
            break
            
    print("Training loop finished for this run/fold.")
    utils.plot_training_history(history, fold_num=fold_num, trial_num=optuna_trial.number if optuna_trial else None)

    best_model_path = os.path.join(config.OUTPUT_DIR, best_model_filename)
    metrics_from_best_model_eval = final_metrics_for_log 
    eval_model_instance = model 

    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation...")
        eval_model_instance = GaitLSTM(
             input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
             num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
             bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=0, linear_dropout=0
        ).to(config.DEVICE)
        checkpoint = utils.load_checkpoint(best_model_path, eval_model_instance, optimizer=None)
        
        if checkpoint:
            if 'final_metrics' in checkpoint:
                 metrics_from_best_model_eval = checkpoint['final_metrics']
            
            print("\n--- Final Validation Set Evaluation (using loaded best model state) ---")
            val_loss_eval, val_acc_eval, val_metrics_dict_eval, val_targets_flat, val_preds_flat = validate_epoch(eval_model_instance, val_loader, criterion, config.DEVICE)
            print(f"Eval Best Model - Val Loss: {val_loss_eval:.4f}, Val Acc: {val_acc_eval:.4f}, Val F1 (weighted): {val_metrics_dict_eval['f1_score']:.4f}")
            
            metrics_from_best_model_eval['best_epoch_val_loss'] = val_loss_eval
            metrics_from_best_model_eval['best_epoch_val_acc'] = val_acc_eval
            metrics_from_best_model_eval['best_epoch_val_f1'] = val_metrics_dict_eval['f1_score']
            metrics_from_best_model_eval.setdefault('best_epoch_train_loss', np.nan)
            metrics_from_best_model_eval.setdefault('best_epoch_train_acc', np.nan)
            metrics_from_best_model_eval.setdefault('best_epoch_num', checkpoint.get('epoch',0))

            class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
            utils.plot_confusion_matrix_custom(val_targets_flat, val_preds_flat, class_names, 
                                               title="Validation Confusion Matrix (Best Model)", fold_num=fold_num, 
                                               trial_num=optuna_trial.number if optuna_trial else None)
        else:
            print("Could not load best model checkpoint. Using last model state metrics.")
            if history['val_loss']:
                metrics_from_best_model_eval = {
                    'best_epoch_train_loss': history['train_loss'][-1], 'best_epoch_train_acc': history['train_acc'][-1],
                    'best_epoch_val_loss': history['val_loss'][-1], 'best_epoch_val_acc': history['val_acc'][-1],
                    'best_epoch_val_f1': history['val_f1'][-1], 'best_epoch_num': len(history['val_loss'])
                }
    else:
        print(f"Best model file {best_model_path} not found.")
        if history['val_loss']:
             metrics_from_best_model_eval = {key: history[key.replace('best_epoch_', '')][-1] for key in ['best_epoch_train_loss', 'best_epoch_train_acc', 'best_epoch_val_loss', 'best_epoch_val_acc', 'best_epoch_val_f1']}
             metrics_from_best_model_eval['best_epoch_num'] = len(history['val_loss'])


    final_model_to_return = eval_model_instance if os.path.exists(best_model_path) and checkpoint else model
    
    optuna_metric_to_return = 0.0
    if config.OPTIMIZE_METRIC_FOR_OPTUNA == 'f1':
        optuna_metric_to_return = metrics_from_best_model_eval.get('best_epoch_val_f1', 0.0)
    else:
        optuna_metric_to_return = metrics_from_best_model_eval.get('best_epoch_val_loss', float('inf'))

    return final_model_to_return, scaler, optuna_metric_to_return, metrics_from_best_model_eval


if __name__ == '__main__':
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
            # FIX: Unpack all four returned values, even if some are not used.
            _, _, _, _ = main_train_loop(fold_num=i) 
        print("\nCross-validation finished.")
    else:
        print("Starting Single Train/Validation/Test Training...")
        # FIX: Unpack all four returned values for the single training run as well.
        main_train_loop()