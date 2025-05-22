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
from sklearn.utils.class_weight import compute_class_weight # For class weights
import optuna # Import optuna for TrialPruned exception

import config
import data_loader
from model import BiLSTMCnn1D # Import the new model
import utils

def calculate_class_weights(train_loader, num_classes, device):
    """
    Calculates class weights based on inverse frequency in the training data.
    """
    print("Calculating class weights for loss function...")
    all_targets_for_weights = []
    
    # Check if the dataset is expected to return time_seq
    # This depends on how GaitPhaseDataset is initialized, which in turn uses config.TIME_COLUMN
    # A simple way is to check the first item if the loader is not empty
    
    # Corrected loop to handle variable number of items from DataLoader
    for batch_data in train_loader:
        if len(batch_data) == 3: # features, targets, time_seq
            _, targets_batch, _ = batch_data # Unpack all three, ignore features and time_seq
        elif len(batch_data) == 2: # features, targets
            _, targets_batch = batch_data # Unpack two, ignore features
        else:
            raise ValueError(f"Unexpected number of items in DataLoader batch: {len(batch_data)}")
            
        all_targets_for_weights.extend(targets_batch.view(-1).numpy())
    
    if not all_targets_for_weights:
        print("Warning: No target data found to calculate class weights. Using uniform weights.")
        return torch.ones(num_classes).to(device)

    unique_classes = np.unique(all_targets_for_weights)
    print(f"Unique classes found in training data for weighting: {unique_classes}")
    
    filtered_targets = [t for t in all_targets_for_weights if 0 <= t < num_classes]
    if not filtered_targets:
        print("Warning: No valid target labels (0 to num_classes-1) found. Using uniform weights.")
        return torch.ones(num_classes).to(device)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes), 
        y=filtered_targets
    )
    print(f"Computed class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_targets_flat = []
    all_predictions_flat = []

    for batch_data in tqdm(data_loader, desc="Training", leave=False): # Corrected loop
        if len(batch_data) == 3:
            features, targets, _ = batch_data # Ignore time_seq if present
        else: # len(batch_data) == 2
            features, targets = batch_data

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

    avg_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(all_targets_flat, all_predictions_flat) if all_targets_flat else 0
    return avg_loss, accuracy

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_targets_flat = []
    all_predictions_flat = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Validating", leave=False): # Corrected loop
            if len(batch_data) == 3:
                features, targets, _ = batch_data # Ignore time_seq if present
            else: # len(batch_data) == 2
                features, targets = batch_data
                
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
            
            epoch_loss += loss.item()

            _, predicted_classes = torch.max(outputs, 2)
            all_targets_flat.extend(targets.view(-1).cpu().numpy())
            all_predictions_flat.extend(predicted_classes.view(-1).cpu().numpy())
            
    avg_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = accuracy_score(all_targets_flat, all_predictions_flat) if all_targets_flat else 0
    metrics = utils.calculate_metrics(all_targets_flat, all_predictions_flat) if all_targets_flat else {
        'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0
    }
    
    return avg_loss, accuracy, metrics, all_targets_flat, all_predictions_flat

def main_train_loop(fold_num=None, optuna_trial=None):
    print(f"\n--- Starting Training Loop {'for Fold ' + str(fold_num) if fold_num is not None else ''} ---")
    if optuna_trial: print(f"--- Optuna Trial: {optuna_trial.number} ---")
    print(f"Using device: {config.DEVICE}")
    print(f"Model Type: {config.MODEL_TYPE}")

    # --- Data Loading ---
    # Initialize test_loader to None for the case of K_FOLDS > 1
    test_loader = None 
    if config.K_FOLDS > 1:
        if fold_num is None: raise ValueError("fold_num required for CV.")
        train_loader, val_loader, scaler = data_loader.get_data_loaders(fold_num=fold_num)
        # test_loader remains None for CV folds during hyperparameter tuning / main CV loop
    else: # Single split
        train_loader, val_loader, test_loader, scaler = data_loader.get_data_loaders()
    
    if not train_loader or len(train_loader.dataset) == 0: # Check if dataset is empty
        print("Error: Train loader is None or empty. Aborting.")
        if optuna_trial: raise optuna.TrialPruned("Train loader empty")
        return None, None, float('inf'), {}
    if not val_loader or len(val_loader.dataset) == 0: # Check if dataset is empty
        print("Error: Validation loader is None or empty. Aborting.")
        if optuna_trial: raise optuna.TrialPruned("Validation loader empty")
        return None, None, float('inf'), {}


    class_weights = calculate_class_weights(train_loader, config.NUM_CLASSES, config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")

    if config.MODEL_TYPE == "BiLSTMCnn1D":
        model = BiLSTMCnn1D(
            input_size=config.NUM_FEATURES,
            num_classes=config.NUM_CLASSES,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            cnn_strides=config.CNN_STRIDES,
            cnn_padding=config.CNN_PADDING,
            cnn_activation=config.CNN_ACTIVATION,
            cnn_dropout_rate=config.CNN_DROPOUT,
            lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
            num_lstm_layers=config.NUM_LSTM_LAYERS,
            lstm_dropout_rate=config.LSTM_DROPOUT,
            bidirectional_lstm=config.BIDIRECTIONAL_LSTM,
            linear_dropout_rate=config.LINEAR_DROPOUT
        ).to(config.DEVICE)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE} in config.py")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True) # Increased patience for scheduler

    best_val_metric = float('inf')
    best_epoch_num = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    fold_model_name_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    trial_prefix = f"trial{optuna_trial.number}_" if optuna_trial else ""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True) # Ensure output dir for this trial exists
    checkpoint_filename = f"{trial_prefix}checkpoint{fold_model_name_suffix}.pth"
    best_model_filename = f"{trial_prefix}best_model{fold_model_name_suffix}.pth"
    final_metrics_for_log = {}

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_metrics_dict, _, _ = validate_epoch(model, val_loader, criterion, config.DEVICE)
        val_f1 = val_metrics_dict['f1_score']
        scheduler.step(val_loss)
        
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
        if current_val_metric < best_val_metric - config.EARLY_STOPPING_DELTA:
            best_val_metric = current_val_metric
            best_epoch_num = epoch + 1
            epochs_no_improve = 0
            final_metrics_for_log = {
                'best_epoch_train_loss': train_loss, 'best_epoch_train_acc': train_acc,
                'best_epoch_val_loss': val_loss, 'best_epoch_val_acc': val_acc,
                'best_epoch_val_f1': val_f1, 'best_epoch_num': best_epoch_num
            }
            utils.save_checkpoint({
                'epoch': best_epoch_num, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'best_val_metric': best_val_metric,
                'scaler': scaler, 'model_type': config.MODEL_TYPE,
                'hyperparameters': optuna_trial.params if optuna_trial else None,
                'final_metrics': final_metrics_for_log
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
            
    print("Training loop finished for this run/fold.")
    utils.plot_training_history(history, fold_num=fold_num, trial_num=optuna_trial.number if optuna_trial else None)

    best_model_path = os.path.join(config.OUTPUT_DIR, best_model_filename)
    metrics_from_best_model_eval = {}
    eval_model = None 

    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final evaluation...")
        eval_model = BiLSTMCnn1D( 
            input_size=config.NUM_FEATURES, num_classes=config.NUM_CLASSES,
            cnn_out_channels=config.CNN_OUT_CHANNELS, cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            cnn_strides=config.CNN_STRIDES, cnn_padding=config.CNN_PADDING, cnn_activation=config.CNN_ACTIVATION,
            cnn_dropout_rate=0, lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
            num_lstm_layers=config.NUM_LSTM_LAYERS, lstm_dropout_rate=0,
            bidirectional_lstm=config.BIDIRECTIONAL_LSTM, linear_dropout_rate=0
        ).to(config.DEVICE)
        
        checkpoint = utils.load_checkpoint(best_model_path, eval_model, optimizer=None) # Returns full checkpoint dict
        
        if checkpoint is not None: # Checkpoint loaded successfully
            loaded_metrics = checkpoint.get('final_metrics', {}) # Get metrics saved at best epoch
            # Initialize with Nones or NaNs
            metrics_from_best_model_eval = {
                'best_epoch_train_loss': loaded_metrics.get('best_epoch_train_loss', np.nan),
                'best_epoch_train_acc': loaded_metrics.get('best_epoch_train_acc', np.nan),
                'best_epoch_val_loss': checkpoint.get('best_val_metric', np.nan),
                'best_epoch_val_acc': loaded_metrics.get('best_epoch_val_acc', np.nan),
                'best_epoch_val_f1': loaded_metrics.get('best_epoch_val_f1', np.nan),
                'best_epoch_num': checkpoint.get('epoch', 0)
            }
            print("\n--- Final Validation Set Re-evaluation (using loaded best model state) ---")
            val_loss_eval, val_acc_eval, val_metrics_dict_eval, val_targets_flat, val_preds_flat = validate_epoch(eval_model, val_loader, criterion, config.DEVICE)
            print(f"Re-eval Best Model - Val Loss: {val_loss_eval:.4f}, Val Acc: {val_acc_eval:.4f}, Val F1: {val_metrics_dict_eval['f1_score']:.4f}")
            
            # Update with these re-evaluated metrics for val, keep train from checkpoint
            metrics_from_best_model_eval['best_epoch_val_loss'] = val_loss_eval
            metrics_from_best_model_eval['best_epoch_val_acc'] = val_acc_eval
            metrics_from_best_model_eval['best_epoch_val_f1'] = val_metrics_dict_eval['f1_score']

            class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
            utils.plot_confusion_matrix_custom(val_targets_flat, val_preds_flat, class_names, 
                                               title="Validation CM (Best Model)", fold_num=fold_num, 
                                               trial_num=optuna_trial.number if optuna_trial else None)
        else: 
            print("Could not load best model checkpoint. Metrics will be from last model state if available.")
            eval_model = model 
            if history['val_loss']:
                last_epoch_idx = np.argmin(history['val_loss']) # Find index of min val_loss in history
                metrics_from_best_model_eval = {
                    'best_epoch_train_loss': history['train_loss'][last_epoch_idx],
                    'best_epoch_train_acc': history['train_acc'][last_epoch_idx],
                    'best_epoch_val_loss': history['val_loss'][last_epoch_idx],
                    'best_epoch_val_acc': history['val_acc'][last_epoch_idx],
                    'best_epoch_val_f1': history['val_f1'][last_epoch_idx] if history['val_f1'] else np.nan,
                    'best_epoch_num': last_epoch_idx + 1
                }
            else: 
                 metrics_from_best_model_eval = {k: np.nan for k in ['best_epoch_train_loss', 'best_epoch_train_acc', 'best_epoch_val_loss', 'best_epoch_val_acc', 'best_epoch_val_f1', 'best_epoch_num']}
    else:
        print(f"Best model file {best_model_path} not found. No final evaluation metrics to report.")
        eval_model = model 
        metrics_from_best_model_eval = {k: np.nan for k in ['best_epoch_train_loss', 'best_epoch_train_acc', 'best_epoch_val_loss', 'best_epoch_val_acc', 'best_epoch_val_f1', 'best_epoch_num']}

    final_model_to_return = eval_model if eval_model is not None else model
    return final_model_to_return, scaler, best_val_metric, metrics_from_best_model_eval


if __name__ == '__main__':
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda" or config.DEVICE == "mps":
        if config.DEVICE == "cuda": torch.cuda.manual_seed_all(config.RANDOM_SEED)

    if config.K_FOLDS > 1:
        print(f"Starting {config.K_FOLDS}-Fold Cross-Validation Training for {config.MODEL_TYPE}...")
        for i in range(config.K_FOLDS):
            print(f"\n===== FOLD {i+1}/{config.K_FOLDS} =====")
            _, _, _, _ = main_train_loop(fold_num=i)
        print("\nCross-validation finished.")
    else:
        print(f"Starting Single Train/Validation/Test Training for {config.MODEL_TYPE}...")
        main_train_loop()