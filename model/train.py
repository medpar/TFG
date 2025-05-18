# train.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score # For per-epoch accuracy

import config
import data_loader
from model import GaitLSTM
import utils

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_targets_flat = []
    all_predictions_flat = []

    for features, targets in tqdm(data_loader, desc="Training", leave=False):
        features = features.to(device)
        targets = targets.to(device) # Shape: (batch_size, seq_len)

        optimizer.zero_grad()
        outputs = model(features) # Shape: (batch_size, seq_len, num_classes)
        
        # Reshape for CrossEntropyLoss:
        # Outputs: (batch_size * seq_len, num_classes)
        # Targets: (batch_size * seq_len)
        loss = criterion(outputs.view(-1, config.NUM_CLASSES), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # For accuracy calculation
        _, predicted_classes = torch.max(outputs, 2) # Get class with max logit for each timestep
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
    metrics = utils.calculate_metrics(all_targets_flat, all_predictions_flat)
    
    return avg_loss, accuracy, metrics, all_targets_flat, all_predictions_flat

def main_train_loop(fold_num=None):
    """
    Main training and validation loop.
    If fold_num is provided, runs for a specific fold of cross-validation.
    Otherwise, runs a single train/val/test split.
    """
    print(f"\n--- Starting Training Loop {'for Fold ' + str(fold_num) if fold_num is not None else ''} ---")
    print(f"Using device: {config.DEVICE}")

    # --- Data Loading ---
    if config.K_FOLDS > 1:
        if fold_num is None: raise ValueError("fold_num required for CV.")
        train_loader, val_loader, scaler = data_loader.get_data_loaders(fold_num=fold_num)
        test_loader = None # Test set is implicitly the fold held out, or handled separately after CV
    else:
        train_loader, val_loader, test_loader, scaler = data_loader.get_data_loaders()
    
    if not train_loader or not val_loader:
        print("Error: Train or Validation loader is None. Aborting.")
        return

    # --- Model, Criterion, Optimizer ---
    model = GaitLSTM(
        input_size=config.NUM_FEATURES,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS,
        num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM,
        lstm_dropout=config.LSTM_DROPOUT,
        linear_dropout=config.LINEAR_DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss() # Handles softmax internally
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)


    # --- Training ---
    best_val_metric = float('inf') # Using validation loss for best model
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    fold_model_name_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    checkpoint_filename = f"checkpoint{fold_model_name_suffix}.pth"
    best_model_filename = f"best_model{fold_model_name_suffix}.pth"


    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_metrics, _, _ = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        # scheduler.step(val_loss)

        epoch_duration = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{epoch_duration:.2f}s] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1_score']:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # --- Checkpoint and Early Stopping ---
        current_val_metric = val_loss # Using validation loss
        is_best = current_val_metric < best_val_metric

        if is_best:
            best_val_metric = current_val_metric
            epochs_no_improve = 0
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                'scaler': scaler # Save the scaler used for this fold/training
            }, is_best=True, filename=checkpoint_filename, best_filename=best_model_filename)
        else:
            epochs_no_improve += 1
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_metric': best_val_metric, # Still store the best seen so far
                'scaler': scaler
            }, is_best=False, filename=checkpoint_filename) # Save current state anyway

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break
            
    # --- Post-Training ---
    print("Training finished.")
    utils.plot_training_history(history, fold_num=fold_num)

    # Load best model for final evaluation on validation set (and test set if applicable)
    print(f"\nLoading best model from {os.path.join(config.OUTPUT_DIR, best_model_filename)} for final evaluation...")
    best_checkpoint = utils.load_checkpoint(os.path.join(config.OUTPUT_DIR, best_model_filename), model, optimizer)
    if best_checkpoint is None:
        print("Could not load best model. Using last model state.")
    
    print("\n--- Final Validation Set Evaluation (using best model) ---")
    val_loss, val_acc, val_metrics, val_targets_flat, val_preds_flat = validate_epoch(model, val_loader, criterion, config.DEVICE)
    print(f"Best Model Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Best Model Val Metrics: {val_metrics}")
    class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
    utils.plot_confusion_matrix_custom(val_targets_flat, val_preds_flat, class_names, title="Validation Confusion Matrix", fold_num=fold_num)

    # If not doing CV, and test_loader exists, evaluate on test set
    if config.K_FOLDS <= 1 and test_loader:
        print("\n--- Final Test Set Evaluation (using best model) ---")
        test_loss, test_acc, test_metrics, test_targets_flat, test_preds_flat = validate_epoch(model, test_loader, criterion, config.DEVICE)
        print(f"Best Model Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Best Model Test Metrics: {test_metrics}")
        utils.plot_confusion_matrix_custom(test_targets_flat, test_preds_flat, class_names, title="Test Confusion Matrix")
        
        # Save test predictions to a CSV if needed
        # This part can be expanded based on inference.py logic for output format

    return model, scaler # Return the trained model and scaler for this run/fold


if __name__ == '__main__':
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    if config.K_FOLDS > 1:
        print(f"Starting {config.K_FOLDS}-Fold Cross-Validation Training...")
        # Store metrics for each fold if you want to average later
        all_fold_val_metrics = [] 
        for i in range(config.K_FOLDS):
            print(f"\n===== FOLD {i+1}/{config.K_FOLDS} =====")
            # For CV, we typically don't test after each fold, but aggregate results
            # The `main_train_loop` will handle validation for the fold.
            _, _ = main_train_loop(fold_num=i)
            # If you need to collect validation metrics from each fold:
            # model, scaler = main_train_loop(fold_num=i)
            # val_loss, val_acc, val_metrics, _, _ = validate_epoch(model, val_loader_for_fold, criterion, config.DEVICE)
            # all_fold_val_metrics.append(val_metrics)
        print("\nCross-validation finished.")
        # Add code here to average metrics across folds if desired
    else:
        print("Starting Single Train/Validation/Test Training...")
        main_train_loop()