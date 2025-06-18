# model_30Hz/train_30Hz.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Import from the new 30Hz files
import config_30Hz as config
import data_loader_30Hz as data_loader
from model_30Hz import GaitLSTM30Hz
import utils_30Hz as utils

def train_epoch(model, data_loader, criterion, optimizer, device):
    # ... (No changes to this function's logic)
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
    # ... (No changes to this function's logic)
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
    metrics_dict = utils.calculate_metrics(all_targets_flat, all_predictions_flat, average='weighted')
    return avg_loss, accuracy, metrics_dict, all_targets_flat, all_predictions_flat


def main_train_loop(fold_num=None):
    print(f"\n--- Starting 30Hz Model Training Loop {'for Fold ' + str(fold_num) if fold_num is not None else ''} ---")
    print(f"Using device: {config.DEVICE}")

    class_weights_tensor = None
    if config.K_FOLDS > 1:
        if fold_num is None: raise ValueError("fold_num required for CV.")
        train_loader, val_loader, scaler, class_weights = data_loader.get_data_loaders_30Hz(fold_num=fold_num)
    else:
        train_loader, val_loader, _, scaler, class_weights = data_loader.get_data_loaders_30Hz()
    
    if config.USE_WEIGHTED_LOSS and class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(config.DEVICE)
        print(f"Using class weights for loss: {class_weights_tensor.cpu().numpy()}")
    
    if not train_loader or not val_loader:
        print("Error: Train or Validation loader is None. Aborting.")
        return

    model = GaitLSTM30Hz(
        input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
        bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=config.LSTM_DROPOUT,
        linear_dropout=config.LINEAR_DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) if config.USE_WEIGHTED_LOSS and class_weights_tensor is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_objective_metric = 0.0 if config.OPTIMIZE_METRIC == 'f1' else float('inf') 
    best_epoch_num = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    fold_model_name_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    best_model_filename = f"best_model{fold_model_name_suffix}.pth"
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, val_metrics_dict, _, _ = validate_epoch(model, val_loader, criterion, config.DEVICE)
        val_f1 = val_metrics_dict['f1_score']
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{epoch_duration:.2f}s] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        current_report_metric = val_f1 if config.OPTIMIZE_METRIC == 'f1' else val_loss
        is_best = False
        if config.OPTIMIZE_METRIC == 'f1':
            if current_report_metric > best_val_objective_metric + config.EARLY_STOPPING_DELTA: is_best = True
        else:
            if current_report_metric < best_val_objective_metric - config.EARLY_STOPPING_DELTA: is_best = True

        if is_best:
            best_val_objective_metric = current_report_metric
            best_epoch_num = epoch + 1
            epochs_no_improve = 0
            utils.save_checkpoint({
                'epoch': best_epoch_num, 'state_dict': model.state_dict(),
                'best_val_objective_metric': best_val_objective_metric,
                'scaler': scaler,
                'class_weights_used': class_weights_tensor.cpu().numpy() if class_weights_tensor is not None else None
            }, is_best=True, filename="checkpoint.pth", best_filename=best_model_filename, output_dir=config.TRAIN_OUTPUT_DIR)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (metric: {config.OPTIMIZE_METRIC}).")
            break
            
    print("Training loop finished for this run/fold.")
    utils.plot_training_history(history, fold_num=fold_num)

    best_model_path = os.path.join(config.TRAIN_OUTPUT_DIR, best_model_filename)
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path} for final validation...")
        eval_model = GaitLSTM30Hz(
             input_size=config.NUM_FEATURES, hidden_size=config.LSTM_HIDDEN_SIZE,
             num_layers=config.NUM_LSTM_LAYERS, num_classes=config.NUM_CLASSES,
             bidirectional=config.BIDIRECTIONAL_LSTM, lstm_dropout=0, linear_dropout=0
        ).to(config.DEVICE)
        checkpoint = utils.load_checkpoint(best_model_path, eval_model, optimizer=None)
        if checkpoint:
            val_loss_eval, val_acc_eval, val_metrics_dict_eval, val_targets_flat, val_preds_flat = validate_epoch(eval_model, val_loader, criterion, config.DEVICE)
            print(f"Final Eval on Best Model - Val Loss: {val_loss_eval:.4f}, Val Acc: {val_acc_eval:.4f}, Val F1 (weighted): {val_metrics_dict_eval['f1_score']:.4f}")
            class_names = [f"Phase {i}" for i in range(config.NUM_CLASSES)]
            utils.plot_confusion_matrix_custom(val_targets_flat, val_preds_flat, class_names, 
                                               title="Validation Confusion Matrix (Best Model)", fold_num=fold_num)

if __name__ == '__main__':
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "mps": torch.mps.manual_seed(config.RANDOM_SEED) if hasattr(torch.mps, 'manual_seed') else None
    elif config.DEVICE == "cuda": torch.cuda.manual_seed_all(config.RANDOM_SEED)

    if config.K_FOLDS > 1:
        print(f"Starting {config.K_FOLDS}-Fold Cross-Validation Training for 30Hz Model...")
        for i in range(config.K_FOLDS):
            print(f"\n===== FOLD {i+1}/{config.K_FOLDS} =====")
            main_train_loop(fold_num=i) 
        print("\nCross-validation finished.")
    else:
        print("Starting Single Train/Validation/Test Training for 30Hz Model...")
        main_train_loop()