#!/usr/bin/env python3
"""
Improved model training script for lottery prediction system.
This script uses enhanced data splitting and training techniques 
to create more robust models that can generate better predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import logging

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_models import EnsembleTrainer
from scripts.fetch_data import load_data, DATA_DIR, prepare_sequence_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/improved_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define output directories
OUTPUT_DIR = Path("outputs")
TRAINING_DIR = OUTPUT_DIR / "training"
VALIDATION_DIR = OUTPUT_DIR / "validation"
MODELS_DIR = Path("models/checkpoints")

# Create all directories
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def split_data_with_rolling_window(X, y, val_size=0.15, test_size=0.1, n_splits=5):
    """
    Split data using a time series cross-validation approach with rolling windows.
    This method respects the temporal nature of lottery data and provides multiple
    training/validation splits for more robust model evaluation.
    """
    print(f"Splitting data using rolling window approach with {n_splits} splits")
    
    # Create TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Calculate total samples and sizes for final test set
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    
    # Reserve final portion for test set
    X_test = X[-test_samples:]
    y_test = y[-test_samples:]
    
    # Use the rest for rolling window CV
    X_train_val = X[:-test_samples]
    y_train_val = y[:-test_samples]
    
    # Generate splits
    cv_splits = []
    for train_idx, val_idx in tscv.split(X_train_val):
        X_train_split, X_val_split = X_train_val[train_idx], X_train_val[val_idx]
        y_train_split, y_val_split = y_train_val[train_idx], y_train_val[val_idx]
        cv_splits.append((X_train_split, y_train_split, X_val_split, y_val_split))
    
    # For detailed analysis, print split sizes
    for i, (X_tr, y_tr, X_vl, y_vl) in enumerate(cv_splits):
        print(f"Split {i+1}: Train size={len(X_tr)}, Validation size={len(X_vl)}")
    
    # Use the final split for the main training/validation
    X_train, y_train, X_val, y_val = cv_splits[-1]
    
    print(f"Final Training set: {X_train.shape}")
    print(f"Final Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'cv_splits': cv_splits
    }

def train_lstm_model(X_train, y_train, X_val, y_val, config=None):
    """Train an optimized LSTM model with the given configuration."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        
        # Default configuration with improved parameters
        if config is None:
            config = {
                'lstm_units_1': 128,
                'lstm_units_2': 64,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 500
            }
        
        print(f"Training LSTM model with configuration: {config}")
        
        # Build model architecture with increased complexity
        model = Sequential()
        model.add(LSTM(units=config['lstm_units_1'], return_sequences=True, 
                      input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))
        model.add(LSTM(units=config['lstm_units_2']))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(config['dropout'] / 2))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        
        # Compile with Adam optimizer and MSE loss
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), 
                     loss='mse', 
                     metrics=['mae'])
        
        # Define callbacks for better training
        checkpoint_path = TRAINING_DIR / f"lstm_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=20,
                min_lr=0.00001,
                verbose=1
            ),
            # Checkpoint to save best model
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with progress bar
        with tqdm(total=config['epochs'], desc="LSTM Training") as pbar:
            class TqdmCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{logs['loss']:.4f}",
                        'val_loss': f"{logs['val_loss']:.4f}"
                    })
            
            # Add tqdm callback to the list
            callbacks.append(TqdmCallback())
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
        
        # Load best model from checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from checkpoint: {checkpoint_path}")
            model = load_model(checkpoint_path)
        
        # Return trained model and training history
        return model, history
    
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
        return None, None

def train_xgboost_model(X_train, y_train, X_val, y_val, config=None):
    """Train an optimized XGBoost model with the given configuration."""
    try:
        import xgboost as xgb
        
        # Flatten input for tree-based models if needed
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val
        
        # Default configuration with improved parameters
        if config is None:
            config = {
                'n_estimators': 1000,
                'max_depth': 7,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        
        print(f"Training XGBoost model with configuration: {config}")
        
        # Create model with configuration
        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            min_child_weight=config.get('min_child_weight', 1),
            gamma=config.get('gamma', 0),
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 1),
            tree_method='hist',
            n_jobs=-1,
            verbosity=0
        )
        
        # Train with early stopping and progress tracking
        with tqdm(total=config['n_estimators'], desc="XGBoost Training") as pbar:
            # Custom callback to update progress bar
            class TqdmCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    pbar.update(1)
                    
                    # If we have validation metrics, display them
                    if evals_log:
                        val_metric = list(evals_log.values())[0]['rmse'][-1]
                        pbar.set_postfix({'val_rmse': f"{val_metric:.4f}"})
                    
                    return False  # Continue training
            
            # Train model with callback
            model.fit(
                X_train_flat, y_train,
                eval_set=[(X_val_flat, y_val)],
                eval_metric='rmse',
                early_stopping_rounds=50,
                callbacks=[TqdmCallback()],
                verbose=False
            )
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            print("\nTop 10 XGBoost Feature Importances:")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            for i in indices:
                print(f"Feature {i}: {importances[i]:.4f}")
        
        # Evaluate on validation set
        val_preds = model.predict(X_val_flat)
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        return None

def train_lightgbm_model(X_train, y_train, X_val, y_val, config=None):
    """Train an optimized LightGBM model with the given configuration."""
    try:
        import lightgbm as lgb
        
        # Flatten input for tree-based models if needed
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val
        
        # Default configuration with improved parameters
        if config is None:
            config = {
                'n_estimators': 1000,
                'num_leaves': 31,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'min_split_gain': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        
        print(f"Training LightGBM model with configuration: {config}")
        
        # Create model with configuration
        model = lgb.LGBMRegressor(
            n_estimators=config['n_estimators'],
            num_leaves=config['num_leaves'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            min_child_samples=config.get('min_child_samples', 20),
            min_split_gain=config.get('min_split_gain', 0),
            reg_alpha=config.get('reg_alpha', 0),
            reg_lambda=config.get('reg_lambda', 0),
            n_jobs=-1,
            verbose=-1
        )
        
        # Create datasets for LightGBM
        train_data = lgb.Dataset(X_train_flat, label=y_train)
        valid_data = lgb.Dataset(X_val_flat, label=y_val, reference=train_data)
        
        # Set up parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1
        }
        
        # Train with early stopping and progress tracking
        with tqdm(total=config['n_estimators'], desc="LightGBM Training") as pbar:
            # Custom callback to update progress bar
            def callback(env):
                pbar.update(1)
                if len(env.evaluation_result_list) > 0:
                    metric_name, val_metric = env.evaluation_result_list[0]
                    pbar.set_postfix({metric_name: f"{val_metric:.4f}"})
            
            # Train model with callback
            lgb_model = lgb.train(
                params,
                train_data,
                num_boost_round=config['n_estimators'],
                valid_sets=[valid_data],
                callbacks=[callback, lgb.early_stopping(50)]
            )
            
            # Convert from low-level to sklearn API for consistency
            model.booster_ = lgb_model
            model._n_features = X_train_flat.shape[1]
            model._n_classes = 1
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            print("\nTop 10 LightGBM Feature Importances:")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            for i in indices:
                print(f"Feature {i}: {importances[i]:.4f}")
        
        # Evaluate on validation set
        val_preds = model.predict(X_val_flat)
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training LightGBM model: {str(e)}")
        return None

def train_cnn_lstm_model(X_train, y_train, X_val, y_val):
    """Train a CNN-LSTM hybrid model for improved feature extraction."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        from tensorflow.keras.optimizers import Adam
        
        print("Training CNN-LSTM hybrid model...")
        
        # Check input dimensions
        print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
        
        # Build CNN-LSTM model
        model = Sequential()
        
        # CNN layers for feature extraction
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', 
                         input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        
        # LSTM layers for temporal dependencies
        model.add(LSTM(units=128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(units=64))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Dense layers for prediction
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Print model summary
        model.summary()
        
        # Define callbacks
        checkpoint_path = TRAINING_DIR / f"cnn_lstm_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=20,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with progress bar
        with tqdm(total=500, desc="CNN-LSTM Training") as pbar:
            class TqdmCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{logs['loss']:.4f}",
                        'val_loss': f"{logs['val_loss']:.4f}"
                    })
            
            callbacks.append(TqdmCallback())
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=500,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        
        # Load best model from checkpoint
        if os.path.exists(checkpoint_path):
            print(f"Loading best model from checkpoint: {checkpoint_path}")
            model = load_model(checkpoint_path)
        
        # Evaluate on validation set
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        
        return model, history
    
    except Exception as e:
        logger.error(f"Error training CNN-LSTM model: {str(e)}")
        return None, None

def main():
    """Main function to train improved models with optimal hyperparameters."""
    try:
        start_time = time.time()
        print("Starting improved model training...")
        
        # 1. Load and prepare data
        print("\n=== LOADING AND PREPARING DATA ===")
        df = load_data(DATA_DIR / "merged_lottery_data.csv")
        print(f"Loaded {len(df)} lottery records")
        
        # Prepare sequence data with enhanced features and longer sequence length
        print("Preparing sequence data with enhanced features...")
        sequences, targets = prepare_sequence_data(df, sequence_length=30, with_enhanced_features=True)
        print(f"Prepared {len(sequences)} sequence records with shape {sequences.shape}")
        
        # 2. Split data with rolling window approach
        print("\n=== SPLITTING DATA WITH ROLLING WINDOW APPROACH ===")
        data_splits = split_data_with_rolling_window(sequences, targets, val_size=0.15, test_size=0.1, n_splits=5)
        
        X_train, y_train = data_splits['train']
        X_val, y_val = data_splits['val']
        X_test, y_test = data_splits['test']
        
        # 3. Initialize trainer
        print("\n=== INITIALIZING TRAINER ===")
        trainer = EnsembleTrainer(X_train, y_train)
        
        # 4. Train individual models
        print("\n=== TRAINING INDIVIDUAL MODELS ===")
        models = {}
        
        # 4.1 Train LSTM model
        print("\n--- Training LSTM Model ---")
        lstm_model, lstm_history = train_lstm_model(X_train, y_train, X_val, y_val)
        if lstm_model is not None:
            models['lstm'] = lstm_model
            print("LSTM model training completed successfully")
        
        # 4.2 Train XGBoost model
        print("\n--- Training XGBoost Model ---")
        xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
        if xgb_model is not None:
            models['xgboost'] = xgb_model
            print("XGBoost model training completed successfully")
        
        # 4.3 Train LightGBM model
        print("\n--- Training LightGBM Model ---")
        lgb_model = train_lightgbm_model(X_train, y_train, X_val, y_val)
        if lgb_model is not None:
            models['lightgbm'] = lgb_model
            print("LightGBM model training completed successfully")
        
        # 4.4 Train CNN-LSTM model
        print("\n--- Training CNN-LSTM Model ---")
        cnn_lstm_model, cnn_lstm_history = train_cnn_lstm_model(X_train, y_train, X_val, y_val)
        if cnn_lstm_model is not None:
            models['cnn_lstm'] = cnn_lstm_model
            print("CNN-LSTM model training completed successfully")
        
        # 5. Update trainer with the trained models
        print(f"\n=== UPDATING TRAINER WITH {len(models)} TRAINED MODELS ===")
        trainer.models = models
        
        # 6. Build ensemble and optimize weights
        print("\n=== BUILDING ENSEMBLE MODEL ===")
        if len(models) >= 2:
            print("Optimizing ensemble weights...")
            trainer.optimize_ensemble_weights()
        else:
            print("Not enough models for ensemble, skipping ensemble creation")
        
        # 7. Validate final models on test set
        print("\n=== VALIDATING MODELS ON TEST SET ===")
        trainer.validate_models(X_test, y_test, output_dir=VALIDATION_DIR)
        
        # 8. Save trained models
        print("\n=== SAVING TRAINED MODELS ===")
        if trainer.save_trained_models(backup=True):
            print("Successfully saved trained models!")
        else:
            print("Error saving models!")
        
        # 9. Generate visualizations and save training metadata
        print("\n=== GENERATING VISUALIZATIONS AND METADATA ===")
        trainer.visualize_results(X_test, y_test, output_dir=VALIDATION_DIR)
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'training_duration': time.time() - start_time,
            'data_size': len(df),
            'sequence_length': 30,
            'feature_count': sequences.shape[2],
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'models_trained': list(models.keys()),
            'training_config': {
                'val_size': 0.15,
                'test_size': 0.1,
                'n_splits': 5,
                'enhanced_features': True
            }
        }
        
        # Save metadata to file
        metadata_file = TRAINING_DIR / f"training_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        duration = time.time() - start_time
        print(f"\nTraining completed in {duration:.2f} seconds")
        print(f"Trained models: {', '.join(models.keys())}")
        print(f"Training metadata saved to {metadata_file}")
        print("\nImproved models are ready for daily predictions")
        
    except Exception as e:
        logger.error(f"Error in main training function: {str(e)}")
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main() 