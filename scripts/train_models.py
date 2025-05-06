"""
Training implementation for lottery prediction models.

This file contains the implementation code for training lottery models:
1. EnsembleTrainer class for training and managing multiple models
2. Methods for training individual model types
3. Training utility functions and evaluation metrics
4. Model persistence (save/load) functionality
5. Validation, visualization and interpretation of trained models

This file IMPLEMENTS TRAINING LOGIC, while models/training_config.py 
DEFINES the configuration parameters used by these implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pickle
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
import warnings
import json
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import optuna
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Attention, Reshape, Flatten
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm, trange
import random

from scripts.utils import setup_logging, log_memory_usage, LOG_DIR
from scripts.utils.model_utils import log_training_errors
from scripts.analyze_data import (
    analyze_lottery_data,
    analyze_patterns,
    analyze_correlations,
    analyze_randomness,
    analyze_spatial_distribution,
    analyze_range_frequency,
    get_prediction_weights,
    identify_patterns,
    find_consecutive_pairs,
    get_hot_cold_numbers
)
from scripts.fetch_data import (
    parse_balls,
    enhance_features,
    load_data,
    prepare_training_data,
    prepare_feature_data,
    prepare_sequence_data,
    split_data,
    get_latest_draw,
    download_new_data,
    merge_data_files
)
from scripts.performance_tracking import (
    calculate_metrics,
    calculate_top_k_accuracy,
    track_model_performance,
    log_performance_trends,
    get_model_weights,
    normalize_weights,
    get_default_weights,
    get_model_performance_summary,
    calculate_ensemble_weights,
    calculate_diversity_bonus
)
from scripts.model_bridge import (
    ensure_valid_prediction,
    import_prediction_function,
    predict_with_model,
    score_combinations,
    ensemble_prediction,
    monte_carlo_simulation,
    save_predictions,
    predict_next_draw,
    calculate_prediction_metrics,
    calculate_metrics,
    backtest,
    get_model_weights
)

# Import model training functions
from models.lstm_model import train_lstm_model, predict_lstm_model
from models.arima_model import train_arima_model
from models.holtwinters_model import train_holtwinters_model
from models.linear_model import train_linear_models, predict_linear_model
from models.xgboost_model import train_xgboost_model, predict_xgboost_model
from models.lightgbm_model import train_lightgbm_model, predict_lightgbm_model
from models.knn_model import train_knn_model, predict_knn_model
from models.gradient_boosting_model import train_gradient_boosting_model, predict_gradient_boosting_model
from models.catboost_model import train_catboost_model, predict_catboost_model
from models.cnn_lstm_model import train_cnn_lstm_model
from models.autoencoder_model import train_autoencoder_model
from models.meta_model import train_meta_model
from models.ensemble import LotteryEnsemble

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path("models/checkpoints")
MODELS_PATH = MODELS_DIR / "trained_models.pkl"

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Model configurations
LSTM_CONFIG = {
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dense_units': 16,
    'dropout_rate': 0.2,
    'l2_reg': 0.01,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}

LIGHTGBM_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0,
    'reg_lambda': 1
}

class EnsembleTrainer:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """Initialize EnsembleTrainer.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        # Validate input shapes
        if X_train is not None and y_train is not None:
            if len(X_train) != len(y_train):
                logger.warning(f"Input arrays have different lengths: X_train ({len(X_train)}) vs y_train ({len(y_train)})")
                # Trim to the shorter length
                min_len = min(len(X_train), len(y_train))
                self.X_train = X_train[:min_len]
                self.y_train = y_train[:min_len]
                logger.info(f"Trimmed inputs to length {min_len}")
            else:
                self.X_train = X_train
                self.y_train = y_train
        else:
            # Allow initialization without data (for loading pre-trained models)
            self.X_train = None
            self.y_train = None
            
        self.models = {}
        self.ensemble = None
        self.is_trained = False
        self.fallback_to_simple_models = True
        
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: str, n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna for the given model type.
        
        Args:
            X: Training features
            y: Training targets
            model_type: Type of model ('lstm', 'xgboost', or 'lightgbm')
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of optimized hyperparameters
        """
        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
            import numpy as np
            from scripts.utils.memory_monitor import log_memory_usage
            
            # Log initial state
            log_memory_usage(f"Hyperparameter optimization - {model_type} - start")
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            def objective_lstm(trial):
                """Objective function for LSTM hyperparameter optimization."""
                # Define hyperparameters to optimize
                lstm_units_1 = trial.suggest_categorical('lstm_units_1', [64, 128, 256])
                lstm_units_2 = trial.suggest_categorical('lstm_units_2', [32, 64, 128])
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64])
                l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
                
                # Cross-validation scores
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create model
                    model = tf.keras.Sequential([
                        tf.keras.layers.LSTM(lstm_units_1, input_shape=(X.shape[1], X.shape[2]), 
                                             return_sequences=True, 
                                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
                        tf.keras.layers.Dropout(dropout_rate),
                        tf.keras.layers.LSTM(lstm_units_2),
                        tf.keras.layers.Dropout(dropout_rate),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(y.shape[1], activation='sigmoid')
                    ])
                    
                    # Compile model
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                    
                    # Train with early stopping
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=10, restore_best_weights=True
                    )
                    
                    model.fit(
                        X_train, y_train,
                        epochs=50,  # Lower epochs for faster trials
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Evaluate
                    _, mae = model.evaluate(X_val, y_val, verbose=0)
                    cv_scores.append(mae)
                    
                    # Clean up
                    tf.keras.backend.clear_session()
                
                return np.mean(cv_scores)
            
            def objective_xgboost(trial):
                """Objective function for XGBoost hyperparameter optimization."""
                import xgboost as xgb
                
                # Define hyperparameters to optimize
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0)
                }
                
                # Cross-validation scores
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create DMatrix
                    dtrain = xgb.DMatrix(X_train, y_train)
                    dval = xgb.DMatrix(X_val, y_val)
                    
                    # XGBoost parameters
                    xgb_params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'max_depth': params['max_depth'],
                        'learning_rate': params['learning_rate'],
                        'subsample': params['subsample'],
                        'colsample_bytree': params['colsample_bytree'],
                        'min_child_weight': params['min_child_weight'],
                        'gamma': params['gamma'],
                        'verbosity': 0
                    }
                    
                    # Train model
                    model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=params['n_estimators'],
                        early_stopping_rounds=20,
                        evals=[(dval, 'val')],
                        verbose_eval=False
                    )
                    
                    # Evaluate
                    preds = model.predict(dval)
                    mae = np.mean(np.abs(preds - y_val))
                    cv_scores.append(mae)
                    
                    # Clean up
                    del dtrain, dval, model
                    import gc
                    gc.collect()
                
                return np.mean(cv_scores)
            
            def objective_lightgbm(trial):
                """Objective function for LightGBM hyperparameter optimization."""
                import lightgbm as lgb
                
                # Define hyperparameters to optimize
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
                }
                
                # Cross-validation scores
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create LightGBM datasets
                    train_data = lgb.Dataset(X_train, label=y_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    
                    # LightGBM parameters
                    lgb_params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'num_leaves': params['num_leaves'],
                        'learning_rate': params['learning_rate'],
                        'feature_fraction': params['colsample_bytree'],
                        'bagging_fraction': params['subsample'],
                        'bagging_freq': 5,
                        'min_child_samples': params['min_child_samples'],
                        'verbose': -1
                    }
                    
                    # Train model
                    model = lgb.train(
                        lgb_params,
                        train_data,
                        num_boost_round=params['n_estimators'],
                        early_stopping_rounds=20,
                        valid_sets=[val_data],
                        verbose_eval=False
                    )
                    
                    # Evaluate
                    preds = model.predict(X_val)
                    mae = np.mean(np.abs(preds - y_val))
                    cv_scores.append(mae)
                    
                    # Clean up
                    del train_data, val_data, model
                    import gc
                    gc.collect()
                
                return np.mean(cv_scores)
            
            # Select the appropriate objective function
            if model_type == 'lstm':
                objective = objective_lstm
            elif model_type == 'xgboost':
                objective = objective_xgboost
            elif model_type == 'lightgbm':
                objective = objective_lightgbm
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create a study object and optimize
            study = optuna.create_study(direction='minimize')
            
            # Reduce verbosity during trials
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Run optimization
            print(f"Running {n_trials} trials for {model_type} hyperparameter optimization...")
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            print(f"Best {model_type} parameters: {best_params}")
            print(f"Best value: {study.best_value:.4f}")
            
            # Log final state
            log_memory_usage(f"Hyperparameter optimization - {model_type} - end")
            
            return best_params
            
        except ImportError:
            print("Optuna not available. Using default parameters.")
            
            # Return default parameters based on model type
            if model_type == 'lstm':
                return {
                    'lstm_units_1': 128,
                    'lstm_units_2': 64,
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'l2_reg': 0.001
                }
            elif model_type == 'xgboost':
                return {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0
                }
            elif model_type == 'lightgbm':
                return {
                    'n_estimators': 500,
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")
    
    def train_lstm_model(self, X=None, y=None):
        """Train an LSTM model for lottery number prediction."""
        try:
            if X is None:
                X = self.X_train
            if y is None:
                y = self.y_train
                
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from tensorflow.keras.optimizers import Adam
            from tqdm import tqdm
            import numpy as np
            import time
            
            # Start timing
            start_time = time.time()
            print("  Building LSTM model architecture...")
            
            # Check input shape
            if len(X.shape) != 3:
                print(f"  Reshaping input from {X.shape} to 3D for LSTM")
                # Assuming X is (samples, features) - reshape to (samples, 1, features)
                X = X.reshape(X.shape[0], 1, -1)
            
            # Create LSTM model
            model = Sequential()
            model.add(LSTM(units=100, return_sequences=True, 
                          input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=100))
            model.add(Dropout(0.2))
            model.add(Dense(y.shape[1], activation='sigmoid'))  # Output shape matches y
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='mse', 
                         metrics=['mae'])
            
            # Early stopping and LR reduction
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=20,
                min_lr=0.00001,
                verbose=1
            )
            
            # Create a tqdm callback
            class TqdmCallback(tf.keras.callbacks.Callback):
                def __init__(self, epochs):
                    self.epochs = epochs
                    self.pbar = None
                    
                def on_train_begin(self, logs=None):
                    print("  Starting LSTM training...")
                    self.pbar = tqdm(total=self.epochs, desc="  LSTM Training", 
                                   position=0, leave=True, unit="epoch")
                    
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    loss = logs.get('loss', 0)
                    val_loss = logs.get('val_loss', 0)
                    desc = f"  LSTM Training - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}"
                    self.pbar.set_description(desc)
                    self.pbar.update(1)
                    
                def on_train_end(self, logs=None):
                    self.pbar.close()
            
            # Train model with progress display
            print("  Starting LSTM training with validation split...")
            
            batch_size = 32
            epochs = 300  # We'll rely on early stopping, but allow more epochs
            
            history = model.fit(
                X, y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[
                    early_stopping, 
                    reduce_lr,
                    TqdmCallback(epochs)
                ],
                verbose=0  # Disable default verbose output
            )
            
            # Calculate training duration
            duration = time.time() - start_time
            print(f"  ✅ LSTM training completed in {duration:.2f} seconds")
            print(f"  Final loss: {history.history['loss'][-1]:.4f}")
            print(f"  Best validation loss: {min(history.history['val_loss']):.4f}")
            
            # Monitor memory
            if hasattr(self, 'memory_monitor') and self.memory_monitor:
                from scripts.utils.memory_monitor import log_memory_usage
                log_memory_usage("After LSTM training")
            
            return model
        except Exception as e:
            print(f"  ❌ Error training LSTM model: {str(e)}")
            if self.fallback_to_simple_models:
                print("  Falling back to a simpler LSTM model")
                return self.train_simple_lstm_model(X, y)
            raise
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train XGBoost model with optimized hyperparameters and memory monitoring."""
        try:
            # Try the simpler implementation first
            return self.train_simple_xgboost_model(X, y)
        except Exception as e:
            print(f"  ❌ Error training simple XGBoost model: {str(e)}")
            print("  Falling back to native XGBoost implementation")
            return self.train_native_xgboost_model(X, y)
    
    def train_simple_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train XGBoost using the sklearn API which is more robust to different data shapes."""
        import xgboost as xgb
        from scripts.utils.memory_monitor import log_memory_usage
        
        try:
            print("  Building simple XGBoost model...")
            
            # Flatten input if it's 3D (for sequence data)
            if len(X.shape) == 3:
                print(f"  Reshaping input from {X.shape} to 2D for XGBoost")
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
                
            # Make sure targets are in the right format
            if len(y.shape) > 1 and y.shape[1] > 1:
                print(f"  Multi-output regression with {y.shape[1]} outputs. Training for first output only.")
                y_flat = y[:, 0]
            else:
                y_flat = y.ravel() if len(y.shape) > 1 else y
                
            # Verify shapes before training
            print(f"  Input shape: {X_flat.shape}, Target shape: {y_flat.shape}")
            
            # Create progress reporting
            from tqdm import tqdm
            
            # Set parameters - increase n_estimators for more thorough training
            n_estimators = 300  # Increased from 100
            
            # Train the model with progress tracking manually
            print("  Training XGBoost model...")
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                n_jobs=-1,
                verbosity=0
            )
            
            # Use tqdm to show a progress bar during training
            with tqdm(total=n_estimators, desc="  XGBoost Training", position=0) as pbar:
                # Train in smaller chunks to show progress
                for i in range(0, n_estimators, 10):
                    # Determine how many estimators to train in this step
                    step_estimators = min(10, n_estimators - i)
                    
                    # Create temporary model for this step
                    step_model = xgb.XGBRegressor(
                        n_estimators=step_estimators,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        tree_method='hist',
                        n_jobs=-1,
                        verbosity=0
                    )
                    
                    # Train on current step
                    step_model.fit(X_flat, y_flat)
                    
                    # Update progress bar
                    pbar.update(step_estimators)
                    
                    # If this is the first step, use it as our main model
                    if i == 0:
                        model = step_model
                    else:
                        # For later steps, we need to train a new full model
                        # This is inefficient but necessary since we can't easily combine models
                        new_model = xgb.XGBRegressor(
                            n_estimators=i + step_estimators,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method='hist',
                            n_jobs=-1,
                            verbosity=0
                        )
                        new_model.fit(X_flat, y_flat)
                        model = new_model
            
            print("  ✅ XGBoost training completed.")
            
            # Display feature importance
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            print("\n  Top feature importance:")
            for i in range(min(10, len(indices))):
                print(f"    Feature {indices[i]}: {importance[indices[i]]:.4f}")
                
            return model
            
        except Exception as e:
            print(f"  ❌ Error in simple XGBoost training: {str(e)}")
            # Create a very simple model as fallback
            try:
                print("  Attempting fallback to very simple XGBoost model...")
                model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
                model.fit(X_flat, y_flat)
                print("  ✅ Simple fallback XGBoost model trained successfully.")
                return model
            except Exception as inner_e:
                print(f"  ❌ Fallback also failed: {str(inner_e)}")
                raise
    
    def train_native_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train XGBoost using the native API with optimized hyperparameters and memory monitoring."""
        import xgboost as xgb
        from scripts.utils.memory_monitor import log_memory_usage, get_memory_monitor
        
        # Start memory monitoring
        monitor = get_memory_monitor(log_dir="logs/memory", interval=5.0)
        monitor.start()
        
        try:
            # Log initial memory state
            log_memory_usage("XGBoost training - initial state")
            
            # Flatten input if it's 3D (for sequence data)
            if len(X.shape) == 3:
                print(f"Flattening 3D input of shape {X.shape} for XGBoost")
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
                
            print(f"Training XGBoost model with {X_flat.shape[0]} samples and {X_flat.shape[1]} features")
            
            # Hyperparameter tuning
            use_hyperparameter_tuning = True
            if use_hyperparameter_tuning:
                try:
                    print("Optimizing hyperparameters for XGBoost model...")
                    best_params = self.optimize_hyperparameters(X_flat, y, 'xgboost')
                    
                    # Extract optimized parameters
                    n_estimators = best_params.get('n_estimators', 500)
                    max_depth = best_params.get('max_depth', 6)
                    learning_rate = best_params.get('learning_rate', 0.01)
                    subsample = best_params.get('subsample', 0.8)
                    colsample_bytree = best_params.get('colsample_bytree', 0.8)
                    min_child_weight = best_params.get('min_child_weight', 1)
                    gamma = best_params.get('gamma', 0)
                    
                    print(f"Using optimized hyperparameters: n_estimators={n_estimators}, "
                          f"max_depth={max_depth}, learning_rate={learning_rate}")
                except Exception as e:
                    print(f"Error during hyperparameter optimization: {str(e)}. Using default parameters.")
                    # Default parameters
                    n_estimators = 500
                    max_depth = 6
                    learning_rate = 0.01
                    subsample = 0.8
                    colsample_bytree = 0.8
                    min_child_weight = 1
                    gamma = 0
            else:
                # Default parameters
                n_estimators = 500
                max_depth = 6
                learning_rate = 0.01
                subsample = 0.8
                colsample_bytree = 0.8
                min_child_weight = 1
                gamma = 0
                
            # Log memory before model creation
            log_memory_usage("XGBoost training - before model creation")
            
            # Cross-validation for robust training
            from sklearn.model_selection import TimeSeriesSplit
            time_series_cv = TimeSeriesSplit(n_splits=5)
            last_split = list(time_series_cv.split(X_flat))[-1]
            train_idx, val_idx = last_split
            X_train_cv, X_val_cv = X_flat[train_idx], X_flat[val_idx]
            
            # Make sure y has the right shape
            if len(y.shape) > 2:
                print(f"Reshaping targets from {y.shape} for XGBoost")
                y_flat = y.reshape(y.shape[0], -1)
            else:
                y_flat = y
                
            y_train_cv, y_val_cv = y_flat[train_idx], y_flat[val_idx]
            
            # Verify shapes
            print(f"X_train_cv shape: {X_train_cv.shape}, y_train_cv shape: {y_train_cv.shape}")
            print(f"X_val_cv shape: {X_val_cv.shape}, y_val_cv shape: {y_val_cv.shape}")
            
            # Make sure the arrays have the same length
            min_train_len = min(len(X_train_cv), len(y_train_cv))
            min_val_len = min(len(X_val_cv), len(y_val_cv))
            
            if len(X_train_cv) != len(y_train_cv) or len(X_val_cv) != len(y_val_cv):
                print(f"Warning: Length mismatch between X and y. Trimming to match.")
                X_train_cv = X_train_cv[:min_train_len]
                y_train_cv = y_train_cv[:min_train_len]
                X_val_cv = X_val_cv[:min_val_len]
                y_val_cv = y_val_cv[:min_val_len]
            
            print(f"Training on {X_train_cv.shape[0]} samples, validating on {X_val_cv.shape[0]} samples")
            
            # Create DMatrix objects (more memory efficient)
            print("Converting to DMatrix format...")
            try:
                # For multi-output regression, handle each output separately
                if y_train_cv.shape[1] > 1:
                    print(f"Multi-output regression with {y_train_cv.shape[1]} outputs. Training first output only.")
                    dtrain = xgb.DMatrix(X_train_cv, y_train_cv[:, 0])
                    dval = xgb.DMatrix(X_val_cv, y_val_cv[:, 0])
                else:
                    dtrain = xgb.DMatrix(X_train_cv, y_train_cv)
                    dval = xgb.DMatrix(X_val_cv, y_val_cv)
            except Exception as e:
                print(f"Error creating DMatrix: {str(e)}. Trying with simplified approach.")
                # If we're still having issues, try a really simplified approach
                dtrain = xgb.DMatrix(X_train_cv, y_train_cv[:, 0] if y_train_cv.shape[1] > 1 else y_train_cv)
                dval = xgb.DMatrix(X_val_cv, y_val_cv[:, 0] if y_val_cv.shape[1] > 1 else y_val_cv)
            
            # Log memory after DMatrix creation
            log_memory_usage("XGBoost training - after DMatrix creation")
            
            # Set up parameters
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'gamma': gamma,
                'tree_method': 'hist',  # For faster training
                'nthread': -1  # Use all CPU cores
            }
            
            # Custom callback to update tqdm progress bar
            class TqdmCallback(object):
                def __init__(self, n_estimators):
                    self.pbar = tqdm(total=n_estimators, desc="XGBoost Training", position=1)
                    self.curr_iteration = 0
                    self.best_score = float('inf')
                
                def __call__(self, env):
                    # Only update on new iterations
                    if env.iteration > self.curr_iteration:
                        iterations_to_update = env.iteration - self.curr_iteration
                        self.pbar.update(iterations_to_update)
                        self.curr_iteration = env.iteration
                        
                        # Extract metrics
                        try:
                            # Handle the evaluation_result_list format correctly
                            if hasattr(env, 'evaluation_result_list') and env.evaluation_result_list:
                                for eval_result in env.evaluation_result_list:
                                    if isinstance(eval_result, tuple) and len(eval_result) >= 3:
                                        dataset, metric, value = eval_result[0], eval_result[1], eval_result[2]
                                        if dataset == 'val' and metric == 'rmse':
                                            # Update if we have a new best model
                                            if value < self.best_score:
                                                self.best_score = value
                                            self.pbar.set_postfix(val_rmse=f"{value:.4f}", best=f"{self.best_score:.4f}")
                        except Exception as e:
                            # Continue silently if there's any issue with metrics extraction
                            pass
                    
                    # Close on completion
                    if env.iteration >= n_estimators - 1:
                        self.pbar.close()
            
            # Train model with early stopping and progress tracking
            print(f"Training XGBoost model with {n_estimators} estimators...")
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False,  # Disable default verbosity
                callbacks=[TqdmCallback(n_estimators)]
            )
            
            # Log final memory state
            log_memory_usage("XGBoost training - after training")
            
            # Feature importance analysis
            importance = model.get_score(importance_type='gain')
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("\nTop feature importance:")
            for feat, score in sorted_importance[:10]:
                print(f"  {feat}: {score:.4f}")
            
            # Memory cleanup
            del dtrain, dval
            import gc
            gc.collect()
            
            return model
            
        finally:
            # Stop memory monitoring
            monitor.stop()
            print("Memory monitoring for XGBoost completed.")
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train LightGBM model with optimized hyperparameters and memory monitoring."""
        try:
            # Try the simpler implementation first
            return self.train_simple_lightgbm_model(X, y)
        except Exception as e:
            print(f"  ❌ Error training simple LightGBM model: {str(e)}")
            print("  Falling back to native LightGBM implementation")
            return self.train_native_lightgbm_model(X, y)
    
    def train_simple_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train LightGBM using the sklearn API which is more robust to different data shapes."""
        import lightgbm as lgb
        from scripts.utils.memory_monitor import log_memory_usage
        
        try:
            print("  Building simple LightGBM model...")
            
            # Flatten input if it's 3D (for sequence data)
            if len(X.shape) == 3:
                print(f"  Reshaping input from {X.shape} to 2D for LightGBM")
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
                
            # Make sure targets are in the right format
            if len(y.shape) > 1 and y.shape[1] > 1:
                print(f"  Multi-output regression with {y.shape[1]} outputs. Training for first output only.")
                y_flat = y[:, 0]
            else:
                y_flat = y.ravel() if len(y.shape) > 1 else y
                
            # Verify shapes before training
            print(f"  Input shape: {X_flat.shape}, Target shape: {y_flat.shape}")
            
            # Create progress reporting
            from tqdm import tqdm
            
            # Set parameters - increase n_estimators for more thorough training
            n_estimators = 300  # Increased from 100
            
            # Train the model with progress tracking manually
            print("  Training LightGBM model...")
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbose=-1
            )
            
            # Use tqdm to show a progress bar during training
            with tqdm(total=n_estimators, desc="  LightGBM Training", position=0) as pbar:
                # Train in smaller chunks to show progress
                for i in range(0, n_estimators, 10):
                    # Determine how many estimators to train in this step
                    step_estimators = min(10, n_estimators - i)
                    
                    # Create temporary model for this step
                    step_model = lgb.LGBMRegressor(
                        n_estimators=step_estimators,
                        max_depth=6,
                        learning_rate=0.1,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=-1,
                        verbose=-1
                    )
                    
                    # Train on current step
                    step_model.fit(X_flat, y_flat)
                    
                    # Update progress bar
                    pbar.update(step_estimators)
                    
                    # If this is the first step, use it as our main model
                    if i == 0:
                        model = step_model
                    else:
                        # For later steps, we need to train a new full model
                        # This is inefficient but necessary since we can't easily combine models
                        new_model = lgb.LGBMRegressor(
                            n_estimators=i + step_estimators,
                            max_depth=6,
                            learning_rate=0.1,
                            num_leaves=31,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=-1,
                            verbose=-1
                        )
                        new_model.fit(X_flat, y_flat)
                        model = new_model
            
            print("  ✅ LightGBM training completed.")
            
            # Display feature importance
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            print("\n  Top feature importance:")
            for i in range(min(10, len(indices))):
                print(f"    Feature {indices[i]}: {importance[indices[i]]:.4f}")
                
            return model
            
        except Exception as e:
            print(f"  ❌ Error in simple LightGBM training: {str(e)}")
            # Create a very simple model as fallback
            try:
                print("  Attempting fallback to very simple LightGBM model...")
                model = lgb.LGBMRegressor(n_estimators=10, max_depth=3)
                model.fit(X_flat, y_flat)
                print("  ✅ Simple fallback LightGBM model trained successfully.")
                return model
            except Exception as inner_e:
                print(f"  ❌ Fallback also failed: {str(inner_e)}")
                raise
    
    def train_native_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train LightGBM using the native API with optimized hyperparameters and memory monitoring."""
        import lightgbm as lgb
        from scripts.utils.memory_monitor import log_memory_usage, get_memory_monitor
        
        # Start memory monitoring
        monitor = get_memory_monitor(log_dir="logs/memory", interval=5.0)
        monitor.start()
        
        try:
            # Log initial memory state
            log_memory_usage("LightGBM training - initial state")
            
            # Flatten input if it's 3D (for sequence data)
            if len(X.shape) == 3:
                print(f"Flattening 3D input of shape {X.shape} for LightGBM")
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
                
            print(f"Training LightGBM model with {X_flat.shape[0]} samples and {X_flat.shape[1]} features")
            
            # Hyperparameter tuning
            use_hyperparameter_tuning = True
            if use_hyperparameter_tuning:
                try:
                    print("Optimizing hyperparameters for LightGBM model...")
                    best_params = self.optimize_hyperparameters(X_flat, y, 'lightgbm')
                    
                    # Extract optimized parameters
                    n_estimators = best_params.get('n_estimators', 500)
                    num_leaves = best_params.get('num_leaves', 31)
                    learning_rate = best_params.get('learning_rate', 0.01)
                    subsample = best_params.get('subsample', 0.8)
                    colsample_bytree = best_params.get('colsample_bytree', 0.8)
                    min_child_samples = best_params.get('min_child_samples', 20)
                    
                    print(f"Using optimized hyperparameters: n_estimators={n_estimators}, "
                          f"num_leaves={num_leaves}, learning_rate={learning_rate}")
                except Exception as e:
                    print(f"Error during hyperparameter optimization: {str(e)}. Using default parameters.")
                    # Default parameters
                    n_estimators = 500
                    num_leaves = 31
                    learning_rate = 0.01
                    subsample = 0.8
                    colsample_bytree = 0.8
                    min_child_samples = 20
            else:
                # Default parameters
                n_estimators = 500
                num_leaves = 31
                learning_rate = 0.01
                subsample = 0.8
                colsample_bytree = 0.8
                min_child_samples = 20
                
            # Log memory before model creation
            log_memory_usage("LightGBM training - before model creation")
            
            # Cross-validation for robust training
            from sklearn.model_selection import TimeSeriesSplit
            time_series_cv = TimeSeriesSplit(n_splits=5)
            last_split = list(time_series_cv.split(X_flat))[-1]
            train_idx, val_idx = last_split
            X_train_cv, X_val_cv = X_flat[train_idx], X_flat[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            print(f"Training on {X_train_cv.shape[0]} samples, validating on {X_val_cv.shape[0]} samples")
            
            # Create LightGBM datasets
            print("Converting to LightGBM Dataset format...")
            train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
            valid_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
            
            # Log memory after dataset creation
            log_memory_usage("LightGBM training - after Dataset creation")
            
            # Set up parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': colsample_bytree,
                'bagging_fraction': subsample,
                'bagging_freq': 5,
                'min_child_samples': min_child_samples,
                'verbose': -1,
                'n_jobs': -1  # Use all CPU cores
            }
            
            # Custom callback for tqdm progress
            class TqdmCallback(object):
                def __init__(self, n_estimators):
                    self.pbar = tqdm(total=n_estimators, desc="LightGBM Training", position=1)
                    self.curr_iteration = 0
                    self.best_score = float('inf')
                
                def __call__(self, env):
                    # Only update on new iterations
                    if env.iteration > self.curr_iteration:
                        iterations_to_update = env.iteration - self.curr_iteration
                        self.pbar.update(iterations_to_update)
                        self.curr_iteration = env.iteration
                        
                        # Update metrics in progress bar
                        if 'valid_0' in env.evaluation_result_list:
                            eval_name, eval_result = env.evaluation_result_list['valid_0']
                            if eval_result < self.best_score:
                                self.best_score = eval_result
                            self.pbar.set_postfix(val_rmse=f"{eval_result:.4f}", best=f"{self.best_score:.4f}")
                    
                    # Close on completion
                    if env.iteration >= n_estimators - 1:
                        self.pbar.close()
                    
                    # Continue training
                    return False
            
            # Train model with early stopping and progress display
            print(f"Training LightGBM model with {n_estimators} estimators...")
            callbacks = [
                lgb.callback.log_evaluation(period=0),  # Disable default logging
                TqdmCallback(n_estimators)
            ]
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[valid_data, train_data],
                valid_names=['val', 'train'],
                early_stopping_rounds=50,
                callbacks=callbacks
            )
            
            # Log final memory state
            log_memory_usage("LightGBM training - after training")
            
            # Feature importance analysis
            importance = model.feature_importance(importance_type='gain')
            feature_names = model.feature_name()
            sorted_idx = np.argsort(importance)[::-1]
            print("\nTop feature importance:")
            for i in range(min(10, len(feature_names))):
                idx = sorted_idx[i]
                print(f"  {feature_names[idx]}: {importance[idx]:.4f}")
            
            # Memory cleanup
            del train_data, valid_data
            import gc
            gc.collect()
        
            return model
            
        finally:
            # Stop memory monitoring
            monitor.stop()
            print("Memory monitoring for LightGBM completed.")
    
    def train_all_models(self) -> None:
        """Train all models sequentially with progress tracking."""
        try:
            # Keep track of models
            self.models = {}
            models_trained = []
            
            # Check if we have training data
            if self.X_train is None or self.y_train is None:
                print("Error: No training data available")
                return
            
            # List of models to train
            model_list = [
                ('lstm', self.train_lstm_model),
                ('xgboost', self.train_simple_xgboost_model),
                ('lightgbm', self.train_simple_lightgbm_model),
                ('cnn_lstm', self.train_cnn_lstm_model)
            ]
            
            # Create a progress bar for overall training
            print("\n=== STARTING SEQUENTIAL MODEL TRAINING ===")
            with tqdm(total=len(model_list), desc="Overall progress") as pbar:
                # Train each model
                for i, (model_name, train_func) in enumerate(model_list, 1):
                    print(f"[{i}/{len(model_list)}] STARTING {model_name.upper()} MODEL TRAINING")
                    start_time = time.time()
                    
                    try:
                        # Prepare input data based on model type
                        if model_name in ('lstm', 'cnn_lstm'):
                            # These models need 3D data
                            model = train_func(self.X_train, self.y_train)
                        else:
                            # These models need 2D data
                            X_2d = self.X_train.reshape(self.X_train.shape[0], -1)
                            model = train_func(X_2d, self.y_train)
                        
                        if model is not None:
                            self.models[model_name] = model
                            models_trained.append(model_name)
                            
                        duration = time.time() - start_time
                        print(f"[{i}/{len(model_list)}] COMPLETED {model_name.upper()} MODEL TRAINING in {duration:.2f} seconds")
                    except Exception as e:
                        print(f"Error training {model_name} model: {str(e)}")
                        traceback.print_exc()
                    
                    # Update progress
                    pbar.update(1)
            
            # If we have at least one trained model, create an ensemble
            if self.models:
                print("Creating ensemble with trained models...")
                try:
                    # Set default weights (equal weighting)
                    weights = {name: 1.0 / len(self.models) for name in self.models}
                    
                    # Create the ensemble
                    from models.ensemble import LotteryEnsemble
                    self.ensemble = LotteryEnsemble(self.models, weights)
                    print(f"Created ensemble with {len(models_trained)} models: {', '.join(models_trained)}")
                except Exception as e:
                    print(f"Error in train_all_models: {str(e)}")
                    traceback.print_exc()
                
                # Save models
                self.save_trained_models()
            else:
                print("No models were successfully trained.")
                
        except Exception as e:
            print(f"Error training models: {str(e)}")
            traceback.print_exc()
            if self.models:
                print(f"Saving {len(self.models)} trained models despite error...")
                self.save_trained_models()
    
    def save_trained_models(self, backup=True):
        """Save trained models to disk with backup and error handling."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
            
            # Create a backup of existing models if requested
            if backup and os.path.exists(MODELS_PATH):
                backup_path = str(MODELS_PATH) + ".backup"
                try:
                    import shutil
                    shutil.copy2(MODELS_PATH, backup_path)
                    print(f"  Created backup of previous models at {backup_path}")
                except Exception as backup_err:
                    print(f"  Warning: Could not create backup: {str(backup_err)}")
            
            # Save the models
            with open(MODELS_PATH, 'wb') as f:
                pickle.dump(self.models, f)
                
            # Also save ensemble separately if available
            if self.ensemble:
                ensemble_path = os.path.join(os.path.dirname(MODELS_PATH), "ensemble.pkl")
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.ensemble, f)
                
            print(f"  Successfully saved {len(self.models)} models to {MODELS_PATH}")
            return True
            
        except Exception as e:
            print(f"  Error saving models: {str(e)}")
            
            # Try to save to a different location
            try:
                import time
                alt_path = os.path.join(os.path.dirname(MODELS_PATH), f"models_emergency_{int(time.time())}.pkl")
                with open(alt_path, 'wb') as f:
                    pickle.dump(self.models, f)
                print(f"  Saved emergency backup to {alt_path}")
            except Exception as alt_err:
                print(f"  Fatal error: Could not save anywhere: {str(alt_err)}")
                
            return False
    
    def validate_models(self, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path = None) -> None:
        """Validate models on validation data."""
        print("Validating models...")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate individual models
        all_metrics = {}
        for name, model in self.models.items():
            if name == 'lstm':
                X = X_val
            else:
                X = X_val.reshape(X_val.shape[0], -1)
            
            y_pred = model.predict(X)
            metrics = calculate_prediction_metrics(y_val, y_pred)
            print(f"{name} model metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            all_metrics[name] = metrics
        
        # Validate ensemble
        y_pred = self.ensemble.predict(X_val)
        metrics = calculate_prediction_metrics(y_val, y_pred)
        print("Ensemble model metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        all_metrics['ensemble'] = metrics
        
        # Save metrics if output directory is provided
        if output_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = Path(output_dir) / f"validation_metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': all_metrics,
                    'data_size': len(X_val)
                }, f, indent=2)
    
    def monitor_performance(self, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path = None) -> None:
        """Monitor model performance."""
        print("Monitoring performance...")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize backtest metrics dictionary
        backtest_metrics = {}
        
        # If ensemble exists, perform backtesting
        if hasattr(self, 'ensemble') and self.ensemble is not None:
            try:
                backtest_metrics = backtest(self.ensemble, X_val, y_val)
                print("Backtest metrics:")
                for metric_name, value in backtest_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
            except Exception as e:
                print(f"Error in ensemble backtesting: {str(e)}")
                
                # Try individual models instead
                print("Trying individual model backtesting...")
                for model_name, model in self.models.items():
                    try:
                        print(f"Backtesting {model_name} model...")
                        if model_name in ['lstm', 'cnn_lstm']:
                            X = X_val  # 3D input
                        else:
                            X = X_val.reshape(X_val.shape[0], -1)  # Flatten
                            
                        y_pred = model.predict(X)
                        metrics = calculate_prediction_metrics(y_val, y_pred)
                        backtest_metrics[model_name] = metrics
                        print(f"  {model_name} metrics:")
                        for metric_name, value in metrics.items():
                            print(f"    {metric_name}: {value:.4f}")
                    except Exception as model_e:
                        print(f"Error backtesting {model_name}: {str(model_e)}")
        else:
            print("No ensemble model available. Trying individual models...")
            # No ensemble, try individual models
            for model_name, model in self.models.items():
                try:
                    print(f"Backtesting {model_name} model...")
                    if model_name in ['lstm', 'cnn_lstm']:
                        X = X_val  # 3D input
                    else:
                        X = X_val.reshape(X_val.shape[0], -1)  # Flatten
                        
                    y_pred = model.predict(X)
                    metrics = calculate_prediction_metrics(y_val, y_pred)
                    backtest_metrics[model_name] = metrics
                    print(f"  {model_name} metrics:")
                    for metric_name, value in metrics.items():
                        print(f"    {metric_name}: {value:.4f}")
                except Exception as model_e:
                    print(f"Error backtesting {model_name}: {str(model_e)}")
        
        # Save monitoring outputs/results if output directory is provided
        if output_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            monitoring_file = Path(output_dir) / f"performance_metrics_{timestamp}.json"
            
            with open(monitoring_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'backtest_metrics': backtest_metrics,
                    'data_size': len(X_val)
                }, f, indent=2)
    
    def interpret_models(self, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path = None) -> None:
        """Interpret model predictions."""
        print("Interpreting models...")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get feature importance for tree-based models
        X_2d = X_val.reshape(X_val.shape[0], -1)
        feature_names = [f"feature_{i}" for i in range(X_2d.shape[1])]
        
        # Dictionary to store feature importances
        feature_importances = {}
        
        for name, model in self.models.items():
            if name in ['xgboost', 'lightgbm']:
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'get_score'):
                        importances = model.get_score(importance_type='gain')
                    elif hasattr(model, 'layers') and len(model.layers) > 0:
                        # Try getting weights from the last layer
                        importances = model.layers[-1].get_weights()[0][:, 0]
                    else:
                        print(f"Cannot get feature importance for {name}, no suitable method found")
                        continue
                        
                    if isinstance(importances, dict):
                        # Convert dictionary to array
                        imp_array = np.zeros(len(feature_names))
                        for key, value in importances.items():
                            try:
                                idx = int(key.replace('f', ''))
                                if idx < len(imp_array):
                                    imp_array[idx] = value
                            except ValueError:
                                continue
                        importances = imp_array
                    
                    indices = np.argsort(importances)[::-1]
                    print(f"\n{name} feature importance:")
                    
                    model_importances = {}
                    # Only show up to 10 features or as many as we have
                    top_n = min(10, len(indices))
                    for f in range(top_n):
                        if f < len(indices) and indices[f] < len(feature_names):
                            feature_name = feature_names[indices[f]]
                            importance = float(importances[indices[f]])
                            print(f"  {feature_name}: {importance:.4f}")
                            model_importances[feature_name] = importance
                    
                    feature_importances[name] = model_importances
                except Exception as e:
                    print(f"Error getting feature importance for {name}: {str(e)}")
                    continue
        
        # Save interpretation outputs/results if output directory is provided
        if output_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            interpretation_file = Path(output_dir) / f"feature_importance_{timestamp}.json"
            
            with open(interpretation_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'feature_importances': feature_importances,
                    'data_size': len(X_val)
                }, f, indent=2)
    
    def visualize_results(self, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path = None) -> None:
        """Visualize model results."""
        print("Visualizing results...")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = Path('outputs/results')
            os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions - either from ensemble or individual models
        if hasattr(self, 'ensemble') and self.ensemble is not None:
            try:
                y_pred = self.ensemble.predict(X_val)
                print("Generated ensemble predictions for visualization")
            except Exception as e:
                print(f"Error generating ensemble predictions: {str(e)}")
                # Fallback to first available model
                model_name = next(iter(self.models.keys())) if self.models else None
                if model_name:
                    try:
                        model = self.models[model_name]
                        if model_name in ['lstm', 'cnn_lstm']:
                            y_pred = model.predict(X_val)
                        else:
                            y_pred = model.predict(X_val.reshape(X_val.shape[0], -1))
                        print(f"Generated {model_name} predictions as fallback")
                    except Exception as model_e:
                        print(f"Error generating {model_name} predictions: {str(model_e)}")
                        # Last resort - use dummy predictions
                        y_pred = np.zeros_like(y_val)
                        print("Using zero predictions as last resort")
                else:
                    # No models available
                    y_pred = np.zeros_like(y_val)
                    print("No models available. Using zero predictions.")
        else:
            # No ensemble, try first available model
            model_name = next(iter(self.models.keys())) if self.models else None
            if model_name:
                try:
                    model = self.models[model_name]
                    if model_name in ['lstm', 'cnn_lstm']:
                        y_pred = model.predict(X_val)
                    else:
                        y_pred = model.predict(X_val.reshape(X_val.shape[0], -1))
                    print(f"Generated {model_name} predictions in absence of ensemble")
                except Exception as model_e:
                    print(f"Error generating {model_name} predictions: {str(model_e)}")
                    # Last resort - use dummy predictions
                    y_pred = np.zeros_like(y_val)
                    print("Using zero predictions as last resort")
            else:
                # No models available
                y_pred = np.zeros_like(y_val)
                print("No models available. Using zero predictions.")
        
        # Ensure y_pred has the right shape
        if len(y_pred.shape) == 1 and len(y_val.shape) > 1:
            # Reshape to match y_val
            y_pred = np.expand_dims(y_pred, axis=1)
            if y_val.shape[1] > 1:
                y_pred = np.repeat(y_pred, y_val.shape[1], axis=1)
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val.flatten(), y_pred.flatten(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.savefig(Path(output_dir) / 'actual_vs_predicted.png')
        plt.close()
        
        # Plot prediction error distribution
        errors = y_val.flatten() - y_pred.flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Prediction Error Distribution')
        plt.savefig(Path(output_dir) / 'error_distribution.png')
        plt.close()
        
        # Save visualization metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = Path(output_dir) / f"visualization_meta_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'plots_generated': [
                    'actual_vs_predicted.png',
                    'error_distribution.png'
                ],
                'data_size': len(X_val)
            }, f, indent=2)

    def load_trained_models(self, models_path: str = 'models/checkpoints/trained_models.pkl') -> bool:
        """Load trained models from disk.
        
        Args:
            models_path: Path to saved models file
            
        Returns:
            True if models were successfully loaded, False otherwise
        """
        try:
            if os.path.exists(models_path):
                with open(models_path, 'rb') as f:
                    saved_models = pickle.load(f)
                
                self.models = saved_models
                
                # Initialize ensemble with loaded models
                weights = self.get_model_weights()
                self.ensemble = LotteryEnsemble(models=self.models, weights=weights)
                
                self.is_trained = True
                logging.info(f"Successfully loaded {len(self.models)} models from {models_path}")
                return True
            else:
                logging.warning(f"Models file not found at {models_path}")
                return False
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get weights for each model in the ensemble based on their performance.
        Includes all trained models with appropriate weights.
        
        Returns:
            Dictionary mapping model names to their weights
        """
        # Try to load performance history
        try:
            with open('outputs/outputs/results/performance_log.json', 'r') as f:
                performance = json.load(f)
            
            # Calculate weights based on inverse error
            weights = {}
            for model_name, metrics in performance.items():
                # Use latest MAE as performance metric
                if 'mae' in metrics and metrics['mae']:
                    error = metrics['mae'][-1]
                    # Lower error = higher weight
                    weights[model_name] = 1.0 / (error + 1e-5)
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                for model_name in weights:
                    weights[model_name] /= total
                return weights
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Could not load performance log: {str(e)}")
        
        # Default weights if performance log not available or error occurred
        weights = {
            # Primary models with specified weights
            'lstm': 0.30,
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'cnn_lstm': 0.20,
            
            # Secondary models (only used if available)
            'catboost': 0.00,
            'gradient_boosting': 0.00,
            'arima': 0.00,
            'holtwinters': 0.00,
            'knn': 0.00
        }
        
        # Dynamic adjustment: If primary models are missing, redistribute to available models
        primary_models = ['lstm', 'xgboost', 'lightgbm', 'cnn_lstm']
        available_primary = [m for m in primary_models if m in self.models]
        missing_primary = [m for m in primary_models if m not in self.models]
        
        # If any primary models are missing, redistribute their weights
        if missing_primary and available_primary:
            missing_weight = sum(weights[m] for m in missing_primary)
            weight_per_model = missing_weight / len(available_primary)
            for m in available_primary:
                weights[m] += weight_per_model
                
        # Only keep weights for models that exist in self.models
        weights = {k: v for k, v in weights.items() if k in self.models}
        
        # Re-normalize if needed
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 1e-5:  # Only normalize if not close to 1.0
            for k in weights:
                weights[k] /= total
        
        return weights
        
    def get_model_insights(self) -> Dict[str, Any]:
        """
        Get insights about the trained models, their performance and interpretations.
        
        Returns:
            Dictionary with model insights
        """
        # Initialize insights dictionary
        insights = {
            'performance': {},
            'interpretation': {},
            'metrics': {},
            'validation': {},
            'models': list(self.models.keys()) if self.models else []
        }
        
        # If no models are available, return basic insights
        if not self.models:
            insights['status'] = "No trained models available"
            return insights
            
        # Get model weights
        try:
            weights = self.get_model_weights()
            insights['weights'] = weights
        except Exception as e:
            insights['weights'] = {"error": str(e)}
        
        # Try to load performance history
        try:
            performance_path = Path('outputs/outputs/results/performance_log.json')
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    performance = json.load(f)
                insights['performance'] = performance
            else:
                insights['performance']['status'] = "No performance log file found"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            insights['performance']['status'] = f"No performance log available: {str(e)}"
        
        # Try to load interpretations
        try:
            interpretation_dir = Path('outputs/interpretations')
            if interpretation_dir.exists():
                interpretation_files = list(interpretation_dir.glob('feature_importance_*.json'))
                if interpretation_files:
                    latest_file = max(interpretation_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        interpretations = json.load(f)
                    insights['interpretation'] = interpretations
                else:
                    insights['interpretation']['status'] = "No interpretation files found"
            else:
                insights['interpretation']['status'] = "Interpretation directory not found"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            insights['interpretation']['status'] = f"No interpretation data available: {str(e)}"
        
        # Try to load validation metrics
        try:
            validation_dir = Path('outputs/validation')
            if validation_dir.exists():
                validation_files = list(validation_dir.glob('validation_*.json'))
                if validation_files:
                    latest_file = max(validation_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        validation = json.load(f)
                    insights['validation'] = validation
                else:
                    insights['validation']['status'] = "No validation files found"
            else:
                insights['validation']['status'] = "Validation directory not found"
        except (FileNotFoundError, json.JSONDecodeError) as e:
            insights['validation']['status'] = f"No validation data available: {str(e)}"
        
        # Add basic model metrics if available
        for name, model in self.models.items():
            try:
                # Get basic info about the model
                model_info = {
                    'type': type(model).__name__,
                }
                
                # Try to get parameters if available
                if hasattr(model, 'get_params'):
                    try:
                        params = model.get_params()
                        # Convert any non-serializable values to strings
                        serializable_params = {}
                        for param_name, param_value in params.items():
                            try:
                                # Test JSON serialization
                                json.dumps({param_name: param_value})
                                serializable_params[param_name] = param_value
                            except (TypeError, OverflowError):
                                # If not serializable, convert to string
                                serializable_params[param_name] = str(param_value)
                        model_info['parameters'] = serializable_params
                    except Exception as param_err:
                        model_info['parameters'] = f"Error getting parameters: {str(param_err)}"
                        
                insights['metrics'][name] = model_info
            except Exception as e:
                insights['metrics'][name] = {'error': str(e)}
        
        return insights

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions with all models in the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}
        
        # Ensure we have trained models
        if not self.is_trained and not self.models:
            logger.warning("No trained models available for prediction")
            return predictions
        
        try:
            # Get predictions from individual models
            for model_name, model in self.models.items():
                try:
                    # Handle different input shapes for different models
                    if model_name == 'lstm' or model_name == 'cnn_lstm':
                        # For LSTM models, ensure input is 3D
                        if len(X.shape) < 3:
                            # If 2D, reshape to 3D
                            X_reshaped = X.reshape(X.shape[0], -1, 6)  # Assuming 6 features
                        else:
                            X_reshaped = X
                        pred = model.predict(X_reshaped)
                    else:
                        # For other models, ensure input is 2D
                        if len(X.shape) > 2:
                            # If 3D, flatten to 2D
                            X_flat = X.reshape(X.shape[0], -1)
                        else:
                            X_flat = X
                        pred = model.predict(X_flat)
                    
                    # Add to predictions dictionary
                    predictions[model_name] = pred
                    
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {str(e)}")
            
            # Also add ensemble prediction if available
            if self.ensemble is not None:
                try:
                    predictions['ensemble'] = self.ensemble.predict(X)
                except Exception as e:
                    logger.error(f"Error predicting with ensemble: {str(e)}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {}

    def optimize_ensemble_weights(self):
        """Optimize the weights for the ensemble based on validation performance."""
        if not self.models or len(self.models) <= 1:
            print("  Need at least 2 models to optimize ensemble weights")
            return
        
        try:
            print("  Optimizing ensemble weights based on validation performance...")
            
            # Create a small validation set from training data
            train_size = int(0.8 * len(self.X_train))
            X_opt = self.X_train[train_size:]
            y_opt = self.y_train[train_size:]
            
            # Get predictions from each model
            model_predictions = {}
            for model_name, model in self.models.items():
                try:
                    if model_name in ['lstm', 'cnn_lstm']:
                        # Use 3D input for these models
                        preds = model.predict(X_opt)
                    else:
                        # Flatten input for tree-based models
                        X_flat = X_opt.reshape(X_opt.shape[0], -1)
                        preds = model.predict(X_flat)
                    
                    model_predictions[model_name] = preds
                except Exception as e:
                    print(f"  Error getting predictions from {model_name}: {str(e)}")
            
            # If we have predictions from multiple models, optimize weights
            if len(model_predictions) > 1:
                # Start with equal weights
                initial_weights = {name: 1.0/len(model_predictions) for name in model_predictions}
                
                # Calculate errors for each model
                model_errors = {}
                for name, preds in model_predictions.items():
                    mse = np.mean((preds - y_opt) ** 2)
                    model_errors[name] = mse
                
                # Calculate weights inversely proportional to error
                total_inverse_error = sum(1.0/err for err in model_errors.values())
                weights = {name: (1.0/err)/total_inverse_error for name, err in model_errors.items()}
                
                # Round and normalize weights
                weights = {name: round(weight, 3) for name, weight in weights.items()}
                total = sum(weights.values())
                weights = {name: weight/total for name, weight in weights.items()}
                
                # Set the optimized weights
                if self.ensemble:
                    self.ensemble.set_weights(weights)
                
                # Log the weights
                print("  Optimized ensemble weights:")
                for name, weight in weights.items():
                    print(f"    {name}: {weight:.3f}")
            else:
                print("  Not enough model predictions for weight optimization")
                
        except Exception as e:
            print(f"  Error in ensemble weight optimization: {str(e)}")
            print("  Using default weights")
            
            # Try to save to a different location
            try:
                import time
                alt_path = os.path.join(os.path.dirname(MODELS_PATH), f"models_emergency_{int(time.time())}.pkl")
                with open(alt_path, 'wb') as f:
                    pickle.dump(self.models, f)
                print(f"  Saved emergency backup to {alt_path}")
            except Exception as alt_err:
                print(f"  Fatal error: Could not save anywhere: {str(alt_err)}")
                
            return False

    def train_cnn_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train a CNN-LSTM model with progress bar."""
        try:
            # Import TensorFlow here to avoid importing it if we don't need it
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Reshape, Flatten
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from tensorflow.keras.optimizers import Adam
            from tqdm import tqdm
            import time
            
            # Start timing
            start_time = time.time()
            print("  Building CNN-LSTM model architecture...")
            
            # Ensure correct input shape
            if len(X.shape) != 3:
                print(f"  Reshaping input from {X.shape} to 3D for CNN-LSTM")
                # Reshape if needed
                if len(X.shape) == 2:
                    # Try to infer time steps for reshaping
                    if X.shape[1] % 6 == 0:  # Assuming 6 features per time step
                        time_steps = X.shape[1] // 6
                        X = X.reshape(X.shape[0], time_steps, 6)
                    else:
                        # Default reshape
                        X = X.reshape(X.shape[0], -1, 1)
            
            # Get input dimensions
            samples, time_steps, features = X.shape
            print(f"  Input shape: samples={samples}, time_steps={time_steps}, features={features}")
            
            # Add a preprocessing layer to ensure 4D input for Conv1D
            # Conv1D expects shape (batch, steps, channels)
            model = Sequential()
            
            # Add reshape layer if needed
            if features == 1:
                # If we have only 1 feature, we need to add a step dimension
                model.add(Reshape((time_steps, features, 1), input_shape=(time_steps, features)))
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, features)))
            else:
                # Normal case - use features as channels
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, features)))
            
            # Continue with regular layers
            model.add(MaxPooling1D(pool_size=2))
            
            # Make sure we don't have too many pooling operations for short sequences
            if time_steps >= 8:
                model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
                model.add(MaxPooling1D(pool_size=2))
            
            # LSTM layers
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32))
            model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(y.shape[1], activation='sigmoid'))  # Output shape matches y
            
            # Print model summary
            model.summary()
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Setup callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
            
            # Train model
            epochs = 100
            batch_size = min(32, samples)  # Make sure batch size isn't larger than sample count
            
            # Use tqdm to show a progress bar during training
            print("  Training CNN-LSTM model...")
            with tqdm(total=epochs, desc="  CNN-LSTM Training", position=0) as pbar:
                # Train in epochs
                for epoch in range(epochs):
                    # Train for one epoch
                    history = model.fit(
                        X, y,
                        batch_size=batch_size,
                        epochs=1,
                        validation_split=0.2,
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0
                    )
                    
                    # Update progress bar with metrics
                    loss = history.history['loss'][0]
                    val_loss = history.history.get('val_loss', [0])[0]
                    pbar.set_description(f"  CNN-LSTM - Loss: {loss:.4f}, Val: {val_loss:.4f}")
                    pbar.update(1)
                    
                    # Check if early stopping triggered
                    if hasattr(early_stopping, 'stopped_epoch') and early_stopping.stopped_epoch > 0:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            # Calculate training duration
            duration = time.time() - start_time
            print(f"  ✅ CNN-LSTM training completed in {duration:.2f} seconds")
            
            return model
            
        except Exception as e:
            print(f"  ❌ Error training CNN-LSTM model: {str(e)}")
            # Try a simple fallback model
            try:
                print("  Attempting simpler CNN-LSTM model as fallback...")
                # Get input dimensions
                samples, time_steps, features = X.shape
                
                # Create a simpler model
                model = Sequential([
                    # Just use a single LSTM layer without Conv1D
                    LSTM(32, input_shape=(time_steps, features)),
                    Dense(y.shape[1], activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
                print("  ✅ Simple LSTM trained successfully as fallback")
                return model
            except Exception as inner_e:
                print(f"  ❌ Fallback model failed: {str(inner_e)}")
                return None
                
    def validate_models(self, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path = None) -> None:
        """Validate models on validation data."""
        print("Validating models...")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate individual models
        all_metrics = {}
        for name, model in self.models.items():
            try:
                if name == 'lstm' or name == 'cnn_lstm':
                    # Use 3D input for these models
                    X = X_val
                else:
                    # Flatten input for tree-based models
                    X = X_val.reshape(X_val.shape[0], -1)
                
                # Get predictions
                y_pred = model.predict(X)
                
                # Ensure prediction has the right shape
                if len(y_pred.shape) == 1 and len(y_val.shape) > 1:
                    # Expand dimensions to match y_val
                    y_pred = np.expand_dims(y_pred, axis=1)
                    
                    # If y_val has more than one column, repeat y_pred to match
                    if y_val.shape[1] > 1:
                        y_pred = np.repeat(y_pred, y_val.shape[1], axis=1)
                
                # Calculate metrics
                metrics = calculate_prediction_metrics(y_val, y_pred)
                print(f"{name} model metrics:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                
                all_metrics[name] = metrics
            except Exception as e:
                print(f"Error validating {name} model: {str(e)}")
        
        # Validate ensemble if available
        if hasattr(self, 'ensemble') and self.ensemble is not None:
            try:
                y_pred = self.ensemble.predict(X_val)
                metrics = calculate_prediction_metrics(y_val, y_pred)
                print("Ensemble model metrics:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                all_metrics['ensemble'] = metrics
            except Exception as e:
                print(f"Error validating ensemble model: {str(e)}")
        
        # Save metrics if output directory is provided
        if output_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = Path(output_dir) / f"validation_metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {k: {metric: float(value) for metric, value in v.items()} 
                               for k, v in all_metrics.items()},
                    'data_size': len(X_val)
                }, f, indent=2)

    def predict_next_numbers(self, num_predictions=10, confidence_level=0.9, use_monte_carlo=True, normalize=True):
        """
        Generate multiple predictions for the next lottery draw with confidence level.
        
        Args:
            num_predictions (int): Number of predictions to generate
            confidence_level (float): Confidence level for the predictions (0-1)
            use_monte_carlo (bool): Whether to use Monte Carlo simulation for predictions
            normalize (bool): Whether to normalize predictions to valid lottery numbers
            
        Returns:
            List of predictions with confidence scores
        """
        try:
            # Get the last available draws
            if not hasattr(self, 'latest_data') or self.latest_data is None:
                print("No training data available for predictions. Loading from data cache.")
                # Try to load from data cache
                try:
                    if os.path.exists('outputs/results/data_cache.pkl'):
                        with open('outputs/results/data_cache.pkl', 'rb') as f:
                            cached_data = pickle.load(f)
                            if isinstance(cached_data, dict) and 'X' in cached_data:
                                self.latest_data = cached_data['X'][-10:]  # Get last 10 draws
                            elif isinstance(cached_data, list) and len(cached_data) > 0:
                                self.latest_data = np.array(cached_data[-10:])  # Get last 10 draws
                            else:
                                print("No suitable data found in cache")
                                # Generate dummy data
                                self.latest_data = np.random.random((10, 6))
                except Exception as e:
                    print(f"Error loading data from cache: {e}")
                    # Generate dummy data as fallback
                    self.latest_data = np.random.random((10, 6))
                
            # Ensure we have some data
            if self.latest_data is None or len(self.latest_data) == 0:
                print("No data available for prediction, generating random numbers")
                return self._generate_monte_carlo_predictions(num_predictions)
                
            # Extract last draws
            last_draws = self.latest_data
            
            # Normalize if needed
            if normalize and hasattr(self, 'scaler') and self.scaler is not None:
                # Scale the data
                try:
                    last_draws_scaled = self.scaler.transform(last_draws)
                except Exception as e:
                    print(f"Error scaling data: {e}")
                    last_draws_scaled = last_draws
            else:
                last_draws_scaled = last_draws
                
            # Format data for different model types
            if hasattr(self, 'ensemble') and self.ensemble is not None:
                # Try ensemble prediction
                try:
                    # Prepare input for LSTM
                    if len(last_draws_scaled.shape) == 2:
                        # For LSTM models, reshape to 3D
                        lstm_input = last_draws_scaled.reshape(1, last_draws_scaled.shape[0], last_draws_scaled.shape[1])
                    else:
                        lstm_input = last_draws_scaled
                        
                    # Make prediction with ensemble
                    predictions = []
                    for _ in range(num_predictions):
                        # Get base prediction
                        if use_monte_carlo:
                            # Monte Carlo - add some random noise to input
                            noise = np.random.normal(0, 0.05, lstm_input.shape)
                            noisy_input = lstm_input + noise
                            # Clip to ensure valid range (0-1)
                            noisy_input = np.clip(noisy_input, 0, 1)
                            raw_prediction = self.ensemble.predict(noisy_input)
                        else:
                            raw_prediction = self.ensemble.predict(lstm_input)
                            
                        # Ensure the prediction has the right shape
                        if len(raw_prediction.shape) > 1 and raw_prediction.shape[0] == 1:
                            raw_prediction = raw_prediction[0]
                            
                        # Calculate confidence score
                        confidence = confidence_level * (1 - np.std(raw_prediction))
                            
                        # Scale back to original range if needed
                        if normalize and hasattr(self, 'scaler') and self.scaler is not None:
                            try:
                                # Convert to valid lottery numbers
                                scaled_prediction = self.scaler.inverse_transform(raw_prediction.reshape(1, -1))[0]
                            except Exception as e:
                                print(f"Error in inverse scaling: {e}")
                                scaled_prediction = raw_prediction
                        else:
                            scaled_prediction = raw_prediction
                            
                        # Append prediction with confidence
                        predictions.append({
                            'prediction': scaled_prediction.tolist(),
                            'confidence': float(confidence)
                        })
                        
                    return predictions
                except Exception as e:
                    print(f"Error making ensemble prediction: {e}")
                    # Fall back to individual models
            
            # Try with individual models if ensemble failed or is not available
            if hasattr(self, 'models') and self.models:
                try:
                    predictions = []
                    # Use each model to make a prediction
                    for model_name, model in self.models.items():
                        if model is None:
                            continue
                            
                        # Skip certain model types
                        if model_name.lower() in ['autoregressive', 'arima', 'prophet']:
                            continue
                            
                        # Prepare input based on model type
                        if model_name in ['LSTM', 'CNN_LSTM']:
                            # For LSTM models, reshape to 3D
                            if len(last_draws_scaled.shape) == 2:
                                model_input = last_draws_scaled.reshape(1, last_draws_scaled.shape[0], last_draws_scaled.shape[1])
                            else:
                                model_input = last_draws_scaled
                        else:
                            # For other models, flatten to 2D
                            model_input = last_draws_scaled.reshape(1, -1)
                        
                        # Make prediction
                        try:
                            raw_prediction = model.predict(model_input)
                            
                            # Ensure the prediction has the right shape
                            if len(raw_prediction.shape) > 1 and raw_prediction.shape[0] == 1:
                                raw_prediction = raw_prediction[0]
                                
                            # Calculate confidence score
                            confidence = confidence_level * 0.8  # Slightly lower confidence for individual models
                                
                            # Scale back to original range if needed
                            if normalize and hasattr(self, 'scaler') and self.scaler is not None:
                                try:
                                    # Convert to valid lottery numbers
                                    scaled_prediction = self.scaler.inverse_transform(raw_prediction.reshape(1, -1))[0]
                                except Exception as e:
                                    print(f"Error in inverse scaling: {e}")
                                    scaled_prediction = raw_prediction
                            else:
                                scaled_prediction = raw_prediction
                                
                            # Append prediction with confidence
                            predictions.append({
                                'prediction': scaled_prediction.tolist(),
                                'confidence': float(confidence),
                                'model': model_name
                            })
                            
                            # If we have enough predictions, break
                            if len(predictions) >= num_predictions:
                                break
                        except Exception as model_e:
                            print(f"Error with {model_name} prediction: {model_e}")
                            continue
                    
                    # If we got at least one prediction, return them
                    if predictions:
                        return predictions
                except Exception as e:
                    print(f"Error making individual model predictions: {e}")
            
            # If all else fails, use Monte Carlo method
            print("Falling back to Monte Carlo predictions")
            return self._generate_monte_carlo_predictions(num_predictions)
            
        except Exception as e:
            print(f"Error in predict_next_numbers: {e}")
            # Final fallback - generate random predictions
            return self._generate_monte_carlo_predictions(num_predictions)
            
    def _generate_monte_carlo_predictions(self, num_predictions=10):
        """Generate lottery predictions using Monte Carlo method as a fallback."""
        predictions = []
        
        # Get lottery range
        min_number = 1
        max_number = 49  # Default for many lotteries
        
        # Try to get actual range from configuration
        if hasattr(self, 'config') and self.config:
            min_number = self.config.get('min_lottery_number', 1)
            max_number = self.config.get('max_lottery_number', 49)
            
        # Generate predictions
        for _ in range(num_predictions):
            # Generate random numbers and sort them
            numbers = sorted(np.random.randint(min_number, max_number+1, size=6))
            
            # Ensure unique numbers
            while len(set(numbers)) < 6:
                # Replace duplicates
                numbers = sorted(np.random.randint(min_number, max_number+1, size=6))
                
            predictions.append({
                'prediction': numbers,
                'confidence': float(0.5),  # Low confidence for Monte Carlo
                'model': 'Monte Carlo'
            })
            
        return predictions