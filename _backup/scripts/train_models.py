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

from scripts.utils import setup_logging, log_memory_usage, LOG_DIR
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
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
        self.ensemble = None
        self.is_trained = False
        
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
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train LSTM model with advanced features, hyperparameter tuning, and memory monitoring."""
        from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
        from scripts.utils.memory_monitor import memory_callback, get_memory_monitor, optimize_batch_size, log_memory_usage
        
        # Start memory monitoring
        monitor = get_memory_monitor(log_dir="logs/memory", interval=2.0)
        monitor.start()
        
        # Log initial memory state
        log_memory_usage("LSTM training - initial state")
        
        try:
            # Optimize hyperparameters
            use_hyperparameter_tuning = True
            if use_hyperparameter_tuning:
                try:
                    print("Optimizing hyperparameters for LSTM model...")
                    best_params = self.optimize_hyperparameters(X, y, 'lstm')
                    
                    # Extract optimized parameters
                    lstm_units_1 = best_params.get('lstm_units_1', 128)
                    lstm_units_2 = best_params.get('lstm_units_2', 64)
                    dropout_rate = best_params.get('dropout_rate', 0.3)
                    learning_rate = best_params.get('learning_rate', 0.001)
                    batch_size = best_params.get('batch_size', 48)
                    l2_reg = best_params.get('l2_reg', 0.001)
                    
                    print(f"Using optimized hyperparameters: units={lstm_units_1}/{lstm_units_2}, "
                          f"dropout={dropout_rate}, lr={learning_rate}, batch_size={batch_size}")
                except Exception as e:
                    print(f"Error during hyperparameter optimization: {str(e)}. Using default parameters.")
                    lstm_units_1 = 128
                    lstm_units_2 = 64
                    dropout_rate = 0.3
                    learning_rate = 0.001
                    batch_size = 48
                    l2_reg = 0.001
            else:
                # Use default parameters
                lstm_units_1 = 128
                lstm_units_2 = 64
                dropout_rate = 0.3
                learning_rate = 0.001
                batch_size = 48
                l2_reg = 0.001
            
            # Define model creation function for batch size optimization
            def create_lstm_model(batch_size):
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
                
                # Compile with Adam optimizer
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    clipnorm=1.0,
                    clipvalue=0.5
                )
                
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                return model
            
            # Find optimal batch size based on memory constraints
            print("Finding optimal batch size based on memory constraints...")
            optimal_batch_size = optimize_batch_size(
                create_lstm_model, 
                X, 
                y, 
                min_batch=16, 
                max_batch=256
            )
            print(f"Using optimized batch size: {optimal_batch_size}")
            
            # Create the final model
            print("Creating LSTM model with optimized parameters...")
            model = create_lstm_model(optimal_batch_size)
            
            # Use cosine decay learning rate schedule
            lr_schedule = CosineDecayRestarts(
                initial_learning_rate=learning_rate,
                first_decay_steps=1000,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.0001
            )
            
            # Compile with advanced optimizer
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=1.0,  # Gradient clipping to prevent exploding gradients
                clipvalue=0.5
            )
            
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Enhanced callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=40,  # Increased patience
                restore_best_weights=True,
                min_delta=0.0001  # Smaller threshold for improvement
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,  # More gradual reduction
                patience=8,  # Reduced patience for faster adaptation
                min_lr=1e-6,
                verbose=1
            )
            
            # Add tensorboard logging
            log_dir = "logs/lstm_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                profile_batch=0
            )
            
            # Add memory monitoring callback
            mem_callback = memory_callback("LSTM", log_interval=10)
            
            # Use TimeSeriesSplit for validation
            from sklearn.model_selection import TimeSeriesSplit
            time_series_cv = TimeSeriesSplit(n_splits=5)
            last_split = list(time_series_cv.split(X))[-1]
            train_idx, val_idx = last_split
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Train the model with memory monitoring
            print(f"Training LSTM model with {len(X_train_cv)} samples...")
            log_memory_usage("LSTM training - before fit")
            
            history = model.fit(
                X_train_cv, y_train_cv, 
                epochs=300,  # Increased epochs
                batch_size=optimal_batch_size,
                validation_data=(X_val_cv, y_val_cv),
                verbose=1,
                callbacks=[early_stopping, reduce_lr, tensorboard_callback, mem_callback]
            )
            
            # Log final memory state
            log_memory_usage("LSTM training - after fit")
            
            # Memory cleanup
            gc.collect()
            tf.keras.backend.clear_session()
            
            return model
        
        finally:
            # Stop memory monitoring and generate report
            monitor.stop()
            print("Memory monitoring completed and report generated.")
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train XGBoost model with optimized hyperparameters and memory monitoring."""
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
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            print(f"Training on {X_train_cv.shape[0]} samples, validating on {X_val_cv.shape[0]} samples")
            
            # Create DMatrix objects (more memory efficient)
            print("Converting to DMatrix format...")
            dtrain = xgb.DMatrix(X_train_cv, y_train_cv)
            dval = xgb.DMatrix(X_val_cv, y_val_cv)
            
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
            
            # Train model with early stopping
            print(f"Training XGBoost model with {n_estimators} estimators...")
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=25
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
            
            # Add categorical features if any
            # (In a real-world scenario, you'd identify categorical features)
            
            # Train model with early stopping
            print(f"Training LightGBM model with {n_estimators} estimators...")
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[valid_data, train_data],
                valid_names=['val', 'train'],
                early_stopping_rounds=50,
                verbose_eval=25
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
        """Train all models in the ensemble including CNN-LSTM."""
        print("Training ensemble model with advanced configuration...")
        
        # Train LSTM model
        print("Training LSTM model...")
        lstm_model = self.train_lstm_model(self.X_train, self.y_train)
        self.models['lstm'] = lstm_model
        
        # Reshape data for XGBoost and LightGBM (from 3D to 2D)
        X_2d = self.X_train.reshape(self.X_train.shape[0], -1)
        
        # Train XGBoost model
        print("Training XGBoost model...")
        xgb_model = self.train_xgboost_model(X_2d, self.y_train)
        self.models['xgboost'] = xgb_model
        
        # Train LightGBM model
        print("Training LightGBM model...")
        lgb_model = self.train_lightgbm_model(X_2d, self.y_train)
        self.models['lightgbm'] = lgb_model
        
        # Train CNN-LSTM model if input shape is appropriate
        if len(self.X_train.shape) == 3 and self.X_train.shape[1] >= 5 and self.X_train.shape[2] >= 6:
            try:
                print("Training CNN-LSTM model...")
                from models.cnn_lstm_model import train_cnn_lstm_model
                
                # Reshape data for CNN-LSTM (add channel dimension)
                X_cnn_lstm = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
                
                # Define CNN-LSTM config
                cnn_lstm_config = {
                    'sequence_length': self.X_train.shape[1],
                    'num_features': self.X_train.shape[2],
                    'filters': 32,
                    'kernel_size': 3,
                    'lstm_units': 64,
                    'dropout_rate': 0.3,
                    'l2_reg': 0.001,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 300
                }
                
                # Train the CNN-LSTM model
                cnn_lstm_model, _ = train_cnn_lstm_model(X_cnn_lstm, self.y_train, cnn_lstm_config)
                self.models['cnn_lstm'] = cnn_lstm_model
                print("Successfully trained CNN-LSTM model")
            except Exception as e:
                logging.error(f"Error training CNN-LSTM model: {str(e)}")
                logging.error("Continuing with other models...")
        
        # Get model weights with CNN-LSTM included if available
        weights = self.get_model_weights()
        
        # Initialize ensemble
        self.ensemble = LotteryEnsemble(models=self.models, weights=weights)
        self.is_trained = True
        print("Ensemble model training completed with all models")
        
        # Save trained models
        self.save_trained_models()
    
    def save_trained_models(self, models_path: str = 'models/checkpoints/trained_models.pkl') -> bool:
        """Save trained models to disk.
        
        Args:
            models_path: Path to save models file
            
        Returns:
            True if models were successfully saved, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(models_path), exist_ok=True)
            
            with open(models_path, 'wb') as f:
                pickle.dump(self.models, f)
                
            logging.info(f"Successfully saved {len(self.models)} models to {models_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
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
        
        # Perform backtesting
        backtest_metrics = backtest(self.ensemble, X_val, y_val)
        print("Backtest metrics:")
        for metric_name, value in backtest_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
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
                importances = model.layers[-1].get_weights()[0][:, 0]
                indices = np.argsort(importances)[::-1]
                print(f"\n{name} feature importance:")
                
                model_importances = {}
                for f in range(10):  # Print top 10 features
                    feature_name = feature_names[indices[f]]
                    importance = float(importances[indices[f]])
                    print(f"  {feature_name}: {importance:.4f}")
                    model_importances[feature_name] = importance
                
                feature_importances[name] = model_importances
        
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
    
    def visualize_outputs/results(self, X_val: np.ndarray, y_val: np.ndarray, output_dir: Path = None) -> None:
        """Visualize model outputs/results."""
        print("Visualizing outputs/results...")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = Path('outputs/results')
            os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions
        y_pred = self.ensemble.predict(X_val)
        
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
        Includes CNN-LSTM if available.
        
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
            'lstm': 0.3,
            'xgboost': 0.2, 
            'lightgbm': 0.2
        }
        
        # Add CNN-LSTM if it exists
        if 'cnn_lstm' in weights:
            weights['cnn_lstm'] = 0.3
        
        return weights

def train_model(model_name: str, X: Union[np.ndarray, pd.DataFrame], params: Dict[str, Any], tracker: Optional[Dict[str, Any]] = None) -> Any:
    """
    Train a model using the specified parameters.
    
    Args:
        model_name: Name of the model to train
        X: Input features as numpy array or DataFrame
        params: Parameters for the model
        tracker: Progress tracker object
    
    Returns:
        Trained model or model tuple
    """
    try:
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Get feature names if available
        feature_names = None
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            
        # Train the model
        update_model_status(model_name, "training", tracker)
        
        if model_name == 'lstm':
            model = train_lstm_model(X, params)
        elif model_name == 'arima':
            model = train_arima_model(X, params)
        elif model_name == 'holtwinters':
            model = train_holtwinters_model(X, params)
        elif model_name == 'linear':
            model = train_linear_models(X, params)
        elif model_name == 'xgboost':
            model = train_xgboost_model(X, params)
        elif model_name == 'lightgbm':
            model = train_lightgbm_model(X, params)
        elif model_name == 'knn':
            model = train_knn_model(X, params)
        elif model_name == 'gradient_boosting':
            model = train_gradient_boosting_model(X, params)
        elif model_name == 'catboost':
            model = train_catboost_model(X, params)
        elif model_name == 'cnn_lstm':
            model = train_cnn_lstm_model(X, params)
        elif model_name == 'autoencoder':
            model = train_autoencoder_model(X, params)
        elif model_name == 'meta':
            model = train_meta_model(X, params)
        else:
            update_model_status(model_name, "model not implemented", tracker)
            return None
            
        # Track model performance
        if model is not None:
            predictions = predict_with_model(model_name, model, X)
            metrics = calculate_metrics(predictions, X[:, -1, :] if len(X.shape) == 3 else X)
            track_model_performance(model_name, predictions, metrics, metadata=params)
            
            # Log performance trends
            history = get_model_performance_summary(model_name)
            log_performance_trends(model_name, history)
            
            update_model_status(model_name, "trained", tracker)
            return model
    
    except Exception as e:
        update_model_status(model_name, f"error: {str(e)}", tracker)
        logger.error(f"Error training {model_name} model: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def validate_model_predictions(model_name: str, model: Any, df: pd.DataFrame) -> bool:
    """
    Validate model predictions.
    
    Args:
        model_name: Name of the model
        model: Model instance
        df: DataFrame with lottery data
        
    Returns:
        Whether the model predictions are valid
    """
    try:
        # Initialize utility classes
        validator = ModelValidator()
        monitor = ModelMonitor()
        metrics = ModelMetrics()
        
        # Prepare data
        X, y = prepare_data(df.values)
        
        # Validate model
        validation_outputs/results = validator.validate_model(model, X, y)
        
        # Monitor validation
        monitor.monitor_validation(model_name, validation_outputs/results)
        
        # Check if validation passed
        return validation_outputs/results.get('is_valid', False)
        
    except Exception as e:
        logger.error(f"Error validating model {model_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
    """Train the ensemble model with normalized data."""
    print("Training ensemble model...")
    
    # Initialize models dictionary
    self.models = {}
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm_model = self.train_lstm_model(X, y)
    self.models['lstm'] = lstm_model
    
    # Reshape data for XGBoost and LightGBM (from 3D to 2D)
    X_2d = X.reshape(X.shape[0], -1)  # Flatten the sequences
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = self.train_xgboost_model(X_2d, y)
    self.models['xgboost'] = xgb_model
    
    # Train LightGBM model
    print("Training LightGBM model...")
    lgb_model = self.train_lightgbm_model(X_2d, y)
    self.models['lightgbm'] = lgb_model
    
    # Get model weights
    weights = self.get_model_weights()
    
    # Initialize ensemble
    self.ensemble = LotteryEnsemble(models=self.models, weights=weights)
    print("Ensemble model training completed")

def train_all_models(df: pd.DataFrame, force_retrain: Union[bool, str] = False) -> Dict[str, Any]:
    """
    Train all models and save them.
    
    Args:
        df: DataFrame with lottery data
        force_retrain: Whether to force retraining of all models or specific models
        
    Returns:
        Dictionary of trained models
    """
    try:
        # Analyze data for training
        analysis_outputs/results = analyze_lottery_data(df)
        patterns = analyze_patterns(df)
        correlations = analyze_correlations(df)
        randomness = analyze_randomness(df.values)
        spatial_dist = analyze_spatial_distribution(df.values)
        range_freq = analyze_range_frequency(df)
        
        # Get hot and cold numbers
        hot_numbers, cold_numbers = get_hot_cold_numbers(df)
        
        # Load existing models if not forcing retrain
        if not force_retrain:
            try:
                models = load_trained_models()
                if models:
                    logger.info("Loaded existing models")
                    return models
            except Exception as e:
                logger.warning(f"Could not load existing models: {str(e)}")
                
        # Initialize models dictionary
        models = {}
        
        # Train each model
        for model_name, params in TRAINING_CONFIG.items():
            # Skip if not forcing retrain and model exists
            if not force_retrain and model_name in models:
                continue
                
            # Train model
            model = train_model(model_name, df, params)
            if model is not None:
                models[model_name] = model
                
        # Train ensemble
        ensemble = train_ensemble(df.values)
        if ensemble is not None:
            models['ensemble'] = ensemble
            
        # Save models
        with open(MODELS_PATH, 'wb') as f:
            pickle.dump(models, f)
            
        return models
        
    except Exception as e:
        logger.error(f"Error training all models: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}

def load_trained_models() -> Dict[str, Any]:
    """
    Load trained models from disk.
    
    Returns:
        Dictionary of trained models
    """
    try:
        # Load models
        if not os.path.exists(MODELS_PATH):
            logger.warning(f"Models file not found at {MODELS_PATH}")
            return {}
            
        with open(MODELS_PATH, 'rb') as f:
            models = pickle.load(f)
            
        # Validate loaded models
        for model_name, model in models.items():
            if not validate_model_predictions(model_name, model, pd.DataFrame()):
                logger.warning(f"Model {model_name} failed validation")
                del models[model_name]
                
        # Track loaded models
        for model_name, model in models.items():
            track_model_performance(model_name, None, None, metadata={'status': 'loaded'})
            
        # Log performance trends
        for model_name in models:
            history = get_model_performance_summary(model_name)
            log_performance_trends(model_name, history)
                
        return models
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}

def update_model_status(model_name: str, status: str, tracker: Optional[Dict[str, Any]] = None, save: bool = True) -> None:
    """
    Update the status of a model in the tracker.
    
    Args:
        model_name: Name of the model
        status: Status message to update
        tracker: Progress tracker object
        save: Whether to save the tracker to disk
    """
    # Update model status in the tracker if provided
    if tracker is not None:
        update_model_progress(tracker, model_name, status)
    
    # Log status in any case
    logger.info(f"Model {model_name}: {status}")
    
    # Save tracker to disk if requested
    if save and tracker is not None:
        try:
            with open("model_status.json", "w") as f:
                json.dump({k: str(v) for k, v in tracker.items() if k != "main"}, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save model status: {str(e)}")

def prepare_data(data: np.ndarray, timesteps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training by creating sequences and targets.
    
    Args:
        data: Input data array
        timesteps: Number of timesteps for sequence data
        
    Returns:
        Tuple of (X, y) where X is the input sequences and y is the targets
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    elif len(data.shape) > 2:
        raise ValueError(f"Input data must be 1D or 2D, got shape {data.shape}")
        
    # Create sequences
    X = []
    y = []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps)])
        y.append(data[i + timesteps])
    
    return np.array(X), np.array(y)

def predict_next_numbers(ensemble: LotteryEnsemble, 
                        last_numbers: np.ndarray,
                        n_predictions: int = 1,
                        n_monte_carlo: int = 500) -> np.ndarray:
    """
    Predict next lottery numbers using advanced Monte Carlo simulation.
    
    Args:
        ensemble: Trained ensemble model
        last_numbers: Input features (last numbers)
        n_predictions: Number of predictions to generate
        n_monte_carlo: Number of Monte Carlo simulations per prediction
        
    Returns:
        Array of predictions
    """
    if ensemble is None:
        logger.error("Cannot predict with None ensemble")
        return np.array([])
        
    predictions = []
    for _ in range(n_predictions):
        # Monte Carlo approach: add random noise multiple times and average outputs/results
        mc_predictions = []
        
        for _ in range(n_monte_carlo):
            # Add small random noise to input for robustness
            # Use different noise levels for different simulations
            noise_scale = np.random.uniform(0.005, 0.02)  # Random noise between 0.5% and 2%
            
            if len(last_numbers.shape) == 3:  # 3D input for LSTM
                noise = np.random.normal(0, noise_scale, last_numbers.shape)
            else:  # 2D input
                noise = np.random.normal(0, noise_scale, last_numbers.shape)
                
            noisy_input = last_numbers + noise
            
            # Reshape for prediction if needed
            if len(last_numbers.shape) == 3:
                input_reshaped = noisy_input
            else:
                input_reshaped = noisy_input.reshape(noisy_input.shape[0], -1)
                
            # Get prediction
            pred = ensemble.predict(input_reshaped)
            mc_predictions.append(pred)
        
        # Average all Monte Carlo predictions with weighted averaging
        # Recent simulations get slightly higher weight (recency bias)
        weights = np.linspace(0.8, 1.2, n_monte_carlo)
        weights = weights / np.sum(weights)  # Normalize
        
        # Apply weights to predictions
        weighted_predictions = np.zeros_like(mc_predictions[0])
        for i, pred in enumerate(mc_predictions):
            weighted_predictions += pred * weights[i]
        
        # Convert to lottery numbers
        # Denormalize and ensure uniqueness and valid range
        lottery_numbers = ensemble.predict_next_draw(last_numbers)
        
        predictions.append(lottery_numbers)
        
    return np.array(predictions)

if __name__ == "__main__":
    try:
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train lottery prediction models')
        parser.add_argument('--data', type=str, default=str(DATA_PATH),
                           help=f'Path to lottery data (default: {DATA_PATH})')
        parser.add_argument('--force', type=str, choices=['yes', 'no'], default='no',
                           help='Force retraining of models (yes/no)')
        parser.add_argument('--config', type=str, choices=['default', 'quick', 'deep'], default='default',
                           help='Configuration to use for training (default/quick/deep)')
        
        args = parser.parse_args()
        
        # Apply configuration
        if args.config == 'quick':
            from models.training_config import QUICK_TRAINING_CONFIG as CONFIG
            print("Using QUICK training configuration")
        elif args.config == 'deep':
            from models.training_config import DEEP_TRAINING_CONFIG as CONFIG
            print("Using DEEP training configuration")
        else:
            from models.training_config import LSTM_CONFIG as CONFIG
            print("Using DEFAULT training configuration")
        
        # Load data
        print(f"Loading lottery data from {args.data}...")
        df = load_data(args.data)
        print(f"Loaded {len(df)} lottery draws")
        
        # Train models
        print(f"Training models (force={args.force}, config={args.config})...")
        models = train_all_models(df, force_retrain=args.force)
        
        if models:
            print(f"Successfully trained {len(models)} models:")
            for model_name in models.keys():
                print(f"  - {model_name}")
        else:
            print("No models were successfully trained")
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc() 