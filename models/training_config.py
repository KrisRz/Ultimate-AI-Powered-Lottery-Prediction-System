"""
Configuration settings for neural network models training.

This file contains all the configuration parameters for models:
1. Default configuration values for all supported model types
2. Path constants for data, checkpoints, and output directories
3. Hyperparameter spaces for optimization
4. Training configurations for different scenarios (quick, deep)
5. Hardware utilization settings

This file DEFINES CONFIGURATIONS, while scripts/train_models.py 
IMPLEMENTS the training logic using these configurations.
"""

import os
import multiprocessing
from pathlib import Path

# Basic paths
DATA_PATH = "data/lottery_data_1995_2025.csv"
MODELS_DIR = Path("models/checkpoints")
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("results")

# Ensure directories exist
for directory in [MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# LSTM model configuration
LSTM_CONFIG = {
    # Data parameters
    "look_back": 200,                 # Number of previous draws to use as context
    "validation_split": 0.15,         # Portion of data for validation
    "test_split": 0.1,                # Portion of data for testing
    "use_enhanced_features": True,    # Whether to use advanced feature engineering
    "use_pca": False,                 # Whether to use PCA for dimensionality reduction
    "pca_components": 30,             # Number of PCA components if enabled
    "scaling_method": "robust",       # Scaling method: 'standard', 'robust', or 'power'
    
    # Model architecture
    "lstm_units_1": 256,              # Units in first LSTM layer
    "lstm_units_2": 128,              # Units in second LSTM layer
    "dense_units": 64,                # Units in dense layer
    "dropout_rate": 0.3,              # Dropout rate for regularization
    "l2_reg": 0.001,                  # L2 regularization factor
    "bidirectional": True,            # Whether to use bidirectional LSTM
    
    # Training parameters
    "batch_size": 48,                 # Optimized batch size (reduced from 64)
    "epochs": 300,                    # Increased from original value to allow more training
    "learning_rate": 0.001,           # Initial learning rate
    "lr_patience": 8,                 # Reduced patience for faster adaptation
    "lr_factor": 0.6,                 # Modified for smoother reduction (was 0.5)
    "early_stopping_patience": 40,    # Increased to allow more exploration
    "shuffle": True,                  # Shuffle data during training
    
    # Performance and output
    "save_best_only": True,           # Only save best model based on validation loss
    "monitor_metric": "val_loss",     # Metric to monitor for callbacks
    "verbose": 1,                     # Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    "save_history": True,             # Whether to save training history
    
    # Hardware utilization
    "use_gpu": True,                  # Whether to use GPU if available
    "mixed_precision": True,          # Whether to use mixed precision training
    "num_workers": max(1, multiprocessing.cpu_count() - 1),  # Number of workers for data loading
    "memory_growth": True,            # Allow memory growth on GPU
}

# CNN-LSTM model configuration
CNN_LSTM_CONFIG = {
    # Base settings from LSTM
    **LSTM_CONFIG,
    
    # CNN-specific parameters
    "filters_1": 128,                 # Filters in first Conv1D layer
    "filters_2": 256,                 # Filters in second Conv1D layer
    "kernel_size": 3,                 # Kernel size for Conv1D layers
    "pool_size": 2,                   # Pool size for MaxPooling1D
    "use_batch_norm": True,           # Whether to use batch normalization
}

# Hyperparameter search spaces for optimization
LSTM_HYPERPARAMETER_SPACE = {
    "look_back": [100, 150, 200, 250],
    "lstm_units_1": [128, 256, 384],
    "lstm_units_2": [64, 128, 192],
    "dropout_rate": [0.2, 0.3, 0.4],
    "l2_reg": [0.0001, 0.001, 0.01],
    "learning_rate": [0.0005, 0.001, 0.002],
    "batch_size": [32, 64, 128]
}

# Training configurations for different scenarios
QUICK_TRAINING_CONFIG = {
    **LSTM_CONFIG,
    "epochs": 50,
    "early_stopping_patience": 10,
    "batch_size": 128,
}

DEEP_TRAINING_CONFIG = {
    **LSTM_CONFIG,
    "epochs": 500,
    "early_stopping_patience": 50,
    "lstm_units_1": 384,
    "lstm_units_2": 192,
    "dense_units": 96,
    "batch_size": 32,
}

# Autoencoder configuration
AUTOENCODER_CONFIG = {
    "input_dim": 200,
    "encoding_dim": 32,
    "hidden_layers": [128, 64],
    "dropout_rate": 0.3,
    "l2_reg": 0.001,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 200,
    "early_stopping_patience": 20,
    "validation_split": 0.15,
    "use_batch_norm": True,
    "activation": "relu",
    "final_activation": "sigmoid"
}

# Holt-Winters configuration
HOLTWINTERS_CONFIG = {
    "seasonal_periods": 52,  # Weekly seasonality
    "trend": "add",         # Additive trend
    "seasonal": "add",      # Additive seasonality
    "damped_trend": True,   # Use damped trend
    "use_boxcox": True,     # Apply Box-Cox transformation
    "remove_bias": True,    # Remove bias from the forecast
    "optimized": True,      # Optimize model parameters
    "use_brute": True,      # Use brute force optimization
    "validation_split": 0.15,
    "test_split": 0.1,
    "n_jobs": -1           # Use all available CPU cores
}

# Linear models configuration
LINEAR_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    "use_enhanced_features": True,
    "scaling_method": "robust",
    
    # Model types
    "models": ["linear", "ridge", "lasso", "elastic_net"],
    
    # Hyperparameters for Ridge
    "ridge_params": {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": [True],
        "solver": ["auto"]
    },
    
    # Hyperparameters for Lasso
    "lasso_params": {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": [True],
        "max_iter": [1000]
    },
    
    # Hyperparameters for ElasticNet
    "elastic_net_params": {
        "alpha": [0.1, 1.0, 10.0],
        "l1_ratio": [0.2, 0.5, 0.8],
        "fit_intercept": [True],
        "max_iter": [1000]
    },
    
    # Cross-validation parameters
    "cv_folds": 5,
    "n_jobs": -1,
    
    # Optuna optimization
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "study_direction": "maximize"
}

# XGBoost configuration
XGBOOST_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    "use_enhanced_features": True,
    
    # Model parameters
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 1000,
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    
    # Training parameters
    "early_stopping_rounds": 50,
    "verbose": 100,
    
    # Hyperparameter tuning
    "cv_folds": 5,
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "study_direction": "minimize"
}

# LightGBM configuration
LIGHTGBM_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    "use_enhanced_features": True,
    
    # Model parameters
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "n_estimators": 1000,
    
    # Training parameters
    "early_stopping_rounds": 50,
    
    # Hyperparameter tuning
    "cv_folds": 5,
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "study_direction": "minimize"
}

# KNN configuration
KNN_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    "use_enhanced_features": True,
    "scaling_method": "standard",
    
    # Model parameters
    "n_neighbors": 5,
    "weights": "uniform",
    "algorithm": "auto",
    "leaf_size": 30,
    "p": 2,  # Power parameter for Minkowski metric
    
    # Hyperparameter tuning
    "cv_folds": 5,
    "n_trials": 50,
    "timeout": 1800,  # 30 minutes
    "study_direction": "minimize",
    
    # Parameter search space
    "param_grid": {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"],
        "leaf_size": [20, 30, 40, 50],
        "p": [1, 2]
    }
}

# Gradient Boosting configuration
GRADIENT_BOOSTING_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    "use_enhanced_features": True,
    
    # Model parameters
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 1.0,
    "max_features": "sqrt",
    "random_state": 42,
    
    # Training parameters
    "verbose": 1,
    "warm_start": False,
    
    # Hyperparameter tuning
    "cv_folds": 5,
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "study_direction": "minimize",
    
    # Parameter search space
    "param_grid": {
        "n_estimators": [100, 500, 1000],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None]
    }
}

# CatBoost configuration
CATBOOST_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    "use_enhanced_features": True,
    
    # Model parameters
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "random_seed": 42,
    "allow_writing_files": False,
    
    # Loss function
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    
    # Training parameters
    "early_stopping_rounds": 50,
    "verbose": 100,
    
    # Hyperparameter tuning
    "cv_folds": 5,
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "study_direction": "minimize"
}

# Meta-model configuration
META_CONFIG = {
    # Data parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    
    # Base models to use
    "base_models": [
        "lstm",
        "xgboost",
        "lightgbm",
        "catboost",
        "gradient_boosting",
        "knn"
    ],
    
    # Stacking parameters
    "meta_model": "lightgbm",  # Model to use for final predictions
    "use_probabilities": False,  # Whether to use probability predictions
    "cv_folds": 5,  # Number of folds for stacking
    
    # Training parameters
    "refit_base_models": False,  # Whether to retrain base models
    "n_jobs": -1,  # Number of parallel jobs
    
    # Meta-model specific parameters
    "meta_model_params": {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    # Base models to include in ensemble
    "models": [
        "lstm",
        "cnn_lstm",
        "xgboost",
        "lightgbm",
        "gradient_boosting",
        "catboost"
    ],
    
    # Weighting strategy
    "weighting": "performance",  # Options: "equal", "performance", "dynamic"
    
    # Validation parameters
    "validation_split": 0.15,
    "test_split": 0.1,
    
    # Training parameters
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "verbose": 1,
    
    # Stacking parameters
    "use_stacking": True,
    "stacking_cv": 5,
    "stacking_estimator": "xgboost",
    
    # Performance monitoring
    "save_best_only": True,
    "monitor_metric": "val_loss",
    
    # Hardware utilization
    "use_gpu": True,
    "num_workers": max(1, multiprocessing.cpu_count() - 1)
}

# Default training configuration
TRAINING_CONFIG = {
    # Data parameters
    "look_back": 200,
    "validation_split": 0.05,  # Reduced to match main.py using 95% for training
    "test_split": 0.05,        # Reduced test split as well
    "use_enhanced_features": True,
    "scaling_method": "robust",
    
    # Training parameters
    "batch_size": 48,         # Optimized from 128 based on testing
    "epochs": 300,           # Increased from 500 to 300 - balance between time and accuracy
    "learning_rate": 0.001,
    "learning_rate_schedule": {
        "type": "cosine_decay_restarts",
        "initial_learning_rate": 0.001,
        "first_decay_steps": 1000,
        "t_mul": 2.0,
        "m_mul": 0.9,
        "alpha": 0.0001
    },
    "early_stopping_patience": 40,  # Increased from 30 to allow more training time
    "shuffle": True,
    
    # Model parameters
    "dropout_rate": 0.3,
    "l2_reg": 0.001,
    
    # Monte Carlo parameters
    "n_monte_carlo": 500,     # Increased from 100 for more robust predictions
    
    # Hardware utilization
    "use_gpu": True,
    "mixed_precision": True,
    "num_workers": max(1, multiprocessing.cpu_count() - 1),
    "memory_growth": True,
    
    # Logging and checkpoints
    "save_best_only": True,
    "monitor_metric": "val_loss",
    "verbose": 1,
    "save_history": True
}

# Load environment-specific overrides if available
try:
    from local_config import *
except ImportError:
    pass 