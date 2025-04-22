"""
Configuration settings for neural network models training
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
    "batch_size": 64,                 # Batch size for training
    "epochs": 300,                    # Maximum number of training epochs
    "learning_rate": 0.001,           # Initial learning rate
    "lr_patience": 10,                # Patience for learning rate reduction
    "lr_factor": 0.5,                 # Factor for learning rate reduction
    "early_stopping_patience": 30,    # Patience for early stopping
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

# Load environment-specific overrides if available
try:
    from local_config import *
except ImportError:
    pass 