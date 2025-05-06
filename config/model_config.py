"""
Configuration file for lottery prediction models.
Controls which models are active and their weights in ensemble prediction.
"""

# Model Selection Configuration
ACTIVE_MODELS = {
    'lstm': True,          # Deep learning for sequence patterns
    'cnn_lstm': True,      # CNN-LSTM hybrid for complex patterns
    'xgboost': True,       # Gradient boosting for feature-based prediction
    'lightgbm': True,      # Light GBM for fast feature-based prediction
    'catboost': True,      # CatBoost for robust feature-based prediction
    'knn': False,          # K-nearest neighbors (backup)
    'autoencoder': True,   # Deep learning for pattern detection
    'linear': False,       # Simple linear models (backup)
    'holtwinters': False,  # Time series forecasting (backup)
    'arima': False,        # Statistical time series (backup)
    'meta': True,          # Meta-model combining other models
}

# Model Weights for Ensemble Prediction
# Higher weight = more influence on final prediction
MODEL_WEIGHTS = {
    'lstm': 0.20,         # Good at sequence patterns
    'cnn_lstm': 0.20,     # Good at complex patterns
    'xgboost': 0.15,      # Strong general predictor
    'lightgbm': 0.15,     # Fast and accurate
    'catboost': 0.15,     # Robust to outliers
    'autoencoder': 0.10,  # Good at finding hidden patterns
    'meta': 0.05,         # Meta-learning from other models
}

# Model-specific Parameters
MODEL_PARAMS = {
    'lstm': {
        'sequence_length': 200,
        'units': [128, 64, 32],
        'dropout': 0.2,
    },
    'cnn_lstm': {
        'filters': 64,
        'kernel_size': 3,
        'lstm_units': 128,
    },
    'xgboost': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 6,
    },
    'lightgbm': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'num_leaves': 31,
    },
    'catboost': {
        'iterations': 500,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_strength': 1,
        'bagging_temperature': 1,
        'od_type': 'Iter',
        'od_wait': 20,
        'verbose': False,
        'task_type': 'GPU' if TRAINING_CONFIG['use_gpu'] else 'CPU',
        'devices': '0',
    },
    'autoencoder': {
        'encoding_dim': 32,
        'layers': [64, 32, 16],
    },
    'meta': {
        'method': 'weighted_average',
        'cv_folds': 5,
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'use_historical_stats': True,
    'sequence_length': 300,
    'use_date_features': True,
    'use_frequency_features': True,
    'use_pattern_features': True,
}

# Training Configuration
TRAINING_CONFIG = {
    'validation_split': 0.2,
    'early_stopping_patience': 15,
    'batch_size': 32,
    'epochs': 150,
    'use_gpu': True,
    'parallel_training': True,
    'max_workers': None,  # None = auto-detect CPU cores
    'mixed_precision': True,  # Enable mixed precision training
    'gradient_accumulation_steps': 2,  # Accumulate gradients for larger effective batch size
    'learning_rate_schedule': {
        'initial_learning_rate': 0.001,
        'decay_steps': 1000,
        'decay_rate': 0.9,
        'staircase': True
    },
    'model_checkpointing': {
        'save_best_only': True,
        'save_weights_only': False,
        'save_frequency': 5
    },
    'gpu_memory_growth': True,
    'gpu_memory_fraction': 0.8
} 