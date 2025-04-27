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
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 6,
    },
    'lightgbm': {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'num_leaves': 31,
    },
    'catboost': {
        'iterations': 1000,
        'learning_rate': 0.01,
        'depth': 6,
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
    'sequence_length': 200,
    'use_date_features': True,
    'use_frequency_features': True,
    'use_pattern_features': True,
}

# Training Configuration
TRAINING_CONFIG = {
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'batch_size': 32,
    'epochs': 100,
    'use_gpu': True,
    'parallel_training': True,
    'max_workers': None,  # None = auto-detect CPU cores
} 