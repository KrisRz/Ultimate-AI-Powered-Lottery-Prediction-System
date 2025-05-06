"""Model optimization utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import optuna
from sklearn.model_selection import TimeSeriesSplit
from .cross_validation import monte_carlo_cv, rolling_cv

logger = logging.getLogger(__name__)

def optimize_lstm_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
    """Optimize LSTM model hyperparameters using Optuna."""
    def objective(trial):
        # Define hyperparameter space
        params = {
            'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 512),
            'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 256),
            'dense_units': trial.suggest_int('dense_units', 16, 128),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        }
        
        # Get CV splits
        splits = rolling_cv(X, y, n_splits=3)
        val_scores = []
        
        for X_train, X_val, y_train, y_val in splits:
            # Create and train model
            model = create_lstm_model(X_train.shape[1:], params)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=params['batch_size'],
                verbose=0
            )
            
            # Evaluate
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            val_scores.append(val_loss)
        
        return np.mean(val_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def optimize_xgboost_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
    """Optimize XGBoost model hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1.0)
        }
        
        splits = monte_carlo_cv(X, y, n_splits=3)
        val_scores = []
        
        for X_train, X_val, y_train, y_val in splits:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            val_score = model.score(X_val, y_val)
            val_scores.append(val_score)
        
        return -np.mean(val_scores)  # Negative because we want to maximize
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def optimize_ensemble_weights(models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Optimize ensemble model weights using validation performance."""
    def objective(trial):
        weights = {}
        for model_name in models.keys():
            weights[model_name] = trial.suggest_float(f'weight_{model_name}', 0, 1)
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        splits = rolling_cv(X, y, n_splits=3)
        val_scores = []
        
        for X_train, X_val, y_train, y_val in splits:
            predictions = []
            for model_name, model in models.items():
                pred = model.predict(X_val)
                predictions.append(pred * weights[model_name])
            
            ensemble_pred = np.sum(predictions, axis=0)
            val_score = mean_squared_error(y_val, ensemble_pred)
            val_scores.append(val_score)
        
        return np.mean(val_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Get and normalize final weights
    weights = {k: study.best_params[f'weight_{k}'] for k in models.keys()}
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}

def optimize_model_architecture(model_type: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Optimize model architecture based on type."""
    if model_type == 'lstm':
        return optimize_lstm_hyperparameters(X, y)
    elif model_type == 'xgboost':
        return optimize_xgboost_hyperparameters(X, y)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def optimize_training_parameters(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Optimize training parameters for a given model."""
    def objective(trial):
        params = {
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'epochs': trial.suggest_int('epochs', 50, 300)
        }
        
        splits = rolling_cv(X, y, n_splits=3)
        val_scores = []
        
        for X_train, X_val, y_train, y_val in splits:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='mse'
            )
            
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0
            )
            
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            val_scores.append(val_loss)
        
        return np.mean(val_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    return study.best_params

class ModelOptimizer:
    """Class for optimizing model hyperparameters and architectures."""
    
    def __init__(self, model_type: str = None, model: Any = None):
        """Initialize ModelOptimizer.
        
        Args:
            model_type: Type of model to optimize ('lstm' or 'xgboost')
            model: Pre-trained model instance for optimizing training parameters
        """
        self.model_type = model_type
        self.model = model
        self.best_params = None
        
    def optimize_architecture(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize model architecture hyperparameters."""
        if self.model_type is None:
            raise ValueError("model_type must be specified for architecture optimization")
        
        self.best_params = optimize_model_architecture(self.model_type, X, y)
        return self.best_params
        
    def optimize_training(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize training parameters."""
        if self.model is None:
            raise ValueError("model instance must be provided for training optimization")
            
        self.best_params = optimize_training_parameters(self.model, X, y)
        return self.best_params
        
    def optimize_ensemble_weights(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Optimize weights for ensemble models."""
        return optimize_ensemble_weights(models, X, y)
        
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found during optimization."""
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet")
        return self.best_params 