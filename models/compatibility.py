"""
Model compatibility utilities.

This file contains low-level compatibility functions for model operations:
1. Functions for ensuring valid predictions from models
2. Data format conversion for model inputs/outputs
3. Model import and loading utilities
4. Core prediction functions for individual models
5. Prediction format validation and standardization

This file focuses on MODEL-LEVEL compatibility, while scripts/compatibility.py 
focuses on higher-level MODULE/SYSTEM compatibility and provides convenience 
wrappers for application logic.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import random
import importlib
import sys
from pathlib import Path
import os
import json

logger = logging.getLogger(__name__)

def import_prediction_function(model_name: str) -> Optional[Callable]:
    """Import prediction function for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Prediction function if found, None otherwise
    """
    try:
        # Map model names to their modules
        model_modules = {
            'lstm': 'models.lstm_model',
            'arima': 'models.arima_model',
            'holtwinters': 'models.holtwinters_model',
            'linear': 'models.linear_model',
            'xgboost': 'models.xgboost_model',
            'lightgbm': 'models.lightgbm_model',
            'knn': 'models.knn_model',
            'gradient_boosting': 'models.gradient_boosting_model',
            'catboost': 'models.catboost_model',
            'cnn_lstm': 'models.cnn_lstm_model',
            'autoencoder': 'models.autoencoder_model',
            'meta': 'models.meta_model'
        }
        
        # Get module name
        module_name = model_modules.get(model_name.lower())
        if not module_name:
            logger.warning(f"Unknown model type: {model_name}")
            return None
            
        # Import module
        module = importlib.import_module(module_name)
        
        # Get prediction function
        predict_func_name = f"predict_{model_name.lower()}_model"
        if hasattr(module, predict_func_name):
            return getattr(module, predict_func_name)
        else:
            logger.warning(f"Prediction function {predict_func_name} not found in {module_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error importing prediction function for {model_name}: {str(e)}")
        return None

def is_model_compatible(model: Any) -> bool:
    """Check if a model is compatible with the prediction system.
    
    Args:
        model: Model to check
        
    Returns:
        Whether the model is compatible
    """
    try:
        # Check for required methods
        required_methods = ['fit', 'predict']
        for method in required_methods:
            if not hasattr(model, method):
                logger.warning(f"Model missing required method: {method}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error checking model compatibility: {str(e)}")
        return False

def ensure_model_compatibility(model: Any) -> bool:
    """Ensure a model is compatible with the prediction system.
    
    Args:
        model: Model to check
        
    Returns:
        Whether the model is compatible
    """
    try:
        if not is_model_compatible(model):
            logger.warning("Model is not compatible")
            return False
            
        # Additional compatibility checks can be added here
        
        return True
    except Exception as e:
        logger.error(f"Error ensuring model compatibility: {str(e)}")
        return False

def convert_model_format(model: Any, target_format: str) -> Any:
    """Convert a model to a different format.
    
    Args:
        model: Model to convert
        target_format: Target format
        
    Returns:
        Converted model
    """
    try:
        # Currently only supports the default format
        if target_format != 'default':
            logger.warning(f"Unsupported target format: {target_format}")
            return model
            
        return model
    except Exception as e:
        logger.error(f"Error converting model format: {str(e)}")
        return model

def ensure_valid_prediction(pred: Union[List, np.ndarray], as_array: bool = True) -> Union[np.ndarray, List[int]]:
    """
    Ensure predictions are 6 unique integers between 1 and 59.
    
    Args:
        pred: Raw model prediction
        as_array: Whether to return numpy array (True) or list (False)
        
    Returns:
        Array or list of 6 unique integers between 1 and 59
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(pred, list):
            pred = np.array(pred)
        
        # Handle NaN, infinity, or other invalid values
        if pred is None or not isinstance(pred, (list, np.ndarray)):
            logger.warning(f"Invalid prediction type: {type(pred)}. Generating random numbers.")
            result = sorted(random.sample(range(1, 60), 6))
            return np.array(result) if as_array else list(result)
        
        # Check for NaN or infinite values
        if np.isnan(pred).any() or np.isinf(pred).any():
            logger.warning(f"Prediction contains NaN or Inf values: {pred}. Fixing...")
            pred = np.nan_to_num(pred, nan=30.0, posinf=59.0, neginf=1.0)
        
        # Handle potential fractional values by rounding
        pred = np.round(pred).astype(int)
        
        # Ensure values are within valid range (1-59)
        pred = np.clip(pred, 1, 59)
        
        # Check if we have the right number of predictions
        if len(pred) != 6:
            logger.warning(f"Prediction has {len(pred)} numbers instead of 6. Adjusting...")
            if len(pred) > 6:
                # Take the first 6 values
                pred = pred[:6]
            else:
                # Add random numbers until we have 6
                current = set(pred)
                while len(current) < 6:
                    candidate = random.randint(1, 59)
                    if candidate not in current:
                        current.add(candidate)
                pred = np.array(list(current))
        
        # Ensure uniqueness
        unique_values = set(pred)
        if len(unique_values) < 6:
            logger.warning(f"Prediction has duplicate values: {pred}. Fixing...")
            while len(unique_values) < 6:
                candidate = random.randint(1, 59)
                if candidate not in unique_values:
                    unique_values.add(candidate)
            pred = np.array(list(unique_values))
        
        # Sort and return in requested format
        result = sorted([int(x) for x in pred])
        return np.array(result) if as_array else result
    
    except Exception as e:
        logger.error(f"Error processing prediction: {e}. Generating random numbers.")
        result = sorted(random.sample(range(1, 60), 6))
        return np.array(result) if as_array else list(result)

def predict_with_model(model_name: str, model: Any, X: np.ndarray) -> np.ndarray:
    """Make predictions with a model.
    
    Args:
        model_name: Name of the model
        model: Model instance
        X: Input features
        
    Returns:
        Model predictions
    """
    try:
        # Get prediction function
        predict_func = import_prediction_function(model_name)
        if predict_func is not None:
            # Use model-specific prediction function
            predictions = predict_func(model, X)
        else:
            # Fallback to standard predict method
            predictions = model.predict(X)
            
        # Ensure valid predictions
        return ensure_valid_prediction(predictions)
        
    except Exception as e:
        logger.error(f"Error making predictions with {model_name}: {str(e)}")
        # Return random predictions as fallback
        return ensure_valid_prediction([])

def score_combinations(predictions: List[List[int]], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Score combinations of predictions.
    
    Args:
        predictions: List of prediction lists
        weights: Optional dictionary of model weights
        
    Returns:
        Array of scores for each combination
    """
    try:
        if not predictions:
            return np.array([])
            
        # Convert predictions to numpy array
        pred_array = np.array(predictions)
        
        if weights is None:
            # Equal weights if not provided
            weights = {str(i): 1.0/len(predictions) for i in range(len(predictions))}
            
        # Calculate weighted scores
        scores = np.zeros(len(predictions))
        for i, pred in enumerate(predictions):
            # Score based on frequency of numbers
            freq = np.bincount(pred, minlength=60)[1:]  # Skip 0
            scores[i] = np.sum(freq * np.array([weights.get(str(j), 1.0) for j in range(1, 60)]))
            
        return scores
        
    except Exception as e:
        logger.error(f"Error scoring combinations: {str(e)}")
        return np.zeros(len(predictions))

def ensemble_prediction(models: List[Any], X: np.ndarray, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Make ensemble predictions.
    
    Args:
        models: List of models
        X: Input features
        weights: Optional dictionary of model weights
        
    Returns:
        Array of ensemble predictions
    """
    try:
        if not models:
            return np.zeros((X.shape[0], 6))
            
        # Get predictions from each model
        predictions = []
        for model in models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error getting predictions from model: {str(e)}")
                predictions.append(np.zeros((X.shape[0], 6)))
                
        # Convert predictions to numpy array
        predictions = np.array(predictions)
        
        if weights is None:
            # Equal weights if not provided
            weights = {str(i): 1.0/len(models) for i in range(len(models))}
            
        # Calculate weighted predictions
        weighted_predictions = np.zeros((X.shape[0], 6))
        for i, pred in enumerate(predictions):
            weight = weights.get(str(i), 1.0/len(models))
            weighted_predictions += weight * pred
            
        # Ensure valid predictions
        return ensure_valid_prediction(weighted_predictions)
        
    except Exception as e:
        logger.error(f"Error making ensemble predictions: {str(e)}")
        return np.zeros((X.shape[0], 6))

def monte_carlo_simulation(model: Any, X: np.ndarray, n_simulations: int = 1000) -> np.ndarray:
    """Run Monte Carlo simulation for predictions.
    
    Args:
        model: Model to use for predictions
        X: Input features
        n_simulations: Number of simulations to run
        
    Returns:
        Array of simulated predictions
    """
    try:
        predictions = []
        for _ in range(n_simulations):
            # Add random noise to input
            noise = np.random.normal(0, 0.01, X.shape)
            X_noisy = X + noise
            
            # Get prediction
            pred = predict_with_model('lstm', model, X_noisy)
            predictions.append(pred)
            
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate mean prediction
        mean_pred = np.mean(predictions, axis=0)
        
        # Ensure valid prediction
        return ensure_valid_prediction(mean_pred)
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        return ensure_valid_prediction([])

def save_predictions(predictions: List[List[int]], file_path: str) -> None:
    """Save predictions to a file.
    
    Args:
        predictions: List of prediction lists
        file_path: Path to save predictions
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save predictions
        with open(file_path, 'w') as f:
            json.dump(predictions, f, indent=4)
            
        logger.info(f"Saved predictions to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise

def predict_next_draw(model: Any, X: np.ndarray, n_predictions: int = 1) -> np.ndarray:
    """Predict next lottery draw.
    
    Args:
        model: Model to use for predictions
        X: Input features
        n_predictions: Number of predictions to generate
        
    Returns:
        Array of predictions
    """
    try:
        predictions = []
        for _ in range(n_predictions):
            # Add random noise to input
            noise = np.random.normal(0, 0.01, X.shape)
            X_noisy = X + noise
            
            # Get prediction
            pred = predict_with_model('lstm', model, X_noisy)
            predictions.append(pred)
            
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate mean prediction
        mean_pred = np.mean(predictions, axis=0)
        
        # Ensure valid prediction
        return ensure_valid_prediction(mean_pred)
        
    except Exception as e:
        logger.error(f"Error predicting next draw: {str(e)}")
        return ensure_valid_prediction([])

def calculate_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for lottery number predictions.
    
    Args:
        y_true: True lottery numbers (normalized between 0 and 1)
        y_pred: Predicted lottery numbers (normalized between 0 and 1)
        
    Returns:
        Dictionary of metrics
    """
    # Denormalize predictions
    y_true_denorm = np.round(y_true * 49).astype(int)
    y_pred_denorm = np.round(y_pred * 49).astype(int)
    
    # Calculate metrics
    metrics = {
        'mse': np.mean((y_true - y_pred) ** 2),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'exact_matches': np.mean(np.all(y_true_denorm == y_pred_denorm, axis=1)),
        'partial_matches': np.mean(np.sum(y_true_denorm == y_pred_denorm, axis=1) / y_true.shape[1])
    }
    
    return metrics

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate metrics between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Ensure arrays are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate basic metrics
        metrics = {
            'mse': np.mean((y_true - y_pred) ** 2),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'r2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
            'explained_variance': np.var(y_pred) / np.var(y_true)
        }
        
        # Calculate lottery-specific metrics
        correct_numbers = np.sum(np.isin(y_pred, y_true))
        metrics.update({
            'correct_numbers': correct_numbers,
            'accuracy': correct_numbers / len(y_true)
        })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {}

def backtest(model: Any, X: np.ndarray, y: np.ndarray, window_size: int = 10) -> Dict[str, float]:
    """Perform backtesting on the model.
    
    Args:
        model: Model to backtest
        X: Input features
        y: Target values
        window_size: Size of rolling window
        
    Returns:
        Dictionary of backtest metrics
    """
    metrics = []
    
    # Perform rolling window validation
    for i in range(0, len(X) - window_size, window_size):
        # Get window data
        X_window = X[i:i + window_size]
        y_window = y[i:i + window_size]
        
        # Make predictions
        y_pred = model.predict(X_window)
        
        # Calculate metrics
        window_metrics = calculate_prediction_metrics(y_window, y_pred)
        metrics.append(window_metrics)
    
    # Average metrics across windows
    avg_metrics = {}
    for key in metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics])
    
    return avg_metrics

def get_model_weights() -> Dict[str, float]:
    """Get weights for each model in the ensemble.
    
    Returns:
        Dictionary of model weights
    """
    # For now, use equal weights for all models
    weights = {
        'lstm': 0.4,
        'xgboost': 0.3,
        'lightgbm': 0.3
    }
    
    return weights 