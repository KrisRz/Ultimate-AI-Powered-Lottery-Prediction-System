import numpy as np
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from typing import List, Tuple, Union
from scripts.utils.model_utils import ensure_valid_prediction, log_training_errors
from sklearn.preprocessing import StandardScaler
from numpy import ndarray as NDArray
import pickle
from pathlib import Path

def check_stationarity(series: np.ndarray, significance_level: float = 0.05) -> Tuple[bool, float]:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test
    
    Args:
        series: Time series data
        significance_level: p-value threshold for stationarity
        
    Returns:
        Tuple of (is_stationary, p_value)
    """
    try:
        # Run ADF test
        result = adfuller(series)
        p_value = result[1]
        
        # Check if p-value indicates stationarity
        is_stationary = p_value < significance_level
        
        return is_stationary, p_value
    except Exception as e:
        logging.warning(f"Error checking stationarity: {str(e)}")
        return False, 1.0  # Assume non-stationary if test fails

def difference_series(series: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Apply differencing to a time series to make it stationary
    
    Args:
        series: Time series data
        order: Differencing order
        
    Returns:
        Differenced series with original values stored
    """
    # Store original values needed for later forecasting
    original_values = series[-order:].copy()
    
    # Apply differencing
    diffed_series = np.diff(series, n=order)
    
    # Create a custom class to store both series and original values
    class DiffedSeries(np.ndarray):
        def __new__(cls, input_array, original_values):
            obj = np.asarray(input_array).view(cls)
            obj.original_values = original_values
            return obj
            
    return DiffedSeries(diffed_series, original_values)

def inverse_difference(forecast: np.ndarray, original_values: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Invert differencing to get back to the original scale
    
    Args:
        forecast: Differenced forecast values
        original_values: Original values used for reconstruction
        order: Differencing order that was applied
        
    Returns:
        Forecast in the original scale
    """
    # For first-order differencing
    if order == 1:
        reconstructed = np.zeros(len(forecast))
        reconstructed[0] = forecast[0] + original_values[-1]
        for i in range(1, len(forecast)):
            reconstructed[i] = forecast[i] + reconstructed[i-1]
        return reconstructed
        
    # For higher-order differencing (if needed)
    elif order == 2:
        # For second-order differencing, we need the last two values
        reconstructed = np.zeros(len(forecast))
        reconstructed[0] = forecast[0] + 2*original_values[-1] - original_values[-2]
        reconstructed[1] = forecast[1] + 2*reconstructed[0] - original_values[-1]
        for i in range(2, len(forecast)):
            reconstructed[i] = forecast[i] + 2*reconstructed[i-1] - reconstructed[i-2]
        return reconstructed
    
    # Default fallback
    else:
        logging.warning(f"Unsupported differencing order: {order}, returning undifferenced forecast")
        return forecast

@log_training_errors
def train_holtwinters_model(X_train: np.ndarray, y_train: np.ndarray, check_for_stationarity: bool = True) -> List[ExponentialSmoothing]:
    """
    Train Holt-Winters model for lottery prediction.
    
    Args:
        X_train: Feature matrix (not used, but kept for API consistency)
        y_train: Target matrix with shape [n_samples, 6]
        check_for_stationarity: Whether to check for stationarity
        
    Returns:
        List of trained models
    """
    # Input validation
    if not isinstance(y_train, np.ndarray):
        raise ValueError("y_train must be a numpy array")
        
    if y_train.size == 0:
        raise ValueError("Empty input array")
        
    if not np.issubdtype(y_train.dtype, np.number):
        raise ValueError("Input array must contain numeric values")
        
    if len(y_train.shape) != 2 or y_train.shape[1] != 6:
        raise ValueError(f"Expected y_train shape [n_samples, 6], got {y_train.shape}")
        
    if len(y_train) < 2:
        raise ValueError("Need at least 2 samples for training")
    
    models = []
    
    for i in range(6):
        series = y_train[:, i]
        
        # Check stationarity
        if check_for_stationarity:
            is_stationary, _ = check_stationarity(series)
            if not is_stationary:
                # Apply first difference
                series = np.diff(series)
                
                # Check if still non-stationary
                if not check_stationarity(series)[0]:
                    # Apply second difference
                    series = np.diff(series)
        
        # Train model
        model = ExponentialSmoothing(
            series,
            seasonal_periods=12,
            trend='add',
            seasonal='add'
        ).fit()
        
        models.append(model)
    
    return models

def predict_holtwinters_model(models: List[ExponentialSmoothing], X: np.ndarray) -> np.ndarray:
    """
    Make predictions using trained Holt-Winters models
    
    Args:
        models: List of trained Holt-Winters models
        X: Input features (used only for determining prediction length)
        
    Returns:
        Array of predicted numbers with shape [n_samples, 6]
    """
    try:
        n_samples = 1 if len(X.shape) == 1 else X.shape[0]
        predictions = np.zeros((n_samples, 6))
        
        for i, model in enumerate(models):
            # Get forecast for each model
            forecast = model.forecast(n_samples)
            predictions[:, i] = forecast
            
        return ensure_valid_prediction(predictions)
        
    except Exception as e:
        logging.error(f"Error in Holt-Winters model prediction: {str(e)}")
        raise

logger = logging.getLogger(__name__)

class HoltWintersModel:
    def __init__(self, name='holtwinters', **kwargs):
        self.name = name
        self.models = []  # One model per output dimension
        self.config = {
            'seasonal_periods': 6,
            'trend': 'add',
            'seasonal': 'add',
            'damped_trend': True,
            'initialization_method': 'estimated'
        }
        self.config.update(kwargs)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Holt-Winters model for each output dimension"""
        try:
            # Handle multivariate output
            if len(y_train.shape) > 1:
                n_outputs = y_train.shape[1]
            else:
                n_outputs = 1
                y_train = y_train.reshape(-1, 1)
            
            # Train a model for each output
            self.models = []
            for i in range(n_outputs):
                model = ExponentialSmoothing(
                    y_train[:, i],
                    seasonal_periods=self.config['seasonal_periods'],
                    trend=self.config['trend'],
                    seasonal=self.config['seasonal'],
                    damped_trend=self.config['damped_trend'],
                    initialization_method=self.config['initialization_method']
                )
                
                fitted = model.fit()
                self.models.append(fitted)
                
            logger.info(f"Trained {len(self.models)} Holt-Winters models")
            
        except Exception as e:
            logger.error(f"Error training Holt-Winters model: {str(e)}")
            raise
            
    def predict(self, X):
        """Generate predictions for each output dimension"""
        if not self.models:
            raise ValueError("Model not trained yet")
            
        try:
            predictions = []
            for model in self.models:
                pred = model.forecast(steps=len(X))
                predictions.append(pred)
                
            return np.column_stack(predictions)
            
        except Exception as e:
            logger.error(f"Error predicting with Holt-Winters: {str(e)}")
            raise
            
    def evaluate(self, X_val, y_val):
        """Evaluate model on validation data"""
        try:
            y_pred = self.predict(X_val)
            return np.mean(np.abs(y_pred - y_val))
        except Exception as e:
            logger.error(f"Error evaluating Holt-Winters: {str(e)}")
            return float('inf')
            
    def save(self, path):
        """Save trained models"""
        if not self.models:
            raise ValueError("No trained models to save")
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.models, f)
        except Exception as e:
            logger.error(f"Error saving Holt-Winters model: {str(e)}")
            raise
            
    def load(self, path):
        """Load trained models"""
        try:
            with open(path, 'rb') as f:
                self.models = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading Holt-Winters model: {str(e)}")
            raise