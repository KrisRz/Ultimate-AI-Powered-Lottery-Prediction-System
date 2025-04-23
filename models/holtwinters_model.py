import numpy as np
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from typing import List, Tuple, Union
from .utils import log_training_errors, ensure_valid_prediction

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
        Differenced series
    """
    # Store original values needed for later forecasting
    original_values = series[-order:].copy()
    
    # Apply differencing
    diffed_series = np.diff(series, n=order)
    
    # Attach original values for reconstruction during forecasting
    # This is stored as an attribute of the numpy array
    diffed_series.original_values = original_values
    diffed_series.diff_order = order
    
    return diffed_series

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
def train_holtwinters_model(X_train, y_train, check_for_stationarity=True):
    """
    Train Holt-Winters Exponential Smoothing models for lottery prediction
    
    Args:
        X_train: Feature matrix (not used, but kept for API consistency)
        y_train: Target matrix with shape [n_samples, 6]
        check_for_stationarity: Whether to check and apply differencing for non-stationary series
        
    Returns:
        List of trained models and their configuration
    """
    # Validate input
    if y_train is None or (isinstance(y_train, np.ndarray) and y_train.size == 0):
        raise ValueError("y_train cannot be None or empty")
    
    if not isinstance(y_train, np.ndarray):
        try:
            y_train = np.array(y_train)
        except Exception as e:
            raise ValueError(f"Cannot convert y_train to numpy array: {str(e)}")
    
    # Handle case where y_train might be a 3D array from LSTM models
    if len(y_train.shape) == 3:
        logging.warning(f"Received 3D y_train with shape {y_train.shape}, using the last time step")
        y_train = y_train[:, -1, :]
    
    # Check shape
    if len(y_train.shape) != 2 or y_train.shape[1] != 6:
        raise ValueError(f"y_train must have shape [n_samples, 6], got {y_train.shape}")
    
    models = []
    diff_orders = []  # Store differencing order for each model
    
    for i in range(6):
        series = y_train[:, i]
        diff_order = 0
        
        # Check stationarity and apply differencing if needed
        if check_for_stationarity:
            is_stationary, p_value = check_stationarity(series)
            logging.info(f"Number position {i+1}: Stationarity p-value = {p_value:.4f}")
            
            if not is_stationary:
                logging.info(f"Number position {i+1} is non-stationary, applying differencing")
                series = difference_series(series, order=1)
                diff_order = 1
                
                # Check if first-order differencing was enough
                is_stationary_diff1, p_value_diff1 = check_stationarity(series)
                logging.info(f"After 1st-order diff: Stationarity p-value = {p_value_diff1:.4f}")
                
                # Apply second-order differencing if needed
                if not is_stationary_diff1:
                    logging.info(f"Number position {i+1} still non-stationary, applying 2nd-order differencing")
                    original_values = series.original_values  # Store for later reconstruction
                    series = difference_series(series, order=1)
                    # Update for 2nd-order differencing
                    series.original_values = np.concatenate([original_values, [original_values[-1] + series[0]]])
                    series.diff_order = 2
                    diff_order = 2
        
        try:
            # Fit Holt-Winters model
            model = ExponentialSmoothing(
                series, 
                trend="add",
                seasonal="add" if len(series) > 12 else None,  # Add seasonality if enough data
                seasonal_periods=12 if len(series) > 12 else None
            ).fit(optimized=True)
            
            # Store model and its differencing order
            models.append((model, diff_order, series.original_values if diff_order > 0 else None))
            diff_orders.append(diff_order)
            
            logging.info(f"Holt-Winters model for position {i+1} trained successfully")
            
        except Exception as e:
            logging.error(f"Error training Holt-Winters model for position {i+1}: {str(e)}")
            # Fallback to simpler model
            try:
                model = ExponentialSmoothing(series, trend="add").fit()
                models.append((model, diff_order, series.original_values if diff_order > 0 else None))
                diff_orders.append(diff_order)
            except Exception as e2:
                logging.error(f"Fallback model failed too: {str(e2)}")
                raise
    
    logging.info(f"Differencing orders applied: {diff_orders}")
    return models

def predict_holtwinters_model(models, X=None, n_predictions=1):
    """
    Generate predictions using trained Holt-Winters models
    
    Args:
        models: List of trained models with their differencing configuration
        X: Input features (not used but kept for API consistency)
        n_predictions: Number of time steps to forecast
        
    Returns:
        Array of predicted numbers with shape [n_predictions, 6]
    """
    try:
        predictions = np.zeros((n_predictions, 6))
        
        for i, (model, diff_order, original_values) in enumerate(models):
            # Generate forecasts
            forecast = model.forecast(n_predictions)
            
            # Invert differencing if applied during training
            if diff_order > 0:
                forecast = inverse_difference(forecast, original_values, order=diff_order)
            
            predictions[:, i] = forecast
        
        # Ensure valid predictions
        if n_predictions == 1:
            return ensure_valid_prediction(predictions[0])
        else:
            # For multiple predictions, ensure each row is valid
            valid_predictions = []
            for i in range(n_predictions):
                valid_predictions.append(ensure_valid_prediction(predictions[i]))
            return np.array(valid_predictions)
            
    except Exception as e:
        logging.error(f"Error in Holt-Winters prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6))