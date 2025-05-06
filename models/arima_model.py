import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import warnings
from scripts.utils.model_utils import ensure_valid_prediction
from typing import List, Dict, Any, Optional, Union, Tuple
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pmdarima import auto_arima

logger = logging.getLogger(__name__)

def train_arima_model(X: np.ndarray, y: np.ndarray, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Train ARIMA models for lottery number prediction
    
    Args:
        X: Training features
        y: Target numbers (array of shape [n_samples, 6])
        order: ARIMA order parameters (p,d,q)
        seasonal_order: Seasonal ARIMA parameters (P,D,Q,s)
        
    Returns:
        List of trained ARIMA models
    """
    # Convert numpy arrays to pandas Series for each target
    models = []
    for i in range(y.shape[1]):
        # Create time series data
        ts_data = pd.Series(y[:, i], index=pd.date_range(start='2020-01-01', periods=len(y)))
        
        # Train ARIMA model
        try:
            model = ARIMA(ts_data, order=order)
            model = model.fit()
            models.append(model)
        except Exception as e:
            logging.error(f"Error training ARIMA model for target {i}: {str(e)}")
            raise
            
    return models

def predict_arima_model(models: List[ARIMA], X: np.ndarray) -> np.ndarray:
    """
    Generate predictions using trained ARIMA models
    
    Args:
        models: List of trained ARIMA models (one for each number position)
        X: Input features to predict on
        
    Returns:
        Array of predicted numbers
    """
    # Make predictions for each number position
    predictions = []
    for model in models:
        # Get one-step ahead forecast
        pred = model.forecast(steps=len(X))
        predictions.append(pred)
    
    # Stack predictions and round to integers
    predictions = np.round(np.column_stack(predictions)).astype(int)
    
    # Ensure predictions are within valid range
    predictions = np.clip(predictions, 1, 59)
    
    return predictions

class ARIMAModel:
    def __init__(self, name='arima', **kwargs):
        self.name = name
        self.models = []  # List of models for each output
        self.model_fits = []  # List of fitted models
        self.scaler = StandardScaler()
        self.order = (5, 1, 0)  # Default order
        self.seasonal_order = (1, 1, 0, 12)  # Default seasonal order
        
        # Update parameters if provided
        if 'order' in kwargs:
            self.order = kwargs['order']
        if 'seasonal_order' in kwargs:
            self.seasonal_order = kwargs['seasonal_order']
        
    def train(self, X, y, X_val=None, y_val=None, config=None):
        """Train the ARIMA model"""
        try:
            # Reshape 4D input to 2D
            X_reshaped = X.reshape(-1, X.shape[-1])
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X_reshaped)
            
            # Train a model for each output
            self.models = []
            self.model_fits = []
            for i in range(y.shape[1]):
                # Create time series data
                ts_data = pd.Series(y[:, i], index=pd.date_range(start='2020-01-01', periods=len(y)))
                
                # Fit ARIMA model
                model = ARIMA(ts_data, order=self.order, seasonal_order=self.seasonal_order)
                model_fit = model.fit()
                
                self.models.append(model)
                self.model_fits.append(model_fit)
            
            return self.model_fits
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {str(e)}")
            raise

    def predict(self, X):
        """Generate predictions for each lottery number."""
        if not self.model_fits:
            raise ValueError("Model not trained. Call train() first.")
            
        try:
            # Make predictions for each number position
            predictions = []
            for model_fit in self.model_fits:
                # Get one-step ahead forecast
                pred = model_fit.forecast(steps=len(X))
                predictions.append(pred)
            
            # Stack predictions and round to integers
            predictions = np.round(np.column_stack(predictions)).astype(int)
            
            # Ensure predictions are within valid range
            predictions = np.clip(predictions, 1, 59)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {str(e)}")
            raise

    def evaluate(self, X_val, y_val):
        """Evaluate model on validation data"""
        try:
            y_pred = self.predict(X_val)
            return np.mean(np.abs(y_pred - y_val))
        except Exception as e:
            logger.error(f"Error evaluating ARIMA: {str(e)}")
            return float('inf')
            
    def save(self, path):
        """Save trained models"""
        if not self.model_fits:
            raise ValueError("No trained models to save")
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model_fits, f)
        except Exception as e:
            logger.error(f"Error saving ARIMA model: {str(e)}")
            raise
            
    def load(self, path):
        """Load trained models"""
        try:
            with open(path, 'rb') as f:
                self.model_fits = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading ARIMA model: {str(e)}")
            raise