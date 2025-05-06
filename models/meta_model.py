"""Meta model that combines predictions from other models."""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Union
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

def train_meta_model(X: np.ndarray, y: np.ndarray, base_predictions: Dict[str, np.ndarray], meta_params: Dict = None) -> Tuple[StackingRegressor, StandardScaler]:
    """
    Train a meta model using predictions from base models.
    
    Args:
        X: Training features
        y: Target values
        base_predictions: Dictionary of base model predictions
        meta_params: Optional parameters for meta model
        
    Returns:
        Tuple of (trained meta model, scaler)
    """
    try:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Prepare meta-features
        meta_features = np.hstack([
            pred.reshape(len(X), -1) 
            for pred in base_predictions.values()
        ])
        meta_features = np.hstack([meta_features, X_scaled])
        
        # Configure meta model
        default_params = {
            'estimators': [('ridge', Ridge())],
            'final_estimator': Ridge(),
            'cv': 5,
            'n_jobs': -1
        }
        
        if meta_params:
            default_params.update(meta_params)
            
        # Train separate model for each output
        models = []
        for i in range(y.shape[1]):
            model = StackingRegressor(**default_params)
            model.fit(meta_features, y[:, i])
            models.append(model)
            
        # Create a wrapper model that combines predictions
        class MultiOutputStackingRegressor(StackingRegressor):
            def __init__(self, models, **kwargs):
                super().__init__(**kwargs)
                self.models = models
                self.__dict__.update(models[0].__dict__)
                
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models]).T
                predictions = np.clip(np.round(predictions), 1, 59)
                
                # Ensure unique numbers by replacing duplicates
                for i in range(len(predictions)):
                    while len(set(predictions[i])) < 6:
                        unique_nums = set(predictions[i])
                        needed = 6 - len(unique_nums)
                        candidates = set(range(1, 60)) - unique_nums
                        replacements = np.random.choice(list(candidates), size=needed, replace=False)
                        predictions[i] = np.sort(list(unique_nums) + list(replacements))
                        
                return predictions
                
        wrapper_model = MultiOutputStackingRegressor(models, **default_params)
        return wrapper_model, scaler
        
    except Exception as e:
        logger.error(f"Error training meta model: {str(e)}")
        return None

def predict_meta_model(meta_model, base_models, X_test):
    """
    Generate predictions using the meta model
    
    Args:
        meta_model: Trained meta model
        base_models: Dictionary of base models and their prediction functions
        X_test: Test features
        
    Returns:
        Array of predicted numbers
    """
    try:
        # Generate base model predictions
        base_predictions = []
        for model_name, (model, predict_fn) in base_models.items():
            if model_name == 'lstm':
                # Reshape for LSTM
                X_test_lstm = X_test.reshape(-1, 1, X_test.shape[-1])
                pred = predict_fn(model, X_test_lstm, None)  # Scaler is handled inside predict_fn
            else:
                pred = predict_fn(model, X_test)
            base_predictions.append(pred)
            
        # Stack predictions
        stacked_predictions = np.hstack(base_predictions)
        
        # Generate meta predictions
        meta_predictions = meta_model.predict(stacked_predictions)
        
        # Ensure valid predictions
        meta_predictions = np.clip(np.round(meta_predictions), 1, 59)
        
        # Process each prediction to ensure uniqueness
        valid_predictions = []
        for pred in meta_predictions:
            # Get unique numbers and sort
            unique_nums = sorted(list(set(pred)))
            
            # Ensure exactly 6 numbers
            if len(unique_nums) > 6:
                unique_nums = unique_nums[:6]
            while len(unique_nums) < 6:
                available = list(set(range(1, 60)) - set(unique_nums))
                unique_nums.append(np.random.choice(available))
                unique_nums.sort()
            
            valid_predictions.append(unique_nums)
        
        return np.array(valid_predictions)
        
    except Exception as e:
        logging.error(f"Error in meta model prediction: {str(e)}")
        raise

def predict_meta_model_with_dict(model: Any, X: Dict[str, np.ndarray], max_number: int = 59, n_numbers: int = 6) -> np.ndarray:
    """
    Generate meta model prediction from base model predictions.
    
    Args:
        model: Meta model
        X: Dictionary containing base model predictions
        max_number: Maximum lottery number
        n_numbers: Number of numbers to predict
        
    Returns:
        Array of predicted numbers
    """
    try:
        # Handle dict input for compatibility
        if isinstance(X, dict):
            if 'feat' in X:
                X_use = X['feat']
            elif 'ts' in X:
                X_use = X['ts']
            else:
                raise ValueError("Input dict must contain 'feat' or 'ts' key")
        else:
            X_use = X
        
        # Generate meta prediction
        meta_pred = model.predict(X_use)
        
        # Ensure valid prediction
        meta_pred = ensure_valid_prediction(meta_pred, max_number, n_numbers)
        
        return meta_pred
        
    except Exception as e:
        logger.error(f"Error generating meta prediction: {str(e)}")
        # Return a fallback prediction
        return np.array(sorted(np.random.choice(range(1, max_number+1), size=n_numbers, replace=False)))

def update_models(models: Dict[str, Tuple[Any, Any]], X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[Any, Any]]:
    """
    Update existing models with new data.
    
    Args:
        models: Dictionary of model name to (model, predict_function) tuples
        X: New training data
        y: New target values
        
    Returns:
        Dictionary of updated models
    """
    try:
        updated_models = {}
        for name, (model, predict_fn) in models.items():
            if hasattr(model, 'fit'):
                model.fit(X, y)
            updated_models[name] = (model, predict_fn)
        return updated_models
    except Exception as e:
        logger.error(f"Error updating models: {str(e)}")
        return models 

class MetaModel:
    """Meta model that combines predictions from other models."""
    
    def __init__(self, name='meta', **kwargs):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.base_models = []
        self.use_simple_averaging = False
        
        # Default configuration
        self.config = {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'early_stopping': {
                'patience': 10,
                'restore_best_weights': True
            },
            'reduce_lr': {
                'patience': 5,
                'factor': 0.5,
                'min_lr': 1e-6
            }
        }
        self.config.update(kwargs)
    
    def add_base_model(self, model):
        """Add a base model to the ensemble"""
        self.base_models.append(model)
    
    def train(self, X, y, X_val=None, y_val=None):
        """Train the meta model"""
        try:
            if not self.base_models:
                logger.warning("No base models available for meta-features. Using simple averaging.")
                self.use_simple_averaging = True
                return
            
            # Generate meta-features from base models
            meta_features = []
            for model in self.base_models:
                if hasattr(model, 'predict'):
                    preds = model.predict(X)
                    meta_features.append(preds)
            
            if not meta_features:
                logger.warning("No predictions available from base models. Using simple averaging.")
                self.use_simple_averaging = True
                return
            
            # Stack meta-features
            X_meta = np.hstack(meta_features)
            
            # Create and train meta model
            if self.model is None:
                self.model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_meta.shape[1],)),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                self.model.compile(
                    optimizer=Adam(learning_rate=self.config['learning_rate']),
                    loss='mse',
                    metrics=['mae']
                )
            
            # Train model
            validation_data = None
            if X_val is not None and y_val is not None:
                val_meta_features = []
                for model in self.base_models:
                    if hasattr(model, 'predict'):
                        val_preds = model.predict(X_val)
                        val_meta_features.append(val_preds)
                if val_meta_features:
                    X_val_meta = np.hstack(val_meta_features)
                    validation_data = (X_val_meta, y_val)
            
            history = self.model.fit(
                X_meta, y,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=validation_data,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.config['early_stopping']['patience'],
                        restore_best_weights=self.config['early_stopping']['restore_best_weights']
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=self.config['reduce_lr']['factor'],
                        patience=self.config['reduce_lr']['patience'],
                        min_lr=self.config['reduce_lr']['min_lr']
                    )
                ],
                verbose=1
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Error fitting meta model: {str(e)}")
            self.use_simple_averaging = True
        
    def predict(self, X):
        """Generate predictions using meta model"""
        try:
            if not self.base_models:
                raise ValueError("No base models available for predictions")
            
            # Get predictions from base models
            meta_features = []
            for model in self.base_models:
                pred = model.predict(X)
                # Ensure 2D shape
                if len(pred.shape) > 2:
                    pred = pred.reshape(pred.shape[0], -1)
                meta_features.append(pred)
            
            # Combine predictions into meta features
            X_meta = np.hstack(meta_features)
            
            # Scale features
            X_meta = self.scaler.transform(X_meta)
            
            # Generate predictions
            return self.model.predict(X_meta)
            
        except Exception as e:
            logger.error(f"Error generating meta model predictions: {str(e)}")
            raise
        
    def update(self, X: np.ndarray, y: np.ndarray, models: Dict[str, Tuple[Any, Any]]) -> None:
        """Update model with new data"""
        self.base_models = update_models(models, X, y)
        
    def evaluate(self, X_val, y_val):
        """Evaluate the meta model"""
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() first.")
        predictions = self.predict(X_val)
        return np.mean((predictions - y_val) ** 2)  # MSE
        
    def save(self, save_dir: str) -> None:
        """Save the meta model"""
        if self.model is None:
            raise ValueError("Model not trained. Nothing to save.")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "meta_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open(Path(save_dir) / "meta_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def load(self, model_dir: str) -> None:
        """Load the meta model"""
        model_path = Path(model_dir) / "meta_model.pkl"
        scaler_path = Path(model_dir) / "meta_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_dir}")
            
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f) 