"""
Ensemble model for lottery prediction.

This file contains the core LotteryEnsemble class that implements:
1. The ensemble model structure and architecture
2. Prediction methods for combining multiple model outputs
3. Validation and evaluation functionality
4. Weight management for different models in the ensemble

This is the MODEL DEFINITION file, whereas scripts/train_ensemble.py 
contains the TRAINING LOGIC for this ensemble.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.base import BaseEstimator
from .utils.model_validation import ModelValidator
from .utils.model_utils import evaluate_model

logger = logging.getLogger(__name__)

class LotteryEnsemble(BaseEstimator):
    """Ensemble model for lottery prediction."""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        """Initialize ensemble with models and their weights."""
        self.models = models
        self.weights = weights
        self.validator = ModelValidator(self)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LotteryEnsemble':
        """Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Self
        """
        for model in self.models.values():
            model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted predictions based on the outputs of all models.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Weighted average of all model predictions
        """
        if not self.models:
            print("No models in ensemble, returning empty predictions")
            return np.zeros((X.shape[0], self.output_size))
            
        # Track predicted values and weights used
        all_predictions = []
        used_weights = []
        total_weight = 0
        
        # Target shape for all predictions (will be determined by first successful prediction)
        target_shape = None
        
        for i, (model_key, model) in enumerate(self.models.items()):
            if model is None:
                continue
                
            try:
                # Skip model if weight is zero
                if self.weights.get(model_key, 0) <= 0:
                    continue
                    
                # Prepare input based on model type
                X_model = X.copy()
                
                # Process data based on model type (LSTM needs 3D, XGBoost needs 2D, etc.)
                if model_key in ['LSTM', 'CNN_LSTM']:
                    # For LSTM and CNN_LSTM
                    if len(X_model.shape) == 2:
                        # If 2D, reshape to 3D
                        if 'LSTM' in model_key:
                            # For regular LSTM, reshape to (samples, time_steps, features)
                            if X_model.shape[1] % self.input_shape[1] == 0:
                                features = X_model.shape[1] // self.input_shape[1]
                                X_model = X_model.reshape(X_model.shape[0], self.input_shape[1], features)
                            else:
                                # Default fallback reshape
                                X_model = X_model.reshape(X_model.shape[0], -1, 1)
                        else:
                            # CNN_LSTM may need special handling depending on its architecture
                            X_model = X_model.reshape(X_model.shape[0], -1, 1)
                elif model_key in ['XGBoost', 'LightGBM']:
                    # For tree-based models
                    if len(X_model.shape) == 3:
                        # Flatten 3D to 2D
                        X_model = X_model.reshape(X_model.shape[0], -1)
                
                # Make prediction
                prediction = model.predict(X_model)
                
                # Process prediction shape
                if len(prediction.shape) == 1:
                    # 1D prediction, reshape to 2D
                    prediction = prediction.reshape(-1, 1)
                
                # Set target shape from first successful prediction
                if target_shape is None:
                    if len(prediction.shape) == 2:
                        target_shape = prediction.shape
                    else:
                        # Handle unexpected prediction shape
                        target_shape = (X.shape[0], self.output_size)
                
                # Ensure prediction has the right shape for ensemble
                if prediction.shape != target_shape:
                    # Try to reshape or broadcast
                    if len(prediction.shape) == 2 and prediction.shape[0] == target_shape[0]:
                        # Same number of samples but different output dimension
                        # This can happen with different model output shapes
                        if prediction.shape[1] == 1 and target_shape[1] > 1:
                            # Broadcast single prediction to multiple columns
                            prediction = np.repeat(prediction, target_shape[1], axis=1)
                        elif prediction.shape[1] > 1 and target_shape[1] == 1:
                            # Average multiple predictions to a single column
                            prediction = np.mean(prediction, axis=1, keepdims=True)
                        elif prediction.shape[1] != target_shape[1]:
                            # Try to resize to target shape
                            try:
                                prediction = np.resize(prediction, target_shape)
                            except:
                                print(f"Cannot reshape prediction {prediction.shape} to {target_shape}")
                                continue
                    else:
                        # Different number of samples or more complex shape issue
                        print(f"Incompatible prediction shape: {prediction.shape} vs target {target_shape}")
                        continue
                
                # Apply weight to prediction
                weight = self.weights.get(model_key, 1.0)
                weighted_pred = prediction * weight
                all_predictions.append(weighted_pred)
                used_weights.append(weight)
                total_weight += weight
                
            except Exception as e:
                print(f"Error getting prediction from {model_key}: {str(e)}")
                continue
        
        # If no predictions were made, return default
        if not all_predictions:
            print("No models could generate predictions, returning zeros")
            return np.zeros((X.shape[0], self.output_size))
        
        # Stack all predictions
        stacked_predictions = np.array(all_predictions)
        
        # Combine predictions with proper normalization
        if total_weight > 0:
            # Normalize weights if we have valid predictions
            final_prediction = sum(all_predictions) / total_weight
        else:
            # Equal weighting as fallback
            final_prediction = np.mean(stacked_predictions, axis=0)
        
        return final_prediction
        
    def predict_next_draw(self, X: np.ndarray) -> np.ndarray:
        """Predict next draw numbers."""
        try:
            # Make prediction
            pred = self.predict(X)
            
            # Denormalize (multiply by max lottery number and round to nearest integer)
            max_number = 59  # UK Lottery uses 1-59
            
            # Ensure pred is not empty
            if pred.size == 0:
                return np.array([1, 2, 3, 4, 5, 6])  # Default return if prediction is empty
                
            # Handle different prediction shapes
            if len(pred.shape) > 1 and pred.shape[0] > 0:
                numbers = np.round(pred[0] * max_number).astype(int)
            else:
                numbers = np.round(pred * max_number).astype(int)
            
            # Ensure numbers are within valid range and unique
            valid_numbers = []
            for num in numbers:
                # Ensure within valid range
                num = max(1, min(num, max_number))
                # Ensure uniqueness
                if num not in valid_numbers:
                    valid_numbers.append(num)
            
            # If we don't have enough numbers, add some valid ones
            while len(valid_numbers) < 6:
                # Find an unused number
                for num in range(1, max_number + 1):
                    if num not in valid_numbers:
                        valid_numbers.append(num)
                        break
            
            # If we have too many numbers, keep the first 6
            if len(valid_numbers) > 6:
                valid_numbers = valid_numbers[:6]
            
            # Sort numbers
            valid_numbers.sort()
            
            return np.array(valid_numbers)
            
        except Exception as e:
            logger.error(f"Error in predict_next_draw: {str(e)}")
            # Return a default prediction if something goes wrong
            return np.array([1, 2, 3, 4, 5, 6])
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble model performance.
        
        Args:
            X: Test features
            y: True values
            
        Returns:
            Dictionary of evaluation metrics
        """
        return evaluate_model(self, X, y)
        
    def validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run comprehensive model validation.
        
        Args:
            X: Validation features
            y: True values
            
        Returns:
            Dictionary of validation metrics
        """
        return self.validator.validate_all(X, y)
        
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set model weights.
        
        Args:
            weights: Dictionary of model weights {name: weight}
        """
        if not all(name in self.models for name in weights):
            raise ValueError("Weight keys must match model names")
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()} 