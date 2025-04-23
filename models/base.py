from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .utils import ensure_valid_prediction, log_training_errors
import logging
import json

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.training_errors = []
        self.validation_scores = []

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> List[int]:
        """Make predictions on the given data."""
        pass

    def handle_training_error(self, error: Exception) -> bool:
        """Log training error and attempt recovery"""
        error_msg = f"{self.name} training error: {str(error)}"
        self.training_errors.append(error_msg)
        logging.error(error_msg)
        
        # Attempt recovery based on error type
        if isinstance(error, (ValueError, TypeError)):
            # Data-related errors - try to fix data
            return self.attempt_data_recovery()
        elif isinstance(error, MemoryError):
            # Memory issues - try to free memory
            import gc
            gc.collect()
            return True
        return False
        
    def attempt_data_recovery(self) -> bool:
        """Try to recover from data-related errors"""
        try:
            # Implement recovery logic
            return True
        except:
            return False
            
    def validate_predictions(self, predictions: np.ndarray, full_validation: bool = True) -> bool:
        """
        Validate model predictions thoroughly
        
        Args:
            predictions: Model predictions to validate
            full_validation: Whether to validate all predictions or just sample
            
        Returns:
            bool: Whether predictions are valid
        """
        try:
            if predictions is None:
                return False
                
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
                
            # Shape validation
            if len(predictions.shape) != 2 or predictions.shape[1] != 6:
                return False
                
            # Value validation
            if not np.all((predictions >= 1) & (predictions <= 59)):
                return False
                
            # Uniqueness validation
            if full_validation:
                validate_range = range(len(predictions))
            else:
                # Validate random sample of 100 predictions if full_validation=False
                validate_range = np.random.choice(len(predictions), min(100, len(predictions)), replace=False)
                
            for i in validate_range:
                if len(set(predictions[i])) != 6:
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Prediction validation error: {str(e)}")
            return False
            
    def save_model_state(self) -> None:
        """Save model state for potential recovery"""
        try:
            model_state = {
                'is_trained': self.is_trained,
                'training_errors': self.training_errors,
                'validation_scores': self.validation_scores
            }
            with open(f'models/checkpoints/{self.name}_state.json', 'w') as f:
                json.dump(model_state, f)
        except Exception as e:
            logging.error(f"Error saving model state: {str(e)}")
            
    def load_model_state(self) -> bool:
        """Load saved model state"""
        try:
            with open(f'models/checkpoints/{self.name}_state.json', 'r') as f:
                state = json.load(f)
            self.is_trained = state['is_trained']
            self.training_errors = state['training_errors']
            self.validation_scores = state['validation_scores']
            return True
        except:
            return False

    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass

    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass

class TimeSeriesModel(BaseModel):
    def __init__(self, name: str, look_back: int = 200):
        super().__init__(name)
        self.look_back = look_back

    def prepare_sequence(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for time series models."""
        X, y = [], []
        for i in range(len(data) - self.look_back - 6):
            X.append(data[i:i + self.look_back])
            y.append(data[i + self.look_back:i + self.look_back + 6])
        return np.array(X), np.array(y)

class EnsembleModel(BaseModel):
    def __init__(self, name: str, models: Dict[str, BaseModel]):
        super().__init__(name)
        self.models = models

    def train(self, df: pd.DataFrame) -> None:
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            model.train(df)

    def predict(self, df: pd.DataFrame) -> List[int]:
        """Make predictions using all models and combine results."""
        predictions = []
        for model in self.models.values():
            pred = model.predict(df)
            predictions.append(pred)
        return self.combine_predictions(predictions)

    @abstractmethod
    def combine_predictions(self, predictions: List[List[int]]) -> List[int]:
        """Combine predictions from multiple models."""
        pass

class MetaModel(EnsembleModel):
    def __init__(self, name: str, models: Dict[str, BaseModel]):
        super().__init__(name, models)
        self.meta_model = None

    def train(self, df: pd.DataFrame) -> None:
        """Train base models and meta model."""
        super().train(df)
        self.train_meta_model(df)

    @abstractmethod
    def train_meta_model(self, df: pd.DataFrame) -> None:
        """Train the meta model on base model predictions."""
        pass

    def combine_predictions(self, predictions: List[List[int]]) -> List[int]:
        """Combine predictions using the meta model."""
        if self.meta_model is None:
            return super().combine_predictions(predictions)
        # Use meta model to combine predictions
        return ensure_valid_prediction(self.meta_model.predict(predictions)) 