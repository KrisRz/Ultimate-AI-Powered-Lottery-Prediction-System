"""Base model classes for lottery prediction."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.scaler = None
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_data: Optional[Tuple] = None) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model."""
        pass

class TimeSeriesModel(BaseModel):
    """Base class for time series models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.look_back = config.get('look_back', 10) if config else 10
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series sequences."""
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back)])
            y.append(data[i + self.look_back])
        return np.array(X), np.array(y)

class EnsembleModel(BaseModel):
    """Base class for ensemble models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.models = {}
        self.weights = {}
    
    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set model weights."""
        if not all(name in self.models for name in weights):
            raise ValueError("Weight dictionary contains unknown model names")
        self.weights = weights
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_data: Optional[Tuple] = None) -> None:
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            try:
                logger.info(f"Training model: {name}")
                model.train(X_train, y_train, validation_data)
            except Exception as e:
                logger.error(f"Error training model {name}: {str(e)}")
                raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted ensemble predictions."""
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            try:
                weight = self.weights[name] / total_weight
                pred = model.predict(X)
                predictions.append(weight * pred)
            except Exception as e:
                logger.error(f"Error predicting with model {name}: {str(e)}")
                raise
        
        return np.sum(predictions, axis=0)
    
    def save(self, path: str) -> None:
        """Save all models in the ensemble."""
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            try:
                model_path = base_path / f"{name}.joblib"
                model.save(str(model_path))
            except Exception as e:
                logger.error(f"Error saving model {name}: {str(e)}")
                raise
        
        # Save weights
        weights_path = base_path / "weights.joblib"
        joblib.dump(self.weights, weights_path)
    
    def load(self, path: str) -> None:
        """Load all models in the ensemble."""
        base_path = Path(path)
        
        # Load weights
        weights_path = base_path / "weights.joblib"
        if weights_path.exists():
            self.weights = joblib.load(weights_path)
        
        for name, model in self.models.items():
            try:
                model_path = base_path / f"{name}.joblib"
                model.load(str(model_path))
            except Exception as e:
                logger.error(f"Error loading model {name}: {str(e)}")
                raise 