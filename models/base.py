from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .utils import ensure_valid_prediction, log_training_errors

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> List[int]:
        """Make predictions on the given data."""
        pass

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