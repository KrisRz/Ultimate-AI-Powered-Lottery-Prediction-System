"""Model serialization utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import json
import os
from pathlib import Path
import joblib
import pickle
import onnx
import onnxruntime
import torch
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.models import save_model as save_keras_model

logger = logging.getLogger(__name__)

class ModelSerializer:
    """Serialize and deserialize models in various formats."""
    
    def __init__(self, model: Any, model_name: str):
        """Initialize model serializer."""
        self.model = model
        self.model_name = model_name
    
    def save(self, path: str, format: str = 'keras') -> str:
        """Save model in specified format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'keras':
            return self._save_keras(path)
        elif format == 'joblib':
            return self._save_joblib(path)
        elif format == 'pickle':
            return self._save_pickle(path)
        elif format == 'onnx':
            return self._save_onnx(path)
        elif format == 'torch':
            return self._save_torch(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_keras(self, path: Path) -> str:
        """Save model in Keras format."""
        if not isinstance(self.model, tf.keras.Model):
            raise ValueError("Model is not a Keras model")
        
        save_keras_model(self.model, str(path))
        return str(path)
    
    def _save_joblib(self, path: Path) -> str:
        """Save model in joblib format."""
        joblib.dump(self.model, str(path))
        return str(path)
    
    def _save_pickle(self, path: Path) -> str:
        """Save model in pickle format."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        return str(path)
    
    def _save_onnx(self, path: Path) -> str:
        """Save model in ONNX format."""
        if isinstance(self.model, tf.keras.Model):
            # Convert Keras model to ONNX
            import tf2onnx
            onnx_model, _ = tf2onnx.convert.from_keras(self.model)
            onnx.save(onnx_model, str(path))
        else:
            raise ValueError("ONNX conversion only supported for Keras models")
        return str(path)
    
    def _save_torch(self, path: Path) -> str:
        """Save model in PyTorch format."""
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), str(path))
        else:
            raise ValueError("Model is not a PyTorch model")
        return str(path)
    
    @staticmethod
    def load(path: str, format: str = 'keras') -> Any:
        """Load model from specified format."""
        path = Path(path)
        
        if format == 'keras':
            return ModelSerializer._load_keras(path)
        elif format == 'joblib':
            return ModelSerializer._load_joblib(path)
        elif format == 'pickle':
            return ModelSerializer._load_pickle(path)
        elif format == 'onnx':
            return ModelSerializer._load_onnx(path)
        elif format == 'torch':
            return ModelSerializer._load_torch(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _load_keras(path: Path) -> tf.keras.Model:
        """Load Keras model."""
        return load_keras_model(str(path))
    
    @staticmethod
    def _load_joblib(path: Path) -> Any:
        """Load joblib model."""
        return joblib.load(str(path))
    
    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """Load pickle model."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _load_onnx(path: Path) -> onnxruntime.InferenceSession:
        """Load ONNX model."""
        return onnxruntime.InferenceSession(str(path))
    
    @staticmethod
    def _load_torch(path: Path) -> torch.nn.Module:
        """Load PyTorch model."""
        return torch.load(str(path))
    
    def save_metadata(self, path: str, metadata: Dict[str, Any]) -> str:
        """Save model metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return str(path)
    
    @staticmethod
    def load_metadata(path: str) -> Dict[str, Any]:
        """Load model metadata."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_weights(self, path: str) -> str:
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, tf.keras.Model):
            self.model.save_weights(str(path))
        elif isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), str(path))
        else:
            raise ValueError("Weight saving only supported for Keras and PyTorch models")
        
        return str(path)
    
    @staticmethod
    def load_weights(path: str, model: Any) -> Any:
        """Load model weights."""
        path = Path(path)
        
        if isinstance(model, tf.keras.Model):
            model.load_weights(str(path))
        elif isinstance(model, torch.nn.Module):
            model.load_state_dict(torch.load(str(path)))
        else:
            raise ValueError("Weight loading only supported for Keras and PyTorch models")
        
        return model
    
    def save_config(self, path: str) -> str:
        """Save model configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, tf.keras.Model):
            config = self.model.get_config()
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
        else:
            raise ValueError("Config saving only supported for Keras models")
        
        return str(path)
    
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """Load model configuration."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_architecture(self, path: str) -> str:
        """Save model architecture."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, tf.keras.Model):
            with open(path, 'w') as f:
                f.write(self.model.to_json())
        else:
            raise ValueError("Architecture saving only supported for Keras models")
        
        return str(path)
    
    @staticmethod
    def load_architecture(path: str) -> tf.keras.Model:
        """Load model architecture."""
        with open(path, 'r') as f:
            return tf.keras.models.model_from_json(f.read())
    
    def save_training_history(self, path: str, history: Dict[str, Any]) -> str:
        """Save model training history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)
        
        return str(path)
    
    @staticmethod
    def load_training_history(path: str) -> Dict[str, Any]:
        """Load model training history."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_optimizer_state(self, path: str) -> str:
        """Save optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, tf.keras.Model):
            optimizer_config = self.model.optimizer.get_config()
            with open(path, 'w') as f:
                json.dump(optimizer_config, f, indent=4)
        else:
            raise ValueError("Optimizer state saving only supported for Keras models")
        
        return str(path)
    
    @staticmethod
    def load_optimizer_state(path: str, model: tf.keras.Model) -> tf.keras.Model:
        """Load optimizer state."""
        with open(path, 'r') as f:
            optimizer_config = json.load(f)
        
        model.optimizer = tf.keras.optimizers.deserialize(optimizer_config)
        return model
    
    def save_custom_objects(self, path: str, custom_objects: Dict[str, Any]) -> str:
        """Save custom objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(custom_objects, f, indent=4)
        
        return str(path)
    
    @staticmethod
    def load_custom_objects(path: str) -> Dict[str, Any]:
        """Load custom objects."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_all(self, base_path: str, formats: List[str] = None) -> Dict[str, str]:
        """Save model in all specified formats."""
        if formats is None:
            formats = ['keras', 'joblib', 'pickle', 'onnx']
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        for format in formats:
            try:
                path = base_path / f"{self.model_name}.{format}"
                saved_paths[format] = self.save(str(path), format)
            except Exception as e:
                logger.warning(f"Failed to save model in {format} format: {e}")
        
        return saved_paths 