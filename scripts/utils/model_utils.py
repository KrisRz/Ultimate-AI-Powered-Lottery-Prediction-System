"""Utility functions for model operations."""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import logging
import os
import gc
import psutil
import time
from pathlib import Path
import glob
import traceback
from functools import wraps
import functools

logger = logging.getLogger(__name__)

# Constants
LOOK_BACK = 200  # Number of previous draws to consider

def setup_gpu_memory_growth():
    """Configure GPU memory growth to avoid memory allocation issues."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
        else:
            logger.info("No GPU devices found")
    except Exception as e:
        logger.error(f"Error setting up GPU memory growth: {str(e)}")

def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

def create_sequences(data: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create input sequences and targets for time series prediction."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def normalize_data(data: np.ndarray, min_val: float = 1, max_val: float = 59) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Normalize data to [0, 1] range."""
    data_range = max_val - min_val
    normalized = (data - min_val) / data_range
    return normalized, (min_val, max_val)

def denormalize_data(normalized: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Denormalize data from [0, 1] range."""
    data_range = max_val - min_val
    denormalized = (normalized * data_range) + min_val
    return denormalized

def create_callbacks(
    patience: int = 20,
    min_delta: float = 1e-4,
    model_path: Optional[str] = None
) -> List[tf.keras.callbacks.Callback]:
    """Create training callbacks."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if model_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    return callbacks

def validate_predictions(predictions: np.ndarray, min_val: float = 1, max_val: float = 59) -> np.ndarray:
    """Validate and clean predictions."""
    # Ensure predictions are within valid range
    predictions = np.clip(predictions, min_val, max_val)
    
    # Round to nearest integer
    predictions = np.round(predictions).astype(int)
    
    # Ensure unique values
    for i in range(len(predictions)):
        pred = predictions[i]
        unique_pred = np.unique(pred)
        while len(unique_pred) < 6:
            new_num = np.random.randint(min_val, max_val + 1)
            if new_num not in unique_pred:
                unique_pred = np.append(unique_pred, new_num)
        predictions[i] = np.sort(unique_pred[:6])
    
    return predictions

def calculate_metrics(predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Calculate prediction metrics."""
    metrics = {}
    
    # Accuracy (exact matches)
    exact_matches = np.all(predictions == actual, axis=1)
    metrics['accuracy'] = float(np.mean(exact_matches))
    
    # Match rate (average number of matching numbers)
    match_counts = np.sum(predictions == actual, axis=1)
    metrics['match_rate'] = float(np.mean(match_counts))
    
    # RMSE
    metrics['rmse'] = float(np.sqrt(np.mean((predictions - actual) ** 2)))
    
    # MAE
    metrics['mae'] = float(np.mean(np.abs(predictions - actual)))
    
    return metrics

def log_training_errors(func: Callable) -> Callable:
    """
    Decorator to log errors during model training
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that logs errors
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    return wrapper

def log_prediction_errors(model_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {model_name} prediction: {str(e)}\n{traceback.format_exc()}")
                raise
        return wrapper
    return decorator

def ensure_valid_prediction(prediction: List[int], min_val: int = 1, max_val: int = 59) -> List[int]:
    """Ensure prediction contains 6 unique integers in valid range."""
    try:
        # Convert to list if numpy array
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
            
        # Validate type
        if not isinstance(prediction, list):
            raise ValueError(f"Prediction must be a list, got {type(prediction)}")
            
        # Ensure integers
        prediction = [int(x) for x in prediction]
        
        # Clip to valid range
        prediction = [np.clip(x, min_val, max_val) for x in prediction]
        
        # Ensure uniqueness
        unique_nums = []
        for num in prediction:
            if num not in unique_nums:
                unique_nums.append(num)
                
        # Add random numbers if needed
        while len(unique_nums) < 6:
            num = np.random.randint(min_val, max_val + 1)
            if num not in unique_nums:
                unique_nums.append(num)
                
        # Sort and return first 6
        return sorted(unique_nums[:6])
        
    except Exception as e:
        logger.error(f"Error ensuring valid prediction: {str(e)}")
        # Return random valid numbers as fallback
        return sorted(list(set([np.random.randint(min_val, max_val + 1) for _ in range(10)]))[:6])

def validate_prediction_format(prediction: Union[List[int], np.ndarray]) -> bool:
    """Validate prediction format."""
    try:
        # Convert numpy array to list
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
            
        # Check type
        if not isinstance(prediction, list):
            return False
            
        # Check length
        if len(prediction) != 6:
            return False
            
        # Check values
        if not all(isinstance(x, (int, np.integer)) for x in prediction):
            return False
            
        # Check range
        if not all(1 <= x <= 59 for x in prediction):
            return False
            
        # Check uniqueness
        if len(set(prediction)) != 6:
            return False
            
        return True
        
    except Exception:
        return False

def monitor_gpu(func):
    """Monitor GPU memory usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            logger.warning(f"Error configuring GPU memory growth: {str(e)}")
        return func(*args, **kwargs)
    return wrapper

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], keep_latest: int = 5) -> None:
    """Clean up old model checkpoints, keeping only the latest few."""
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
            
        # Get all checkpoint files
        checkpoints = list(checkpoint_dir.glob("*.h5"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[keep_latest:]:
            try:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
                
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {str(e)}") 