"""Utility functions for model training and prediction."""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
LOOK_BACK = 10

def log_training_errors(error: Exception, model_name: str) -> None:
    """Log training errors with context."""
    logger.error(f"Error training {model_name}: {str(error)}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error traceback: {error.__traceback__}")

def ensure_valid_prediction(prediction: np.ndarray) -> np.ndarray:
    """Ensure prediction is valid lottery numbers."""
    # Round to nearest integer
    prediction = np.round(prediction)
    
    # Clip to valid range (1-59)
    prediction = np.clip(prediction, 1, 59)
    
    # Remove duplicates
    prediction = np.unique(prediction)
    
    # Ensure we have 6 numbers
    if len(prediction) < 6:
        # Add random numbers if needed
        while len(prediction) < 6:
            new_num = np.random.randint(1, 60)
            if new_num not in prediction:
                prediction = np.append(prediction, new_num)
    elif len(prediction) > 6:
        # Take first 6 numbers if too many
        prediction = prediction[:6]
        
    return prediction

def validate_prediction_format(prediction: np.ndarray) -> bool:
    """Validate prediction format."""
    if not isinstance(prediction, np.ndarray):
        return False
    if prediction.shape != (6,):
        return False
    if not all(1 <= x <= 59 for x in prediction):
        return False
    if len(np.unique(prediction)) != 6:
        return False
    return True

def monitor_gpu(func):
    """Decorator to monitor GPU usage."""
    def wrapper(*args, **kwargs):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
            except RuntimeError as e:
                logger.warning(f"Memory growth configuration failed: {e}")
        return func(*args, **kwargs)
    return wrapper

def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3) -> None:
    """Clean up old model checkpoints."""
    try:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
            
        checkpoints = sorted(checkpoint_dir.glob("*.h5"), key=lambda x: x.stat().st_mtime)
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {str(e)}") 