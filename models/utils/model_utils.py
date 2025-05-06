"""General model utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import joblib
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import functools
import psutil
import gc
import traceback

logger = logging.getLogger(__name__)

# Add LOOK_BACK constant
LOOK_BACK = 200  # Number of previous draws to consider

def monitor_gpu(func: Callable) -> Callable:
    """Decorator to monitor GPU memory usage during function execution.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with GPU monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Log initial GPU memory
            initial_mem = get_gpu_memory_info()
            logger.info(f"Initial GPU memory: {initial_mem}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log final GPU memory
            final_mem = get_gpu_memory_info()
            logger.info(f"Final GPU memory: {final_mem}")
            
            # Clear memory if needed
            if check_gpu_availability():
                clear_gpu_memory()
                
            return result
            
        except Exception as e:
            logger.error(f"Error in GPU-monitored function: {str(e)}")
            raise
            
    return wrapper

def create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series prediction.
    
    Args:
        data: Input data array
        lookback: Number of time steps to look back
        
    Returns:
        Tuple of (X, y) where X contains sequences and y contains targets
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape: Tuple[int, ...], params: Dict[str, Any]) -> tf.keras.Model:
    """Create an LSTM model with specified parameters."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units=params['lstm_units_1'],
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
        ),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.LSTM(
            units=params['lstm_units_2'],
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
        ),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(
            units=params['dense_units'],
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg'])
        ),
        tf.keras.layers.Dense(1)
    ])
    
    return model

def create_cnn_lstm_model(input_shape: Tuple[int, ...], params: Dict[str, Any]) -> tf.keras.Model:
    """Create a CNN-LSTM model with specified parameters."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(
            filters=params['filters'],
            kernel_size=params['kernel_size'],
            activation='relu',
            input_shape=input_shape
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(
            units=params['lstm_units'],
            return_sequences=True
        ),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.LSTM(units=params['lstm_units']),
        tf.keras.layers.Dropout(params['dropout_rate']),
        tf.keras.layers.Dense(1)
    ])
    
    return model

def create_ensemble_model(models: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Any:
    """Create an ensemble model from multiple base models."""
    class EnsembleModel:
        def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
            self.models = models
            self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
            
        def predict(self, X: np.ndarray) -> np.ndarray:
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred * self.weights[name])
            return np.sum(predictions, axis=0)
    
    return EnsembleModel(models, weights)

def save_model(model: Any, path: str, name: str) -> None:
    """Save a model to disk."""
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(path, f"{name}_{timestamp}")
    
    if isinstance(model, tf.keras.Model):
        model.save(model_path)
    else:
        joblib.dump(model, f"{model_path}.joblib")
    
    logger.info(f"Model saved to {model_path}")

def load_model(path: str) -> Any:
    """Load a model from disk."""
    if path.endswith('.joblib'):
        return joblib.load(path)
    else:
        return tf.keras.models.load_model(path)

def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X)
    
    metrics = {
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred)
    }
    
    return metrics

def prepare_time_series_data(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare time series data for model training."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def scale_data(data: np.ndarray, scaler_type: str = 'standard') -> Tuple[np.ndarray, Any]:
    """Scale data using specified scaler."""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def inverse_scale_data(data: np.ndarray, scaler: Any) -> np.ndarray:
    """Inverse transform scaled data."""
    return scaler.inverse_transform(data)

def get_model_summary(model: Any) -> str:
    """Get model summary as string."""
    if isinstance(model, tf.keras.Model):
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)
    else:
        return str(model)

def get_model_parameters(model: Any) -> Dict[str, Any]:
    """Get model parameters."""
    if isinstance(model, tf.keras.Model):
        return model.get_config()
    else:
        return model.get_params()

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_gpu_availability() -> bool:
    """Check if GPU is available."""
    return len(tf.config.list_physical_devices('GPU')) > 0

def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return {'available': False}
        
        memory_info = {}
        for gpu in gpus:
            try:
                # Try to get memory info, but don't fail if not available
                memory_info[gpu.name] = {
                    'available': True,
                    'device_type': 'GPU',
                    'name': gpu.name
                }
            except Exception as e:
                logger.warning(f"Could not get memory info for {gpu.name}: {str(e)}")
                memory_info[gpu.name] = {
                    'available': True,
                    'device_type': 'GPU',
                    'name': gpu.name,
                    'error': str(e)
                }
        
        return memory_info
        
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {str(e)}")
        return {'available': False, 'error': str(e)}

def clear_gpu_memory() -> None:
    """Clear GPU memory."""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

def setup_gpu_memory_growth() -> None:
    """Configure GPU memory growth to prevent OOM errors."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        else:
            logger.info("No GPU devices found")
    except Exception as e:
        logger.error(f"Error setting up GPU memory growth: {str(e)}")

def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Any]:
    """Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax' or 'standard')
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    # Reshape for 1D array if needed
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def denormalize_data(data: np.ndarray, scaler: Any) -> np.ndarray:
    """Denormalize data using fitted scaler.
    
    Args:
        data: Normalized data array
        scaler: Fitted scaler object
        
    Returns:
        Denormalized data array
    """
    # Reshape for 1D array if needed
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        
    return scaler.inverse_transform(data)

def create_callbacks(
    checkpoint_dir: str,
    patience: int = 10,
    min_delta: float = 1e-4,
    save_best_only: bool = True,
    monitor: str = 'val_loss'
) -> List[tf.keras.callbacks.Callback]:
    """Create callbacks for model training.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in monitored quantity to qualify as an improvement
        save_best_only: Only save model if monitored quantity improves
        monitor: Quantity to monitor for improvement
        
    Returns:
        List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            verbose=1,
            mode='min',
            restore_best_weights=True
        ),
        
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            monitor=monitor,
            save_best_only=save_best_only,
            mode='min',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(checkpoint_dir, 'logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def validate_predictions(predictions: np.ndarray, min_value: float = 1, max_value: float = 60) -> bool:
    """Validate model predictions.
    
    Args:
        predictions: Array of predictions
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if predictions are valid, raises ValueError otherwise
    """
    try:
        # Check for NaN values
        if np.isnan(predictions).any():
            raise ValueError("Predictions contain NaN values")
            
        # Check for infinity values
        if np.isinf(predictions).any():
            raise ValueError("Predictions contain infinity values")
            
        # Check value range
        if np.any(predictions < min_value) or np.any(predictions > max_value):
            raise ValueError(f"Predictions must be between {min_value} and {max_value}")
            
        # Check for duplicates in each prediction
        if len(predictions.shape) > 1:
            for pred in predictions:
                if len(np.unique(pred)) != len(pred):
                    raise ValueError("Predictions contain duplicate numbers")
                    
        # Check if predictions are sorted
        if len(predictions.shape) > 1:
            for pred in predictions:
                if not np.all(np.diff(pred) > 0):
                    logger.warning("Predictions are not sorted in ascending order")
                    
        return True
        
    except Exception as e:
        logger.error(f"Prediction validation failed: {str(e)}")
        raise

def ensure_valid_prediction(prediction: np.ndarray, min_value: float = 1, max_value: float = 60) -> np.ndarray:
    """Ensure prediction is valid by fixing common issues.
    
    Args:
        prediction: Array of predictions
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Valid prediction array
    """
    try:
        # Convert to numpy array if needed
        prediction = np.array(prediction)
        
        # Replace NaN values with mean
        if np.isnan(prediction).any():
            mean_val = np.nanmean(prediction)
            prediction = np.nan_to_num(prediction, nan=mean_val)
            
        # Clip values to valid range
        prediction = np.clip(prediction, min_value, max_value)
        
        # Round to nearest integer
        prediction = np.round(prediction).astype(int)
        
        # Remove duplicates and sort
        if len(prediction.shape) > 1:
            for i in range(len(prediction)):
                unique_vals = np.unique(prediction[i])
                if len(unique_vals) < len(prediction[i]):
                    # If duplicates exist, replace with next available number
                    used_nums = set(prediction[i])
                    available_nums = set(range(int(min_value), int(max_value) + 1)) - used_nums
                    for j in range(len(prediction[i])):
                        if prediction[i][j] in used_nums:
                            if available_nums:
                                prediction[i][j] = min(available_nums)
                                available_nums.remove(prediction[i][j])
                prediction[i].sort()
                
        return prediction
        
    except Exception as e:
        logger.error(f"Error ensuring valid prediction: {str(e)}")
        raise

def validate_prediction_format(prediction: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
    """Validate prediction format.
    
    Args:
        prediction: Prediction array to validate
        expected_shape: Expected shape of prediction
        
    Returns:
        True if format is valid, raises ValueError otherwise
    """
    try:
        # Check if prediction is numpy array
        if not isinstance(prediction, np.ndarray):
            raise ValueError("Prediction must be a numpy array")
            
        # Check shape
        if prediction.shape != expected_shape:
            raise ValueError(f"Prediction shape {prediction.shape} does not match expected shape {expected_shape}")
            
        # Check data type
        if not np.issubdtype(prediction.dtype, np.number):
            raise ValueError("Prediction must contain numeric values")
            
        return True
        
    except Exception as e:
        logger.error(f"Prediction format validation failed: {str(e)}")
        raise

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
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
    return wrapper

def cleanup_old_checkpoints(checkpoint_dir: str, keep_latest: int = 5) -> None:
    """Clean up old model checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        keep_latest: Number of latest checkpoints to keep
    """
    try:
        # Ensure directory exists
        if not os.path.isdir(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
            return
            
        # Get all checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.h5') or file.endswith('.hdf5') or file.endswith('.keras'):
                file_path = os.path.join(checkpoint_dir, file)
                # Get modification time
                mod_time = os.path.getmtime(file_path)
                checkpoint_files.append((file_path, mod_time))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the latest files
        if len(checkpoint_files) > keep_latest:
            files_to_delete = checkpoint_files[keep_latest:]
            for file_path, _ in files_to_delete:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old checkpoint: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {str(e)}")
                    
            logger.info(f"Cleaned up {len(files_to_delete)} old checkpoints, kept {keep_latest} latest files")
        else:
            logger.info(f"No cleanup needed, only {len(checkpoint_files)} checkpoints found")
    
    except Exception as e:
        logger.error(f"Error cleaning up checkpoints: {str(e)}")
        logger.debug(traceback.format_exc()) 