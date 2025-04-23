import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import logging
import gc
import time
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional
from .utils import log_training_errors, ensure_valid_prediction, LOOK_BACK
from .utils.model_storage import cleanup_old_checkpoints

# Enable mixed precision for faster training on compatible GPUs
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info(f"Enabled mixed precision training with {len(physical_devices)} GPU(s)")
except Exception as e:
    logging.warning(f"Could not enable mixed precision: {str(e)}")

def create_lstm_model(
    input_shape: Tuple[int, int],
    units1: int = 256, 
    units2: int = 128, 
    dropout_rate: float = 0.3, 
    l2_reg: float = 0.001, 
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Create an optimized LSTM model for lottery number prediction
    
    Args:
        input_shape: Input shape (timesteps, features)
        units1: Number of units in first LSTM layer
        units2: Number of units in second LSTM layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras LSTM model
    """
    model = Sequential([
        # Use Bidirectional LSTM for better pattern recognition
        Bidirectional(LSTM(units1, return_sequences=True, 
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg/2),
                          input_shape=input_shape)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Bidirectional(LSTM(units2, return_sequences=False,
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg/2))),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        
        # Add activation to constrain output values
        Dense(6, activation='sigmoid')  # Sigmoid to constrain output
    ])
    
    # Use Adam optimizer with customized learning rate and gradient clipping
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0, clipvalue=0.5)
    
    # Use Huber loss instead of MSE to be more robust to outliers
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
    return model

def vectorized_create_sequences(X_scaled: np.ndarray, look_back: int) -> np.ndarray:
    """
    Create sequences for LSTM in a vectorized way to improve performance
    
    Args:
        X_scaled: Scaled input data
        look_back: Number of timesteps to use
        
    Returns:
        Array of sequences
    """
    n_samples = len(X_scaled) - look_back
    n_features = X_scaled.shape[1]
    
    # Pre-allocate output array for better performance
    X_reshaped = np.zeros((n_samples, look_back, n_features))
    
    # Use numpy's advanced indexing for faster sequence creation
    for i in range(look_back):
        X_reshaped[:, i, :] = X_scaled[i:i+n_samples]
    
    return X_reshaped

def preprocess_data(
    X: np.ndarray, 
    y: np.ndarray, 
    look_back: int
) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
    """
    Preprocess data for LSTM training with careful handling of potential issues
    
    Args:
        X: Input features
        y: Target values
        look_back: Number of timesteps to use
        
    Returns:
        Tuple of (processed_X, processed_y, scaler)
    """
    # Handle NaN values first
    if np.isnan(X).any():
        logging.warning(f"Found {np.isnan(X).sum()} NaN values in input data. Replacing with zeros.")
        X = np.nan_to_num(X, nan=0.0)
    
    if np.isnan(y).any():
        logging.warning(f"Found {np.isnan(y).sum()} NaN values in target data. Replacing with zeros.")
        y = np.nan_to_num(y, nan=0.0)
    
    # Check for and handle infinity values
    if np.isinf(X).any():
        logging.warning(f"Found {np.isinf(X).sum()} infinity values in input data. Replacing with large finite values.")
        X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)
    
    if np.isinf(y).any():
        logging.warning(f"Found {np.isinf(y).sum()} infinity values in target data. Replacing with large finite values.")
        y = np.nan_to_num(y, posinf=1e6, neginf=-1e6)
    
    # Additional clipping of extreme values in target data
    y = np.clip(y, 1, 59)
    
    # Normalize target data to 0-1 range for better training stability
    y_normalized = y / 60.0  # Lottery numbers typically 1-59
    
    # Use RobustScaler for features to handle outliers better
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Further clip any extreme values that might remain after scaling
    X_scaled = np.clip(X_scaled, -10, 10)
    
    return X_scaled, y_normalized, scaler

class LSTMModel:
    def __init__(self, model, scaler, look_back):
        self.model = model
        self.scaler = scaler
        self.look_back = look_back
        
    def predict(self, X):
        # Scale input data
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        # Make prediction
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions

@log_training_errors
def train_lstm_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: int = 20, 
    batch_size: int = 64, 
    look_back: Optional[int] = None, 
    validation_split: float = 0.2
) -> LSTMModel:
    """
    Train LSTM model for lottery prediction
    """
    start_time = time.time()
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Set default look_back if not specified
    if look_back is None:
        look_back = LOOK_BACK
    
    # Initialize scaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    
    # Create model with optimal hyperparameters
    model = create_lstm_model(
        input_shape=(look_back, X_train.shape[2]),  # type: ignore
        units1=256,
        units2=128,
        dropout_rate=0.3,
        l2_reg=0.001,
        learning_rate=0.001
    )
    
    # Advanced callbacks for training optimization
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_dir / 'lstm_model_checkpoint_{epoch:02d}.h5'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        TensorBoard(log_dir='logs/lstm_model_' + time.strftime('%Y%m%d-%H%M%S'))
    ]
    
    try:
        # Train with validation split and callbacks
        history = model.fit(
            X_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True
        )
        
        # Log training results
        val_loss = min(history.history['val_loss'])
        final_loss = history.history['loss'][-1]
        training_time = time.time() - start_time
        
        logging.info(f"LSTM model trained successfully in {training_time:.2f} seconds")
        logging.info(f"Final loss: {final_loss:.4f}, Best validation loss: {val_loss:.4f}")
        
        # Cleanup old checkpoints after successful training
        cleanup_old_checkpoints(checkpoint_dir, pattern="lstm_model_checkpoint_*.h5")
        
    except Exception as e:
        logging.error(f"Error training LSTM model: {str(e)}")
        raise
    
    # Clean up memory
    gc.collect()
    if len(tf.config.list_physical_devices('GPU')) > 0:
        try:
            tf.keras.backend.clear_session()
        except Exception as e:
            logging.warning(f"Could not clear GPU memory: {e}")
    
    return LSTMModel(model, scaler, look_back)

def predict_lstm_model(
    model: tf.keras.Model, 
    scaler: RobustScaler, 
    X: np.ndarray, 
    look_back: Optional[int] = None
) -> List[int]:
    """
    Generate predictions using trained LSTM model
    
    Args:
        model: Trained LSTM model
        scaler: Fitted scaler
        X: Input features for prediction
        look_back: Number of timesteps used in training (default: uses utils.LOOK_BACK)
        
    Returns:
        Array of 6 valid lottery numbers
    """
    try:
        # Set default look_back if not specified
        if look_back is None:
            look_back = LOOK_BACK
        
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Handle already 3D input
        if len(X.shape) == 3:
            # Reshape for scaling (combine batch and time dimensions)
            orig_shape = X.shape
            reshaped_X = X.reshape(-1, X.shape[2])
            
            # Clean input data
            reshaped_X = np.nan_to_num(reshaped_X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale the data
            scaled_X = scaler.transform(reshaped_X)
            
            # Clip values to avoid extremes
            scaled_X = np.clip(scaled_X, -10, 10)
            
            # Reshape back to original shape
            X_sequence = scaled_X.reshape(orig_shape)
        else:
            # Clean input data
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale input data
            X_scaled = scaler.transform(X)
            
            # Clip scaled values
            X_scaled = np.clip(X_scaled, -10, 10)
            
            # Use efficient numpy operations for sequence creation
            if len(X_scaled) >= look_back:
                # Take the most recent look_back entries for prediction
                X_sequence = X_scaled[-look_back:].reshape(1, look_back, X_scaled.shape[1])
            else:
                # Pad with zeros if we don't have enough data
                padding = np.zeros((look_back - len(X_scaled), X_scaled.shape[1]))
                X_padded = np.vstack([padding, X_scaled])
                X_sequence = X_padded.reshape(1, look_back, X_scaled.shape[1])
        
        # Generate prediction
        predictions = model.predict(X_sequence, verbose=0)
        
        # If predictions were normalized to 0-1 range, denormalize
        if np.max(predictions) <= 1.0:
            predictions = predictions * 60.0
        
        # Ensure valid lottery numbers
        valid_prediction = ensure_valid_prediction(predictions[0])
        
        return valid_prediction
        
    except Exception as e:
        logging.error(f"Error in LSTM prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6))

def load_pretrained_lstm_model(model_path: str = "models/checkpoints/lstm_model_checkpoint.h5") -> Optional[tf.keras.Model]:
    """
    Load a pretrained LSTM model from disk
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Loaded pretrained LSTM model from {model_path}")
            return model
        else:
            logging.warning(f"Pretrained model not found at {model_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading pretrained model: {str(e)}")
        return None 