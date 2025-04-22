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
from typing import Tuple, List, Dict, Any, Union
from .utils import log_training_errors, ensure_valid_prediction, LOOK_BACK

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

def create_lstm_model(input_shape, units1=256, units2=128, dropout_rate=0.3, l2_reg=0.001, learning_rate=0.001):
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
        
        Dense(6)  # 6 numbers to predict
    ])
    
    # Use Adam optimizer with customized learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

def vectorized_create_sequences(X_scaled, look_back):
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

@log_training_errors
def train_lstm_model(X_train, y_train, epochs=200, batch_size=64, look_back=None, validation_split=0.2):
    """
    Train an optimized LSTM model for lottery number prediction
    
    Args:
        X_train: Training features
        y_train: Target values (lottery numbers)
        epochs: Number of training epochs
        batch_size: Batch size for training
        look_back: Number of timesteps to use (default: uses utils.LOOK_BACK)
        validation_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (trained model, scaler, look_back)
    """
    start_time = time.time()
    
    # Set default look_back if not specified
    if look_back is None:
        look_back = LOOK_BACK
    
    # Validate input
    if not isinstance(X_train, np.ndarray):
        try:
            X_train = np.array(X_train)
        except Exception as e:
            raise ValueError(f"Cannot convert X_train to numpy array: {str(e)}")
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Create sequences using vectorized approach
    logging.info(f"Creating sequences from {len(X_scaled)} samples with look_back={look_back}")
    X_reshaped = vectorized_create_sequences(X_scaled, look_back)
    
    # If we need to align y_train with the reshaped X
    if y_train.shape[0] != len(X_reshaped):
        y_train = y_train[look_back:]
        if len(y_train) > len(X_reshaped):
            y_train = y_train[:len(X_reshaped)]
    
    # Log the shapes to verify correct preprocessing
    logging.info(f"Training data shapes - X: {X_reshaped.shape}, y: {y_train.shape}")
    
    # Create model directory for checkpoints
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Create and train the model with optimal hyperparameters for large datasets
    model = create_lstm_model(
        input_shape=(look_back, X_train.shape[1]),
        units1=256,
        units2=128,
        dropout_rate=0.3,
        l2_reg=0.001,
        learning_rate=0.001
    )
    
    # Advanced callbacks for training optimization
    callbacks = [
        # Early stopping with longer patience for large datasets
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau is reached
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        # Save model checkpoints
        ModelCheckpoint(
            filepath=str(checkpoint_dir / 'lstm_model_checkpoint.h5'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # Optional TensorBoard logging
        TensorBoard(log_dir='logs/lstm_model_' + time.strftime('%Y%m%d-%H%M%S'))
    ]
    
    # Train with validation split and advanced callbacks
    history = model.fit(
        X_reshaped, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True  # Ensure data is shuffled for better training
    )
    
    # Log training results
    val_loss = min(history.history['val_loss'])
    final_loss = history.history['loss'][-1]
    training_time = time.time() - start_time
    
    logging.info(f"LSTM model trained on {len(X_reshaped)} sequences in {training_time:.2f} seconds")
    logging.info(f"Final training loss: {final_loss:.4f}, Best validation loss: {val_loss:.4f}")
    logging.info(f"Input shape: ({look_back}, {X_train.shape[1]}), Output shape: 6")
    
    # Clean up memory
    gc.collect()
    if len(tf.config.list_physical_devices('GPU')) > 0:
        try:
            # Clear GPU memory
            tf.keras.backend.clear_session()
        except Exception as e:
            logging.warning(f"Could not clear GPU memory: {str(e)}")
    
    return model, scaler, look_back

def predict_lstm_model(model, scaler, X, look_back=None):
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
        
        # Scale input data
        X_scaled = scaler.transform(X)
        
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
        
        # Ensure valid lottery numbers
        valid_prediction = ensure_valid_prediction(predictions[0])
        
        return valid_prediction
        
    except Exception as e:
        logging.error(f"Error in LSTM prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6))

def load_pretrained_lstm_model(model_path="models/checkpoints/lstm_model_checkpoint.h5"):
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