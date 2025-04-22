import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
import logging
import gc
import time
import os
from pathlib import Path
from .utils import LOOK_BACK, ensure_valid_prediction
from .lstm_model import vectorized_create_sequences

# Try to enable GPU acceleration and mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception as e:
    logging.warning(f"Could not configure GPU acceleration: {str(e)}")

def create_cnn_lstm_model(input_shape, filters=128, kernel_size=3, lstm_units1=128, lstm_units2=64, 
                         dropout_rate=0.3, l2_reg=0.001, learning_rate=0.001):
    """
    Create an optimized CNN-LSTM model for time series prediction
    
    Args:
        input_shape: Tuple specifying input shape (timesteps, features)
        filters: Number of filters in the Conv1D layer
        kernel_size: Size of the CNN kernel
        lstm_units1: Number of units in the first LSTM layer
        lstm_units2: Number of units in the second LSTM layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled CNN-LSTM model
    """
    model = Sequential([
        # CNN layers with regularization
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
              padding='same', input_shape=input_shape,
              kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate/2),
        
        # Additional CNN layer for deeper feature extraction
        Conv1D(filters=filters*2, kernel_size=kernel_size, activation='relu', 
              padding='same', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate/2),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(lstm_units1, return_sequences=True,
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg/2))),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Bidirectional(LSTM(lstm_units2, return_sequences=False,
                          kernel_regularizer=l2(l2_reg),
                          recurrent_regularizer=l2(l2_reg/2))),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Dense layers for prediction
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate/2),
        
        Dense(6)  # 6 numbers to predict
    ])
    
    # Use Adam optimizer with customized learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def train_cnn_lstm_model(X_train, y_train, epochs=200, batch_size=64, look_back=None, validation_split=0.2):
    """
    Train an optimized CNN-LSTM model on lottery data
    
    Args:
        X_train: Training data
        y_train: Target values
        epochs: Number of training epochs
        batch_size: Training batch size
        look_back: Number of timesteps (default: use LOOK_BACK from utils)
        validation_split: Portion of data to use for validation
        
    Returns:
        Tuple of (trained model, scaler, look_back)
    """
    start_time = time.time()
    
    # Use provided look_back or default
    if look_back is None:
        look_back = LOOK_BACK
    
    # Validate input shape
    if not isinstance(X_train, np.ndarray):
        try:
            X_train = np.array(X_train)
        except Exception as e:
            raise ValueError(f"X_train must be a numpy array or list, got {type(X_train)}: {str(e)}")
    
    # Create and fit robust scaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Log input data shape
    logging.info(f"Training CNN-LSTM with input shape: {X_train.shape}, look_back: {look_back}")
    
    # Create sequences using vectorized approach
    X_reshaped = vectorized_create_sequences(X_scaled, look_back)
    
    # Align y_train with the reshaped X
    if y_train.shape[0] != len(X_reshaped):
        y_train = y_train[look_back:]
        if len(y_train) > len(X_reshaped):
            y_train = y_train[:len(X_reshaped)]
    
    # Log processed data shapes
    logging.info(f"Processed data shapes - X: {X_reshaped.shape}, y: {y_train.shape}")
    
    # Create model directory for checkpoints
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Create and train model with optimized hyperparameters
    model = create_cnn_lstm_model(
        input_shape=(look_back, X_train.shape[1]),
        filters=128,
        kernel_size=3,
        lstm_units1=128,
        lstm_units2=64,
        dropout_rate=0.3,
        l2_reg=0.001,
        learning_rate=0.001
    )
    
    # Advanced callbacks for training optimization
    callbacks = [
        # Early stopping with appropriate patience
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
            filepath=str(checkpoint_dir / 'cnn_lstm_model_checkpoint.h5'),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(log_dir='logs/cnn_lstm_model_' + time.strftime('%Y%m%d-%H%M%S'))
    ]
    
    try:
        # Train with validation split and callbacks
        history = model.fit(
            X_reshaped, y_train,
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
        
        logging.info(f"CNN-LSTM model trained successfully in {training_time:.2f} seconds")
        logging.info(f"Final loss: {final_loss:.4f}, Best validation loss: {val_loss:.4f}")
        
    except Exception as e:
        logging.error(f"Error training CNN-LSTM model: {str(e)}")
        raise
    
    # Clean up memory
    gc.collect()
    if len(tf.config.list_physical_devices('GPU')) > 0:
        try:
            # Clear GPU memory
            tf.keras.backend.clear_session()
        except Exception as e:
            logging.warning(f"Could not clear GPU memory: {str(e)}")
    
    return model, scaler, look_back

def predict_cnn_lstm_model(model, scaler, X, look_back=None):
    """
    Generate predictions using trained CNN-LSTM model
    
    Args:
        model: Trained CNN-LSTM model
        scaler: Fitted scaler
        X: Input data for prediction
        look_back: Number of timesteps (default: use LOOK_BACK from utils)
        
    Returns:
        Array of 6 valid lottery numbers (sorted, unique, in range 1-59)
    """
    try:
        # Use provided look_back or default
        if look_back is None:
            look_back = LOOK_BACK
            
        # Validate input
        if X is None:
            raise ValueError("Input data cannot be None")
            
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
        
        # Generate prediction with reduced verbosity
        raw_predictions = model.predict(X_sequence, verbose=0)
        
        # Ensure valid prediction format (6 unique numbers between 1-59)
        valid_predictions = ensure_valid_prediction(raw_predictions[0])
        
        return valid_predictions
        
    except Exception as e:
        logging.error(f"Error in CNN-LSTM prediction: {str(e)}")
        # Generate random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6))

def load_pretrained_cnn_lstm_model(model_path="models/checkpoints/cnn_lstm_model_checkpoint.h5"):
    """
    Load a pretrained CNN-LSTM model from disk
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Loaded pretrained CNN-LSTM model from {model_path}")
            return model
        else:
            logging.warning(f"Pretrained CNN-LSTM model not found at {model_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading pretrained CNN-LSTM model: {str(e)}")
        return None 