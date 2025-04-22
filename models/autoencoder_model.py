import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any, Union
from .utils import ensure_valid_prediction, log_training_errors

def create_predictive_autoencoder(input_dim: int, output_dim: int = 6):
    """
    Create an autoencoder model that predicts the next 6 numbers
    
    Args:
        input_dim: Dimension of the input features
        output_dim: Dimension of the output (default 6 for lottery numbers)
        
    Returns:
        Compiled Keras model
    """
    # Encoder
    input_layer = Input(shape=(input_dim,))
    
    # Deeper encoder with batch normalization
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    
    encoded = Dense(32, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    
    latent = Dense(16, activation='relu')(encoded)
    
    # Decoder for next numbers prediction
    decoded = Dense(32, activation='relu')(latent)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    
    # Output layer for next 6 numbers
    output_layer = Dense(output_dim, activation='sigmoid')(decoded)
    
    # Create model
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def create_sequential_training_data(data: np.ndarray, look_back: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data for sequence prediction (X = past draws, y = next draw)
    
    Args:
        data: Array of lottery draws (each row is a draw)
        look_back: Number of previous draws to use for predicting next draw
        
    Returns:
        Tuple of (X_seq, y_seq) for training
    """
    X_seq, y_seq = [], []
    
    # Ensure we have enough data
    if len(data) <= look_back:
        raise ValueError(f"Not enough data for look_back={look_back}. Need more than {look_back} draws.")
    
    # Create sequences
    for i in range(len(data) - look_back):
        # X is a flattened sequence of past draws
        X_seq.append(data[i:i+look_back].flatten())
        # y is the next draw (6 numbers)
        y_seq.append(data[i+look_back])
        
    return np.array(X_seq), np.array(y_seq)

@log_training_errors
def train_autoencoder_model(X_train, y_train=None, look_back=5, epochs=100, batch_size=32):
    """
    Train an autoencoder model to predict the next 6 numbers
    
    Args:
        X_train: Training data (historical lottery draws)
        y_train: Target data (if None, will use X_train to create sequences)
        look_back: Number of previous draws to use for prediction
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Tuple of (model, scaler, look_back)
    """
    try:
        # Validate input
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        
        # Check if X_train has the right shape
        if len(X_train.shape) < 2:
            raise ValueError(f"X_train must be at least 2D, got shape {X_train.shape}")
        
        # If y_train is None, create sequential training data
        if y_train is None:
            if X_train.shape[1] == 6:  # Each row is a draw with 6 numbers
                logging.info(f"Creating sequential training data with look_back={look_back}")
                X_seq, y_seq = create_sequential_training_data(X_train, look_back)
            else:
                raise ValueError(f"When y_train is None, X_train must have 6 columns (got {X_train.shape[1]})")
        else:
            # User provided explicit X_train and y_train
            if not isinstance(y_train, np.ndarray):
                y_train = np.array(y_train)
            
            # Check shapes
            if y_train.shape[1] != 6:
                raise ValueError(f"y_train must have 6 columns, got {y_train.shape[1]}")
            
            X_seq, y_seq = X_train, y_train
            
        # Scale the data
        X_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X_seq)
        
        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y_seq)
        
        # Create and train the model
        input_dim = X_scaled.shape[1]
        model = create_predictive_autoencoder(input_dim=input_dim, output_dim=6)
        
        # Use early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train with validation split
        model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Log model summary
        logging.info(f"Autoencoder model trained on {len(X_scaled)} sequences")
        logging.info(f"Input shape: {input_dim}, Output shape: 6")
        
        return model, (X_scaler, y_scaler), look_back
        
    except Exception as e:
        logging.error(f"Error in autoencoder training: {str(e)}")
        raise

def predict_autoencoder_model(model, scalers, X, look_back=5):
    """
    Generate predictions using trained autoencoder model
    
    Args:
        model: Trained Keras model
        scalers: Tuple of (X_scaler, y_scaler)
        X: Input features or raw draw history
        look_back: Number of previous draws used for prediction
        
    Returns:
        Array of 6 valid lottery numbers
    """
    try:
        X_scaler, y_scaler = scalers
        
        # Check if X is raw draw history or already prepared features
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # Handle different input shapes
        if len(X.shape) == 2 and X.shape[1] == 6:  # Raw draw history
            # Need at least look_back draws
            if len(X) < look_back:
                raise ValueError(f"Need at least {look_back} draws in history, got {len(X)}")
                
            # Get the most recent draws
            recent_draws = X[-look_back:].flatten().reshape(1, -1)
            X_prepared = recent_draws
        else:
            # Assume X is already prepared in the right format
            X_prepared = X
            
        # Scale input
        X_scaled = X_scaler.transform(X_prepared)
        
        # Generate prediction
        y_scaled_pred = model.predict(X_scaled)
        
        # Inverse transform to get actual numbers
        y_pred = y_scaler.inverse_transform(y_scaled_pred)
        
        # Ensure we have valid lottery numbers
        valid_prediction = ensure_valid_prediction(y_pred[0])
        
        return valid_prediction
        
    except Exception as e:
        logging.error(f"Error in autoencoder prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 