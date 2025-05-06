"""LSTM model for lottery prediction."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU, Bidirectional, InputLayer, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import logging
import gc
import time
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional
import pickle
import random
import traceback
from models.utils.model_utils import (
    log_training_errors,
    ensure_valid_prediction,
    validate_prediction_format,
    monitor_gpu,
    cleanup_old_checkpoints,
    LOOK_BACK
)

logger = logging.getLogger(__name__)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except RuntimeError as e:
        logger.warning(f"Memory growth configuration failed: {e}")

# Set fixed random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Force all tensors to use float32
tf.keras.backend.set_floatx('float32')

# Set global mixed precision policy to float32
policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)

def create_lstm_model(input_shape: Tuple[int, int], config: Dict[str, Any]) -> tf.keras.Model:
    """Create LSTM model with given input shape and config"""
    n_outputs = 6  # Fixed for lottery prediction
    
    # Create input layer with explicit shape
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    
    # LSTM layers with proper gradient handling
    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config['lstm_units'][0],
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0.01)),
            dtype=tf.float32,
            go_backwards=False
        ),
        merge_mode='concat',
        backward_layer=tf.keras.layers.LSTM(
            config['lstm_units'][0],
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0.01)),
            dtype=tf.float32,
            go_backwards=True
        )
    )(inputs)
    lstm1 = tf.keras.layers.BatchNormalization(dtype=tf.float32)(lstm1)
    lstm1 = tf.keras.layers.Dropout(config['dropout'])(lstm1)
    
    lstm2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config['lstm_units'][1],
            kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0.01)),
            dtype=tf.float32,
            go_backwards=False
        ),
        merge_mode='concat',
        backward_layer=tf.keras.layers.LSTM(
            config['lstm_units'][1],
            kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0.01)),
            dtype=tf.float32,
            go_backwards=True
        )
    )(lstm1)
    lstm2 = tf.keras.layers.BatchNormalization(dtype=tf.float32)(lstm2)
    lstm2 = tf.keras.layers.Dropout(config['dropout'])(lstm2)
    
    # Dense layers
    dense1 = tf.keras.layers.Dense(config['dense_units'][0], activation='relu', dtype=tf.float32)(lstm2)
    dense1 = tf.keras.layers.BatchNormalization(dtype=tf.float32)(dense1)
    dense1 = tf.keras.layers.Dropout(config['dropout'])(dense1)
    
    # Output layer
    outputs = tf.keras.layers.Dense(n_outputs, activation='sigmoid', dtype=tf.float32)(dense1)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with gradient handling
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        clipnorm=1.0,
        clipvalue=0.5
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model

def vectorized_create_sequences(data, sequence_length):
    """
    Create sequences from data using vectorized operations
    """
    try:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        if data.size == 0:
            raise ValueError("Empty input array")
            
        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")
            
        if len(data) < sequence_length:
            raise ValueError(f"Data length {len(data)} is shorter than sequence length {sequence_length}")
            
        # Create sequences
        n_sequences = len(data) - sequence_length
        X = np.zeros((n_sequences, sequence_length, data.shape[1]))
        y = np.zeros((n_sequences, data.shape[1]))
        
        for i in range(n_sequences):
            X[i] = data[i:i+sequence_length]
            y[i] = data[i+sequence_length]
            
        return X, y
        
    except Exception as e:
        logging.error(f"Sequence creation failed: {str(e)}")
        raise

def preprocess_data(X: np.ndarray, y: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Preprocess input data for LSTM model"""
    # Handle NaN and Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create and fit scaler for X
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Normalize y values between 0 and 1
    y_normalized = y / np.max(y, axis=0)
    
    return X_scaled, y_normalized, scaler

class LSTMModel:
    def __init__(self, input_dim=None, name='lstm', **kwargs):
        self.name = name
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Default configuration
        self.config = {
            'epochs': 100,
            'batch_size': 32,
            'lstm_units': [64, 32],
            'dense_units': [32],
            'dropout': 0.4,
            'learning_rate': 0.0005,
            'l2_reg': 0.01,
            'validation_split': 0.2,
            'early_stopping': {
                'patience': 20,
                'restore_best_weights': True
            },
            'reduce_lr': {
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-6
            }
        }
        # Update config with any provided kwargs
        self.config.update(kwargs)

    def _build_model(self):
        """Build LSTM model"""
        if not hasattr(self, 'input_dim'):
            raise ValueError("input_dim must be specified")
        
        # Reshape input if it's 4D to 3D
        if len(self.input_dim) == 4:
            self.input_dim = (self.input_dim[1], self.input_dim[2])
        elif len(self.input_dim) == 3:
            self.input_dim = (self.input_dim[0], self.input_dim[1])
        
        model = create_lstm_model(self.input_dim, self.config)
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model"""
        try:
            logger.info(f"Starting LSTM training with input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
            if X_val is not None:
                logger.info(f"Validation shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
            
            # Check for NaN values
            if np.isnan(X_train).any():
                raise ValueError("Training data contains NaN values")
            if np.isnan(y_train).any():
                raise ValueError("Training labels contain NaN values")
            if X_val is not None and np.isnan(X_val).any():
                raise ValueError("Validation data contains NaN values")
            if y_val is not None and np.isnan(y_val).any():
                raise ValueError("Validation labels contain NaN values")
            
            # Reshape input if it's 4D
            if len(X_train.shape) == 4:
                logger.info(f"Reshaping 4D input to 3D. Original shape: {X_train.shape}")
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
                logger.info(f"New shape after reshape: {X_train.shape}")
            if X_val is not None and len(X_val.shape) == 4:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
            
            # Build model if not already built
            if self.model is None:
                logger.info("Building LSTM model...")
                self.input_dim = X_train.shape[1:]
                logger.info(f"Input dimensions: {self.input_dim}")
                self.model = create_lstm_model(self.input_dim, self.config)
                logger.info("Model built successfully")
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=self.config['early_stopping']['patience'],
                    restore_best_weights=self.config['early_stopping']['restore_best_weights']
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=self.config['reduce_lr']['patience'],
                    factor=self.config['reduce_lr']['factor'],
                    min_lr=self.config['reduce_lr']['min_lr']
                )
            ]
            
            # If no validation data provided, use validation_split from config
            validation_data = None
            validation_split = 0.0
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            else:
                validation_split = self.config['validation_split']
                logger.info(f"Using validation split: {validation_split}")
            
            # Train model
            logger.info("Starting model training...")
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}\n{traceback.format_exc()}")
            raise

    def evaluate(self, X_val, y_val):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.evaluate(X_val, y_val)[0]  # Return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Reshape input if needed
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        
        return self.model.predict(X, verbose=0)

    def save(self, save_dir):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model not trained. Nothing to save.")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(Path(save_dir) / "lstm_model.h5")
        with open(Path(save_dir) / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, model_dir):
        """Load the model"""
        model_path = Path(model_dir) / "lstm_model.h5"
        scaler_path = Path(model_dir) / "scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_dir}")
        
        self.model = tf.keras.models.load_model(str(model_path))
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict_next_draw(self) -> List[int]:
        """Predict the next lottery draw numbers."""
        try:
            # Get most recent sequence for prediction
            sequence = self._get_recent_sequence()
            
            # Make prediction and denormalize
            prediction = self.model.predict(sequence)
            prediction = self._denormalize_prediction(prediction[0])
            
            # Validate and fix prediction if needed
            is_valid, error_msg = validate_prediction_format(prediction)
            if not is_valid:
                logger.warning(f"Invalid prediction: {error_msg}. Fixing prediction...")
                prediction = fix_prediction(prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}\n{traceback.format_exc()}")
            # Fallback: return random numbers
            return fix_prediction([])  # Empty list will be filled with random numbers

    def predict_lstm_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained LSTM model."""
        try:
            # Ensure X is 3D
            if len(X.shape) == 2:
                X = X.reshape(1, X.shape[0], X.shape[1])
            
            # Scale features
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.scaler = MinMaxScaler()
                # Fit scaler on training data if available
                if hasattr(self, 'X_train') and self.X_train is not None:
                    self.scaler.fit(self.X_train.reshape(-1, self.X_train.shape[-1]))
                else:
                    # If no training data, fit on current data
                    self.scaler.fit(X.reshape(-1, X.shape[-1]))
            
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Make predictions
            predictions = self.model.predict(X_scaled, verbose=0)
            
            # Convert predictions to integers
            predictions = np.clip(predictions.astype(int), 1, 60)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            raise

@monitor_gpu
def train_lstm_model(X_train: np.ndarray,
                    y_train: np.ndarray,
                    config: dict,
                    validation_data: tuple = None,
                    callbacks: list = None) -> Tuple[tf.keras.Model, MinMaxScaler]:
    """Train LSTM model with proper input shape handling and preprocessing"""
    try:
        # Ensure X_train is 3D (samples, timesteps, features)
        if len(X_train.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, timesteps, features), got shape {X_train.shape}")
            
        # Get input shape from data
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Preprocess data
        X_scaled, y_normalized, scaler = preprocess_data(X_train, y_train, config.get('look_back', 200))
        
        # Create model with correct input shape
        model = create_lstm_model(input_shape, config)
        
        # Setup callbacks if not provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
        # Train model with memory monitoring
        logger.info("Starting LSTM model training...")
        history = model.fit(
            X_scaled,
            y_normalized,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training results
        logger.info("LSTM training completed")
        logger.info(f"Final loss: {history.history['loss'][-1]:.4f}")
        if validation_data:
            logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"LSTM training failed: {str(e)}\n{traceback.format_exc()}")
        raise

def generate_random_valid_numbers(num_predictions: int = 1) -> np.ndarray:
    """Generate random valid lottery numbers"""
    return np.array([sorted(np.random.choice(range(1, 60), 6, replace=False)) 
                    for _ in range(num_predictions)])

def load_pretrained_lstm_model(model_path: str = "models/checkpoints/lstm_model_checkpoint.h5") -> Optional[tf.keras.Model]:
    """Load a pretrained LSTM model from disk"""
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

def ensure_valid_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Ensure predictions are valid lottery numbers (6 unique numbers between 1-59)
    
    Args:
        predictions: Raw model predictions
        
    Returns:
        Array of valid lottery number predictions
    """
    valid_predictions = []
    for pred in predictions:
        # Round and clip to valid range
        pred = np.clip(np.round(pred), 1, 59)
        
        # Get unique numbers
        unique_nums = set(pred.astype(int))
        
        # If we have more than 6 numbers, take the first 6
        if len(unique_nums) > 6:
            unique_nums = set(sorted(list(unique_nums))[:6])
            
        # If we have less than 6 numbers, add random ones
        available_nums = list(set(range(1, 60)) - unique_nums)
        while len(unique_nums) < 6:
            # Sort available numbers for deterministic selection
            available_nums.sort()
            new_num = available_nums.pop(0)
            unique_nums.add(new_num)
                
        # Sort the numbers
        valid_predictions.append(sorted(list(unique_nums)))
        
    return np.array(valid_predictions)

def predict_next_numbers(model: tf.keras.Model, 
                        input_data: np.ndarray, 
                        scaler: MinMaxScaler) -> np.ndarray:
    """Predict next lottery numbers using the trained model"""
    # Ensure input is scaled
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    
    # Reshape if needed
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)
    
    # Scale input
    input_scaled = scaler.transform(input_data.reshape(-1, input_data.shape[-1]))
    input_scaled = input_scaled.reshape(input_data.shape)
    
    # Get predictions
    predictions = model.predict(input_scaled)
    
    # Inverse transform predictions
    predictions_unscaled = scaler.inverse_transform(predictions)
    
    # Round to nearest valid lottery numbers and sort
    predictions_rounded = np.round(predictions_unscaled).astype(int)
    predictions_rounded = np.clip(predictions_rounded, 1, 49)  # Ensure valid range
    predictions_rounded.sort()
    
    return predictions_rounded

def predict_lstm_model(model, X, scaler, batch_size=32):
    """Make predictions with proper batching and validation."""
    try:
        # Scale input data
        X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Make predictions in batches
        predictions = model.predict(X_scaled, batch_size=batch_size, verbose=0)
        
        # Denormalize and ensure valid range
        predictions = np.clip(predictions * 59.0 + 1, 1, 59)
        
        # Round to integers
        predictions = np.round(predictions).astype(int)
        
        # Process each prediction to ensure uniqueness and correct count
        valid_predictions = []
        for pred in predictions:
            # Get unique numbers and sort
            unique_nums = sorted(list(set(pred)))
            
            # Ensure exactly 6 numbers
            if len(unique_nums) > 6:
                unique_nums = unique_nums[:6]
            while len(unique_nums) < 6:
                available = list(set(range(1, 60)) - set(unique_nums))
                unique_nums.append(np.random.choice(available))
                unique_nums.sort()
            
            valid_predictions.append(unique_nums)
        
        return np.array(valid_predictions)
        
    except Exception as e:
        logging.error(f"Error in LSTM prediction: {str(e)}\n{traceback.format_exc()}")
        # Return random valid numbers as fallback
        return np.array([sorted(np.random.choice(range(1, 60), 6, replace=False)) 
                        for _ in range(len(X))])

def build_lstm_model(input_shape: Tuple[int, int, int] = (200, 6, 1)) -> tf.keras.Model:
    """Build LSTM model with the correct input shape."""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        
        # Single LSTM layer with higher dropout
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second LSTM layer
        LSTM(32),
        BatchNormalization(),
        Dropout(0.4),
        
        # Single dense layer
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Output layer with sigmoid activation
        Dense(6, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='mse',
        metrics=['mae']
    )
    
    return model 