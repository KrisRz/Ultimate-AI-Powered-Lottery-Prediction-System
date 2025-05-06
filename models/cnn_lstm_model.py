import os
import sys
import logging
import numpy as np
import tensorflow as tf
from typing import Tuple
from pathlib import Path
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, InputLayer, Conv2D, MaxPooling2D, Reshape, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import traceback

logger = logging.getLogger(__name__)

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Force all tensors to use float32
tf.keras.backend.set_floatx('float32')

# Set global mixed precision policy
policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class CNNLSTMModel:
    def __init__(self, input_dim=None, name='cnn_lstm', **kwargs):
        self.name = name
        self.input_dim = input_dim
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Default configuration
        self.config = {
            'epochs': 100,
            'batch_size': 32,
            'cnn_filters': [64, 32],
            'cnn_kernel_size': 3,
            'lstm_units': [50, 25],
            'dense_units': [20],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'early_stopping': {
                'patience': 20,
                'restore_best_weights': True,
                'min_delta': 0.0001
            },
            'reduce_lr': {
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-6
            }
        }
        self.config.update(kwargs)
        
    def _build_model(self, input_shape: Tuple[int, int, int, int]) -> tf.keras.Model:
        """Build the CNN-LSTM model"""
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape[1:])
        
        # CNN layers
        x = tf.keras.layers.Conv2D(
            self.config['cnn_filters'][0],
            (3, 3),
            activation='relu',
            padding='same'
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        
        x = tf.keras.layers.Conv2D(
            self.config['cnn_filters'][1],
            (3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        
        # Reshape for LSTM
        x = tf.keras.layers.Reshape((x.shape[1], -1))(x)
        
        # LSTM layers
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.config['lstm_units'][0],
                return_sequences=True
            )
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.config['lstm_units'][1]
            )
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(
            self.config['dense_units'][0],
            activation='relu'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config['dropout'])(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(6, activation='linear')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=1.0,
            clipvalue=0.5
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> tf.keras.callbacks.History:
        """Train CNN-LSTM model"""
        try:
            logger.info(f"Starting CNN-LSTM training with input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
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
            
            # Ensure input has correct shape (batch_size, timesteps, features, channels)
            if len(X_train.shape) != 4:
                raise ValueError(f"Expected 4D input (batch_size, timesteps, features, channels), got shape {X_train.shape}")
            
            # Build model if not already built
            if self.model is None:
                logger.info("Building CNN-LSTM model...")
                self.input_dim = X_train.shape[1:]
                logger.info(f"Input dimensions: {self.input_dim}")
                self.model = self._build_model(X_train.shape)
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
            logger.error(f"Error training CNN-LSTM model: {str(e)}\n{traceback.format_exc()}")
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
            
        # Ensure input has correct shape
        if len(X.shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, timesteps, features, channels), got shape {X.shape}")
            
        return self.model.predict(X, verbose=0)
    
    def save(self, save_dir):
        """Save the model"""
        if self.model is None:
            raise ValueError("Model not trained. Nothing to save.")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(Path(save_dir) / "cnn_lstm_model.h5")
    
    def load(self, model_dir):
        """Load the model"""
        model_path = Path(model_dir) / "cnn_lstm_model.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in {model_dir}")
        self.model = tf.keras.models.load_model(str(model_path))

def create_model(config):
    with tf.device('/CPU:0'):
        inputs = tf.keras.layers.Input(shape=(config['sequence_length'], config['num_features']), dtype=tf.float32)
        
        # CNN layers
        x = tf.keras.layers.Conv1D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='same',
            dtype=tf.float32
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        
        # LSTM layers
        x = tf.keras.layers.LSTM(
            units=config['lstm_units'],
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            dtype=tf.float32
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        
        x = tf.keras.layers.LSTM(
            units=config['lstm_units'],
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            dtype=tf.float32
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(
            units=config['dense_units'],
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']),
            dtype=tf.float32
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(config['dropout_rate'])(x)
        
        outputs = tf.keras.layers.Dense(
            units=config['num_features'],
            activation='sigmoid',
            dtype=tf.float32
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        
        return model

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate prediction accuracy with multiple metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Number matching accuracy
    correct_numbers = np.sum(y_true == y_pred)
    total_numbers = y_true.size
    metrics['number_accuracy'] = (correct_numbers / total_numbers) * 100
    
    # Exact match accuracy (all 6 numbers correct)
    exact_matches = np.sum(np.all(y_true == y_pred, axis=1))
    total_predictions = len(y_true)
    metrics['exact_match_accuracy'] = (exact_matches / total_predictions) * 100
    
    # Partial match accuracy (3+ numbers correct)
    partial_matches = np.sum(np.sum(y_true == y_pred, axis=1) >= 3)
    metrics['partial_match_accuracy'] = (partial_matches / total_predictions) * 100
    
    return metrics

def train_cnn_lstm_model(X_train: np.ndarray,
                        y_train: np.ndarray,
                        config: dict,
                        validation_data: tuple = None,
                        callbacks: list = None) -> Tuple[tf.keras.Model, MinMaxScaler]:
    """Train CNN-LSTM model with given configuration.
    
    Args:
        X_train: Training data
        y_train: Training labels
        config: Model configuration
        validation_data: Optional validation data tuple (X_val, y_val)
        callbacks: Optional list of callbacks
        
    Returns:
        Tuple of (trained model, scaler)
    """
    try:
        # Create and compile model
        model = create_model(config)
        
        # Initialize scaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        
        # Scale validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.get('early_stopping_patience', 10),
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        
        # Train model
        model.fit(
            X_train_scaled, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error training CNN-LSTM model: {str(e)}")
        raise

def predict_cnn_lstm_model(model: Sequential, X: np.ndarray, scaler: MinMaxScaler) -> Tuple[np.ndarray, dict]:
    """
    Generate predictions using trained CNN-LSTM model and return accuracy metrics
    
    Args:
        model: Trained CNN-LSTM model
        X: Input features for prediction
        scaler: Fitted scaler
        
    Returns:
        Tuple of (predictions, accuracy_metrics)
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")
            
        # Validate input shape
        if len(X.shape) != 3:
            raise ValueError(f"Input must be 3D (samples, timesteps, features), got shape {X.shape}")
        if X.shape[1] != 10:  # Expected timesteps
            raise ValueError(f"Expected 10 timesteps, got {X.shape[1]}")
            
        # Scale input - reshape to 2D for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
            
        # Generate predictions
        predictions = model.predict(X_scaled)
        
        # Scale predictions back to original range
        predictions = predictions * 60.0
        
        # Ensure valid predictions
        predictions = np.clip(np.round(predictions), 1, 59)
        
        # Calculate accuracy metrics
        accuracy_metrics = evaluate_predictions(X, predictions)
        
        logger.info("Prediction Accuracy Metrics:")
        logger.info(f"Number Accuracy: {accuracy_metrics['number_accuracy']:.2f}%")
        logger.info(f"Exact Match Accuracy: {accuracy_metrics['exact_match_accuracy']:.2f}%")
        logger.info(f"Partial Match Accuracy (3+ numbers): {accuracy_metrics['partial_match_accuracy']:.2f}%")
        logger.info(f"MAE: {accuracy_metrics['mae']:.4f}")
        logger.info(f"RMSE: {accuracy_metrics['rmse']:.4f}")
        logger.info(f"MAPE: {accuracy_metrics['mape']:.2f}%")
        
        return predictions, accuracy_metrics
        
    except Exception as e:
        logging.error(f"Error in CNN-LSTM prediction: {str(e)}")
        raise

def save_cnn_lstm_model(models: Tuple[Sequential, MinMaxScaler], save_dir: str) -> None:
    """
    Save trained CNN-LSTM model and scaler
    
    Args:
        models: Tuple of (trained CNN-LSTM model, scaler)
        save_dir: Directory to save model files
    """
    try:
        model, scaler = models
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model architecture and weights
        model.save(save_path / "cnn_lstm_model.h5", save_format='h5')
        
        # Save optimizer state
        symbolic_weights = getattr(model.optimizer, 'weights', None)
        if symbolic_weights:
            optimizer_weights = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(save_path / "optimizer.pkl", 'wb') as f:
                pickle.dump({
                    'class_name': model.optimizer.__class__.__name__,
                    'config': model.optimizer.get_config(),
                    'weights': optimizer_weights
                }, f)
        
        # Save scaler
        with open(save_path / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
            
    except Exception as e:
        logging.error(f"Error saving CNN-LSTM model: {str(e)}")
        raise

def load_pretrained_cnn_lstm_model(model_dir: str) -> Tuple[Sequential, MinMaxScaler]:
    """
    Load pretrained CNN-LSTM model and scaler
    
    Args:
        model_dir: Directory containing saved model files
        
    Returns:
        Tuple of (loaded CNN-LSTM model, scaler)
    """
    try:
        model_path = Path(model_dir)
        
        # Load model
        model = tf.keras.models.load_model(model_path / "cnn_lstm_model.h5")
        
        # Load optimizer state if available
        optimizer_path = model_path / "optimizer.pkl"
        if optimizer_path.exists():
            with open(optimizer_path, 'rb') as f:
                optimizer_dict = pickle.load(f)
                
            # Recreate optimizer with saved configuration
            optimizer_class = getattr(tf.keras.optimizers, optimizer_dict['class_name'])
            optimizer = optimizer_class.from_config(optimizer_dict['config'])
            
            # Set optimizer weights after compiling model
            model.compile(optimizer=optimizer, loss='mse')
            
            # Set optimizer weights if model has been trained
            if len(optimizer_dict['weights']) > 0:
                optimizer.set_weights(optimizer_dict['weights'])
        
        # Load scaler
        with open(model_path / "scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
        
    except Exception as e:
        logging.error(f"Error loading CNN-LSTM model: {str(e)}")
        raise

def build_cnn_lstm_model(input_shape: Tuple[int, int, int] = (200, 6, 1)) -> tf.keras.Model:
    """Build CNN-LSTM model with the correct input shape."""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        
        # Simpler CNN layers
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Single CNN layer
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Reshape for LSTM
        Reshape((-1, 16)),
        
        # Single LSTM layer
        LSTM(32),
        BatchNormalization(),
        Dropout(0.4),
        
        # Single dense layer
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Output layer with sigmoid to bound values
        Dense(6, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='mse',
        metrics=['mae']
    )
    
    return model 