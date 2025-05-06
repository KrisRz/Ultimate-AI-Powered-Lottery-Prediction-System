import os
import sys
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, Reshape, Activation, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Cropping2D, LSTM, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any, Optional
from scripts.utils.model_utils import (
    setup_gpu_memory_growth,
    set_random_seed,
    create_sequences,
    normalize_data,
    denormalize_data,
    create_callbacks,
    validate_predictions,
    log_training_errors,
    ensure_valid_prediction,
    log_prediction_errors
)
from tensorflow.keras.optimizers import Adam

from .base import BaseModel

logger = logging.getLogger(__name__)

def create_encoder_decoder(input_dim: int, config: Dict[str, Any]) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Create encoder and decoder models with improved architecture"""
    # Input layer
    input_layer = tf.keras.layers.Input(shape=(10, 6, 1))
    
    # Encoder - Deeper architecture with residual connections
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    skip1 = x
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip1])
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip2 = x
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip2])
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder - Symmetric to encoder with skip connections
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Add()([x, skip2])
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Add()([x, skip1])
    
    # Output with attention mechanism
    attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Ensure output shape matches input shape
    decoded = tf.keras.layers.Cropping2D(((0, decoded.shape[1] - 10), (0, decoded.shape[2] - 6)))(decoded)
    
    # Create models
    encoder = tf.keras.Model(input_layer, encoded)
    autoencoder = tf.keras.Model(input_layer, decoded)
    
    # Compile with improved optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
        clipnorm=1.0,
        clipvalue=0.5,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    autoencoder.compile(
        optimizer=optimizer,
        loss=['mse', 'mae'],
        loss_weights=[0.7, 0.3],
        metrics=['mae']
    )
    
    return encoder, autoencoder

@log_training_errors
def train_autoencoder_model(X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[Tuple[tf.keras.Model, tf.keras.Model], StandardScaler]:
    """Train autoencoder model for lottery number prediction.

    Args:
        X (np.ndarray): Training features
        y (np.ndarray): Target values
        **kwargs: Additional arguments including:
            - epochs (int): Number of training epochs
            - batch_size (int): Batch size for training
            - encoder_dims (List[int]): Dimensions of encoder layers
            - decoder_dims (List[int]): Dimensions of decoder layers
            - latent_dim (int): Dimension of latent space
            - activation (str): Activation function
            - learning_rate (float): Learning rate for optimizer

    Returns:
        Tuple[Tuple[tf.keras.Model, tf.keras.Model], StandardScaler]: Trained models and scaler
    """
    try:
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match")
        if X.shape[1] < 2:
            raise ValueError(f"X must have at least 2 features, got {X.shape[1]}")

        # Get parameters from kwargs with defaults
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        encoder_dims = kwargs.get('encoder_dims', [64, 32, 16])
        decoder_dims = kwargs.get('decoder_dims', [16, 32, 64])
        latent_dim = kwargs.get('latent_dim', 16)
        activation = kwargs.get('activation', 'relu')
        learning_rate = kwargs.get('learning_rate', 0.001)
        dropout = kwargs.get('dropout', 0.2)

        # Scale input data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create models
        encoder, decoder = create_encoder_decoder(
            input_dim=X.shape[1],
            config={
                'learning_rate': learning_rate,
                'dropout': dropout
            }
        )

        # Create optimizers - create new instances for each model
        encoder_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        decoder_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

        # Compile models with separate optimizers
        encoder.compile(optimizer=encoder_optimizer, loss='mse')
        decoder.compile(optimizer=decoder_optimizer, loss='mse')

        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train autoencoder
        encoded = encoder.predict(X_scaled)
        decoder.fit(
            encoded, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        return encoder, decoder, scaler

    except Exception as e:
        logger.error(f"Error in autoencoder training: {str(e)}")
        raise

def predict_autoencoder_model(encoder: tf.keras.Model, decoder: tf.keras.Model, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Generate predictions using trained autoencoder model.

    Args:
        encoder (tf.keras.Model): Trained encoder model
        decoder (tf.keras.Model): Trained decoder model
        scaler (StandardScaler): Fitted scaler for input features
        X (np.ndarray): Input features to predict on

    Returns:
        np.ndarray: Predicted values
    """
    try:
        # Scale input
        X_scaled = scaler.transform(X)

        # Generate predictions
        encoded = encoder.predict(X_scaled, verbose=0)
        predictions = decoder.predict(encoded, verbose=0)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions)

        # Ensure valid predictions
        predictions = ensure_valid_prediction(predictions)

        return predictions

    except Exception as e:
        log_prediction_errors(e, "autoencoder")
        raise

class AutoencoderModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoder = None
        self.model = None
        self.is_trained = False
        
    def build(self, input_shape: Tuple[int, int, int, int]):
        """Build the autoencoder model"""
        self.encoder, self.model = create_encoder_decoder(input_shape[1], self.config)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> tf.keras.callbacks.History:
        """Train the autoencoder model with improved training process"""
        try:
            if self.model is None:
                self.build(X_train.shape)
                
            # Ensure input has correct shape
            if len(X_train.shape) != 4:
                raise ValueError(f"Expected 4D input (batch_size, timesteps, features, channels), got shape {X_train.shape}")
                
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, X_val)
                
            # Create callbacks with improved settings
            callbacks = [
                # Early stopping with longer patience
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    min_delta=1e-4
                ),
                # Cosine decay learning rate schedule
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    cooldown=3
                ),
                # Model checkpoint
                tf.keras.callbacks.ModelCheckpoint(
                    'best_autoencoder.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                ),
                # TensorBoard logging
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1,
                    write_graph=True
                )
            ]
            
            # Add data augmentation
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.GaussianNoise(0.1),
                tf.keras.layers.RandomRotation(0.02),
                tf.keras.layers.RandomZoom(0.1)
            ])
            
            # Train with augmented data
            history = self.model.fit(
                data_augmentation(X_train, training=True),
                X_train,  # Original data as target
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            self.is_trained = True
            return history
            
        except Exception as e:
            logger.error(f"Error training autoencoder model: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using ensemble of predictions with improved post-processing"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        # Ensure input has correct shape
        if len(X.shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, timesteps, features, channels), got shape {X.shape}")
            
        # Generate multiple predictions with different noise levels
        n_predictions = 5
        predictions = []
        
        for i in range(n_predictions):
            # Add random noise
            X_noisy = X + np.random.normal(0, 0.01 * (i + 1), X.shape)
            
            # Get prediction
            pred = self.model.predict(X_noisy, verbose=0)
            
            # Reshape if needed
            if len(pred.shape) > 2:
                pred = pred.reshape(pred.shape[0], -1)[:, :6]
                
            predictions.append(pred)
            
        # Ensemble predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Apply constraints
        ensemble_pred = np.clip(ensemble_pred, 0, 1)  # Ensure values are between 0 and 1
        
        # Sort each row to ensure ascending order
        ensemble_pred = np.sort(ensemble_pred, axis=1)
        
        # Remove duplicates within each prediction
        for i in range(len(ensemble_pred)):
            row = ensemble_pred[i]
            _, unique_indices = np.unique(row, return_index=True)
            if len(unique_indices) < 6:
                # Fill missing values with new random values
                n_missing = 6 - len(unique_indices)
                missing_values = np.random.uniform(row.min(), row.max(), n_missing)
                row = np.concatenate([row[unique_indices], missing_values])
                ensemble_pred[i] = np.sort(row)
        
        return ensemble_pred

    def encode(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        # Reshape to 4D if needed
        if len(X_scaled.shape) != 4:
            X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], X_scaled.shape[2], 1)
        return self.model.predict(X_scaled, verbose=0)

    def create_encoder_decoder(self, input_dim, encoder_dims, decoder_dims, latent_dim, activation='relu'):
        """Create encoder and decoder models"""
        # Input layer
        input_layer = Input(shape=(input_dim, 6, 1))
        
        # Encoder
        x = Conv2D(32, (3, 3), activation=activation, padding='same')(input_layer)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation=activation, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=activation, padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = Conv2D(8, (3, 3), activation=activation, padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation=activation, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation=activation, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(6, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return autoencoder

    def train_autoencoder_model(self, X, y, X_val=None, y_val=None, config=None):
        """Train the autoencoder model"""
        try:
            # Reshape input to match expected dimensions
            X_reshaped = X.reshape(-1, X.shape[1], 6, 1)
            if X_val is not None:
                X_val_reshaped = X_val.reshape(-1, X_val.shape[1], 6, 1)
            
            # Create and compile model if not already done
            if self.model is None:
                self.model = self.create_encoder_decoder(
                    input_dim=X.shape[1],
                    encoder_dims=self.config['encoder_dims'],
                    decoder_dims=self.config['decoder_dims'],
                    latent_dim=self.config['latent_dim'],
                    activation=self.config['activation']
                )
            
            # Train model
            validation_data = (X_val_reshaped, X_val_reshaped) if X_val is not None else None
            history = self.model.fit(
                X_reshaped, X_reshaped,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=validation_data,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ],
                verbose=1
            )
            
            return history
            
        except Exception as e:
            logger.error(f"Error training autoencoder model: {str(e)}")
            raise

    def save(self, path):
        """Save the model to the specified path"""
        if self.model is not None:
            self.model.save(path)
        else:
            raise ValueError("No trained model to save")

def build_autoencoder(input_shape: Tuple[int, int, int] = (200, 6, 1)) -> tf.keras.Model:
    """Build autoencoder model with the correct input shape."""
    # Encoder
    encoder_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    encoded = Dense(64, activation='relu')(x)
    
    # Decoder
    x = Dense(64 * 25 * 3, activation='relu')(encoded)
    x = Reshape((25, 3, 64))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    # Autoencoder
    autoencoder = Model(encoder_input, decoded)
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder 