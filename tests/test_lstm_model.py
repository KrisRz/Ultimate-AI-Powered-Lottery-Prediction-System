import unittest
import numpy as np
import pandas as pd
from models.lstm_model import train_lstm_model, predict_lstm_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from models.utils import LOOK_BACK
import os

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.n_samples = 100
        self.X = np.random.rand(self.n_samples, 10)  # 10 features
        self.y = np.random.randint(1, 60, size=(self.n_samples, 6))  # 6 target numbers
        
        # Create sequences for LSTM input (samples, timesteps, features)
        self.X_reshaped = np.expand_dims(self.X, axis=1)  # Shape: (n_samples, timesteps=1, features=10)
        
        # Validate input shape
        assert self.X_reshaped.shape == (self.n_samples, 1, 10), f"Invalid shape: {self.X_reshaped.shape}"
        
        # Default config for testing
        self.config = {
            'lstm_units_1': 64,
            'lstm_units_2': 32,
            'dropout_rate': 0.2,
            'l2_reg': 0.01,
            'dense_units': 64,
            'learning_rate': 0.001,
            'epochs': 2,  # Small number for testing
            'batch_size': 32,
            'look_back': LOOK_BACK
        }
            
    def test_model_creation(self):
        """Test if LSTM model can be created with valid parameters"""
        try:
            model, scaler = train_lstm_model(self.X_reshaped, self.y, self.config)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, tf.keras.Model)
            self.assertIsNotNone(scaler)
            self.assertIsInstance(scaler, MinMaxScaler)
        except Exception as e:
            self.fail(f"Model creation failed: {str(e)}")
            
    def test_model_training(self):
        """Test if model can be trained successfully"""
        model, scaler = train_lstm_model(self.X_reshaped, self.y, self.config)
        history = model.fit(
            scaler.transform(self.X_reshaped.reshape(-1, 10)).reshape(self.X_reshaped.shape),
            self.y,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        self.assertIsNotNone(history.history)
        self.assertTrue('loss' in history.history)
        self.assertTrue('mae' in history.history)
            
    def test_prediction_shape(self):
        """Test if predictions have correct shape and range"""
        model, scaler = train_lstm_model(self.X_reshaped, self.y, self.config)
        predictions = predict_lstm_model(model, scaler, self.X)
        
        self.assertEqual(predictions.shape, (len(self.X), 6))
        self.assertTrue(np.all(predictions >= 1))
        self.assertTrue(np.all(predictions <= 59))
        
    def test_prediction_uniqueness(self):
        """Test if predictions contain unique numbers"""
        model, scaler = train_lstm_model(self.X_reshaped, self.y, self.config)
        predictions = predict_lstm_model(model, scaler, self.X)
        
        for pred in predictions:
            unique_nums = len(set(pred.astype(int)))
            self.assertEqual(unique_nums, 6, "Predictions should contain 6 unique numbers")
            self.assertTrue(all(1 <= x <= 59 for x in pred), "All numbers should be between 1 and 59")
            
    def test_invalid_input(self):
        """Test model behavior with invalid input"""
        model, scaler = train_lstm_model(self.X_reshaped, self.y, self.config)
        
        # Test with wrong input shape
        invalid_X = np.random.rand(10, 5)  # Wrong feature count
        with self.assertRaises(ValueError):
            predict_lstm_model(model, scaler, invalid_X)
            
        # Test with NaN values
        invalid_X = np.full((10, 10), np.nan)
        with self.assertRaises(ValueError):
            predict_lstm_model(model, scaler, invalid_X)
            
    def test_model_save_load(self):
        """Test if model can be saved and loaded"""
        model, scaler = train_lstm_model(self.X_reshaped, self.y, self.config)
        
        # Save model
        model.save('test_lstm_model.h5')
        
        # Load model
        loaded_model = tf.keras.models.load_model('test_lstm_model.h5')
        
        # Compare predictions
        orig_pred = predict_lstm_model(model, scaler, self.X)
        loaded_pred = predict_lstm_model(loaded_model, scaler, self.X)
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
        
        # Cleanup
        if os.path.exists('test_lstm_model.h5'):
            os.remove('test_lstm_model.h5')

if __name__ == '__main__':
    unittest.main() 