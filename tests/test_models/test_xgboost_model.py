import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import xgboost as xgb
from models.xgboost_model import (
    objective,
    train_xgboost_model,
    predict_xgboost_model
)
from sklearn.preprocessing import StandardScaler

class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        n_samples = 100
        n_features = 10
        self.X_train = np.random.rand(n_samples, n_features)
        self.y_train = np.random.randint(1, 60, size=(n_samples, 6))
        
        # Create mock XGBoost model
        self.mock_xgb_model = MagicMock()
        self.mock_xgb_model.feature_importances_ = np.random.rand(n_features)
        self.mock_xgb_model.predict.return_value = np.array([10, 20, 30, 40, 50, 60])
        
        # Create mock scaler
        self.mock_scaler = MagicMock(spec=StandardScaler)
        self.mock_scaler.transform.return_value = np.random.rand(n_samples, n_features)
        self.mock_scaler.fit_transform.return_value = np.random.rand(n_samples, n_features)
    
    def test_objective_function(self):
        # Create small sample data for CV
        X = np.random.rand(50, 5)
        y = np.random.randint(1, 60, size=50)
        
        # Mock trial object
        class MockTrial:
            def suggest_int(self, name, low, high):
                return 100 if name == 'n_estimators' else 5
                
            def suggest_float(self, name, low, high, log=False):
                return 0.1
        
        trial = MockTrial()
        
        # Test objective function
        with patch('xgboost.XGBRegressor') as mock_xgb:
            mock_model = MagicMock()
            mock_model.score.return_value = 0.8
            mock_xgb.return_value = mock_model
            
            result = objective(trial, X, y, cv=2)
            
            # Check result is a float and within reasonable range
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, -1.0)
            self.assertLessEqual(result, 1.0)
            
            # Check XGBRegressor was called
            mock_xgb.assert_called()
    
    @patch('models.xgboost_model.xgb.XGBRegressor')
    def test_train_xgboost_model_without_tuning(self, mock_xgb_class):
        # Configure mock
        mock_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(mock_models):
            model.feature_importances_ = np.random.rand(self.X_train.shape[1])
        mock_xgb_class.side_effect = mock_models
        
        # Test without hyperparameter tuning
        models, scaler = train_xgboost_model(
            self.X_train, 
            self.y_train, 
            params={'objective': 'reg:squarederror', 'max_depth': 5},
            tune_hyperparams=False
        )
        
        # Check results
        self.assertEqual(len(models), 6)
        self.assertIsInstance(scaler, StandardScaler)
        
        # Verify XGBRegressor was called 6 times (once for each number position)
        self.assertEqual(mock_xgb_class.call_count, 6)
        
        # Check that fit was called for each model
        for model in mock_models:
            model.fit.assert_called_once()
    
    @patch('models.xgboost_model.optuna.create_study')
    @patch('models.xgboost_model.xgb.XGBRegressor')
    def test_train_xgboost_model_with_tuning(self, mock_xgb_class, mock_create_study):
        # Configure mocks
        mock_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(mock_models):
            model.feature_importances_ = np.random.rand(self.X_train.shape[1])
        mock_xgb_class.side_effect = mock_models
        
        # Mock Optuna study
        mock_study = MagicMock()
        mock_study.best_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 7
        }
        mock_create_study.return_value = mock_study
        
        # Test with hyperparameter tuning
        models, scaler = train_xgboost_model(
            self.X_train, 
            self.y_train, 
            tune_hyperparams=True,
            n_trials=10
        )
        
        # Check results
        self.assertEqual(len(models), 6)
        self.assertIsInstance(scaler, StandardScaler)
        
        # Verify Optuna was used for parameter tuning
        mock_create_study.assert_called()
        mock_study.optimize.assert_called()
        
        # Verify XGBRegressor was called 6 times
        self.assertEqual(mock_xgb_class.call_count, 6)
    
    def test_train_xgboost_model_input_validation(self):
        # Test with invalid input shapes
        with self.assertRaises(ValueError):
            # Test with 1D y_train
            train_xgboost_model(
                self.X_train,
                np.random.randint(1, 60, size=100),  # 1D array
                tune_hyperparams=False
            )
        
        # Test with non-numpy array inputs (should convert automatically)
        X_list = self.X_train.tolist()
        y_list = self.y_train.tolist()
        
        with patch('models.xgboost_model.xgb.XGBRegressor') as mock_xgb:
            mock_model = MagicMock()
            mock_model.feature_importances_ = np.random.rand(self.X_train.shape[1])
            mock_xgb.return_value = mock_model
            
            # Should not raise exception
            models, scaler = train_xgboost_model(
                X_list,
                y_list,
                tune_hyperparams=False
            )
            
            self.assertEqual(len(models), 6)
    
    @patch('models.xgboost_model.xgb.XGBRegressor')
    def test_train_xgboost_model_error_handling(self, mock_xgb_class):
        # Configure mock to raise exception on fit
        mock_xgb_class.return_value = MagicMock()
        mock_xgb_class.return_value.fit.side_effect = Exception("Training failed")
        
        # Should not raise exception but create fallback models
        with patch('logging.error') as mock_log_error:
            with patch('logging.warning') as mock_log_warning:
                with patch('logging.info') as mock_log_info:
                    models, scaler = train_xgboost_model(
                        self.X_train,
                        self.y_train,
                        tune_hyperparams=False
                    )
                    
                    # Check logging calls
                    mock_log_error.assert_called()
                    mock_log_warning.assert_called()
                    
                    # Should still return 6 models (fallbacks)
                    self.assertEqual(len(models), 6)
    
    def test_predict_xgboost_model(self):
        # Setup test data for prediction
        X_test = np.random.rand(5, 10)  # 5 samples, 10 features
        
        # Mock models list
        mock_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(mock_models):
            model.predict.return_value = np.full(X_test.shape[0], i * 10 + 5)
        
        # Test prediction function
        predictions = predict_xgboost_model(mock_models, self.mock_scaler, X_test)
        
        # Check shape of predictions
        self.assertEqual(predictions.shape, (5, 6))
        
        # Check each model was called
        for model in mock_models:
            model.predict.assert_called_once()
        
        # Scaler should be called
        self.mock_scaler.transform.assert_called_once()
    
    def test_predict_xgboost_model_single_sample(self):
        # Test with a single sample
        X_test = np.random.rand(10)  # single sample with 10 features
        
        # Mock models list
        mock_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(mock_models):
            model.predict.return_value = np.array([i * 10 + 5])
        
        # Test prediction function
        with patch('models.xgboost_model.ensure_valid_prediction') as mock_ensure_valid:
            # Configure ensure_valid_prediction to return fixed values
            mock_ensure_valid.return_value = [5, 15, 25, 35, 45, 55]
            
            prediction = predict_xgboost_model(mock_models, self.mock_scaler, X_test)
            
            # Check that ensure_valid_prediction was called
            mock_ensure_valid.assert_called_once()
            
            # Check prediction
            self.assertEqual(len(prediction), 6)
            self.assertEqual(prediction, [5, 15, 25, 35, 45, 55])
    
    def test_predict_xgboost_model_error_handling(self):
        # Test error handling in prediction
        X_test = np.random.rand(5, 10)
        
        # Mock scaler to raise exception
        self.mock_scaler.transform.side_effect = Exception("Transform failed")
        
        # Should not raise exception but return fallback prediction
        with patch('logging.error') as mock_log_error:
            with patch('models.xgboost_model.ensure_valid_prediction') as mock_ensure_valid:
                mock_ensure_valid.return_value = [1, 2, 3, 4, 5, 6]
                
                prediction = predict_xgboost_model([], self.mock_scaler, X_test)
                
                # Check that error was logged
                mock_log_error.assert_called()
                
                # Check that ensure_valid_prediction was called for fallback
                mock_ensure_valid.assert_called_once()
                
                # Should return the fallback prediction
                self.assertEqual(prediction, [1, 2, 3, 4, 5, 6])

if __name__ == '__main__':
    unittest.main() 