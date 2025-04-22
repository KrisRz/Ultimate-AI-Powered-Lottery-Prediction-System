import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from models.prediction import (
    predict_with_lstm,
    predict_with_arima,
    predict_with_holtwinters,
    predict_with_linear,
    predict_with_xgboost,
    predict_with_lightgbm,
    predict_with_knn,
    predict_with_gradientboosting,
    predict_with_catboost,
    predict_with_cnn_lstm,
    predict_with_autoencoder,
    predict_with_meta,
    predict_next_draw,
    ensemble_prediction,
    generate_optimized_predictions,
    score_combinations,
    monte_carlo_simulation,
    backtest,
    calculate_prediction_accuracy
)

class TestPredictionFunctions(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        n_samples = 100
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        main_numbers = []
        bonus_numbers = []
        
        for _ in range(n_samples):
            nums = sorted(np.random.choice(range(1, 60), size=6, replace=False))
            bonus = np.random.randint(1, 60)
            main_numbers.append(nums)
            bonus_numbers.append(bonus)
        
        self.df = pd.DataFrame({
            'Draw Date': dates,
            'Main_Numbers': main_numbers,
            'Bonus_Number': bonus_numbers,
            'DayOfWeek': [i % 7 for i in range(n_samples)],
            'Sum': [sum(nums) for nums in main_numbers],
            'Mean': [np.mean(nums) for nums in main_numbers],
            'Unique': [len(set(nums)) for nums in main_numbers],
            'ZScore_Sum': np.random.randn(n_samples),
            'Primes': [sum(1 for n in nums if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]) for nums in main_numbers],
            'Odds': [sum(1 for n in nums if n % 2 == 1) for nums in main_numbers],
            'Evens': [sum(1 for n in nums if n % 2 == 0) for nums in main_numbers],
            'Gaps': [np.mean([nums[i+1] - nums[i] for i in range(len(nums)-1)]) for nums in main_numbers],
            'Freq_10': np.random.rand(n_samples),
            'Freq_20': np.random.rand(n_samples),
            'Freq_50': np.random.rand(n_samples),
            'Pair_Freq': np.random.rand(n_samples),
            'Triplet_Freq': np.random.rand(n_samples)
        })
        
        # Create mock models
        self.mock_lstm_model = MagicMock()
        self.mock_lstm_model.predict.return_value = np.array([[10, 20, 30, 40, 50, 60]])
        
        self.mock_scaler = MagicMock()
        self.mock_scaler.transform.return_value = np.random.rand(200, 1)
        self.mock_scaler.inverse_transform.return_value = np.array([[10, 20, 30, 40, 50, 60]])
        
        self.mock_arima_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(self.mock_arima_models):
            model.predict.return_value = np.array([i * 10 + 1])
        
        self.mock_holtwinters_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(self.mock_holtwinters_models):
            model.forecast.return_value = np.array([i * 10 + 2])
        
        self.mock_ml_models = [MagicMock() for _ in range(6)]
        for i, model in enumerate(self.mock_ml_models):
            model.predict.return_value = np.array([i * 10 + 3])
        
        self.mock_meta_model = MagicMock()
        self.mock_meta_model.predict.return_value = np.array([[10, 20, 30, 40, 50, 60]])
        
        # Create models dictionary
        self.models = {
            'lstm': (self.mock_lstm_model, self.mock_scaler),
            'arima': self.mock_arima_models,
            'holtwinters': self.mock_holtwinters_models,
            'linear': self.mock_ml_models,
            'xgboost': self.mock_ml_models,
            'lightgbm': self.mock_ml_models,
            'knn': self.mock_ml_models,
            'gradientboosting': self.mock_ml_models,
            'catboost': self.mock_ml_models,
            'cnn_lstm': (self.mock_lstm_model, self.mock_scaler),
            'autoencoder': (self.mock_lstm_model, self.mock_scaler),
            'meta': self.mock_meta_model
        }
    
    def test_predict_with_lstm(self):
        result = predict_with_lstm(self.df, {'lstm': (self.mock_lstm_model, self.mock_scaler)})
        self.assertEqual(len(result), 6)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertTrue(all(1 <= x <= 59 for x in result))
        self.assertEqual(len(set(result)), 6)  # All values should be unique
        
        # Test with empty DataFrame
        result = predict_with_lstm(pd.DataFrame(), {'lstm': (self.mock_lstm_model, self.mock_scaler)})
        self.assertEqual(len(result), 6)
    
    def test_predict_with_arima(self):
        result = predict_with_arima(self.df, {'arima': self.mock_arima_models})
        self.assertEqual(len(result), 6)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertTrue(all(1 <= x <= 59 for x in result))
        self.assertEqual(len(set(result)), 6)  # All values should be unique
        
        # Test with empty DataFrame
        result = predict_with_arima(pd.DataFrame(), {'arima': self.mock_arima_models})
        self.assertEqual(len(result), 6)
    
    def test_predict_with_holtwinters(self):
        result = predict_with_holtwinters(self.df, {'holtwinters': self.mock_holtwinters_models})
        self.assertEqual(len(result), 6)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertTrue(all(1 <= x <= 59 for x in result))
        self.assertEqual(len(set(result)), 6)  # All values should be unique
        
        # Test with empty DataFrame
        result = predict_with_holtwinters(pd.DataFrame(), {'holtwinters': self.mock_holtwinters_models})
        self.assertEqual(len(result), 6)
    
    def test_predict_with_ml_models(self):
        # Test all ML-based models
        ml_models = [
            ('linear', predict_with_linear),
            ('xgboost', predict_with_xgboost),
            ('lightgbm', predict_with_lightgbm),
            ('knn', predict_with_knn),
            ('gradientboosting', predict_with_gradientboosting),
            ('catboost', predict_with_catboost),
        ]
        
        for model_name, predict_func in ml_models:
            result = predict_func(self.df, {model_name: self.mock_ml_models})
            self.assertEqual(len(result), 6, f"Failed for {model_name}")
            self.assertTrue(all(isinstance(x, int) for x in result), f"Failed for {model_name}")
            self.assertTrue(all(1 <= x <= 59 for x in result), f"Failed for {model_name}")
            self.assertEqual(len(set(result)), 6, f"Failed for {model_name}")  # All values should be unique
            
            # Test with empty DataFrame
            result = predict_func(pd.DataFrame(), {model_name: self.mock_ml_models})
            self.assertEqual(len(result), 6, f"Failed for {model_name} with empty DataFrame")
    
    def test_predict_with_deep_learning_models(self):
        # Test CNN-LSTM and autoencoder models
        dl_models = [
            ('cnn_lstm', predict_with_cnn_lstm),
            ('autoencoder', predict_with_autoencoder),
        ]
        
        for model_name, predict_func in dl_models:
            result = predict_func(self.df, {model_name: (self.mock_lstm_model, self.mock_scaler)})
            self.assertEqual(len(result), 6, f"Failed for {model_name}")
            self.assertTrue(all(isinstance(x, int) for x in result), f"Failed for {model_name}")
            self.assertTrue(all(1 <= x <= 59 for x in result), f"Failed for {model_name}")
            self.assertEqual(len(set(result)), 6, f"Failed for {model_name}")  # All values should be unique
            
            # Test with empty DataFrame
            result = predict_func(pd.DataFrame(), {model_name: (self.mock_lstm_model, self.mock_scaler)})
            self.assertEqual(len(result), 6, f"Failed for {model_name} with empty DataFrame")
    
    def test_predict_with_meta(self):
        # Create a more specific mock for meta model
        meta_model = MagicMock()
        meta_model.predict.return_value = np.array([[10, 20, 30, 40, 50, 60]])
        
        models = self.models.copy()
        models['meta'] = meta_model
        
        # Define expected input shape
        result = predict_with_meta(self.df, models)
        
        self.assertEqual(len(result), 6)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertTrue(all(1 <= x <= 59 for x in result))
        self.assertEqual(len(set(result)), 6)  # All values should be unique
        
        # Test with empty DataFrame
        result = predict_with_meta(pd.DataFrame(), models)
        self.assertEqual(len(result), 6)
    
    def test_predict_next_draw(self):
        result = predict_next_draw(self.df, self.models)
        
        # Check that predictions exist for all model types
        self.assertEqual(len(result), len(self.models))
        for model_name in self.models.keys():
            self.assertIn(model_name, result)
            self.assertEqual(len(result[model_name]), 6)
            self.assertTrue(all(isinstance(x, int) for x in result[model_name]))
            self.assertTrue(all(1 <= x <= 59 for x in result[model_name]))
            self.assertEqual(len(set(result[model_name])), 6)  # All values should be unique
    
    def test_ensemble_prediction(self):
        # Mock the predict methods for each model
        for model_name, model in self.models.items():
            if isinstance(model, tuple):
                model[0].predict.return_value = [10, 20, 30, 40, 50, 60]
            else:
                for m in model:
                    m.predict.return_value = [10, 20, 30, 40, 50, 60]
        
        # Test with n_predictions = 3
        n_predictions = 3
        result = ensemble_prediction(self.df, self.models, n_predictions)
        
        self.assertEqual(len(result), n_predictions)
        for pred in result:
            self.assertEqual(len(pred), 6)
            self.assertTrue(all(isinstance(x, int) for x in pred))
            self.assertTrue(all(1 <= x <= 59 for x in pred))
            self.assertEqual(len(set(pred)), 6)  # All values should be unique
    
    def test_generate_optimized_predictions(self):
        # Mock the predict methods for each model
        for model_name, model in self.models.items():
            if isinstance(model, tuple):
                model[0].predict.return_value = [10, 20, 30, 40, 50, 60]
            else:
                for m in model:
                    m.predict.return_value = [10, 20, 30, 40, 50, 60]
        
        # Test with n_predictions = 3
        n_predictions = 3
        result = generate_optimized_predictions(self.df, self.models, n_predictions)
        
        self.assertEqual(len(result), n_predictions)
        for pred in result:
            self.assertEqual(len(pred), 6)
            self.assertTrue(all(isinstance(x, int) for x in pred))
            self.assertTrue(all(1 <= x <= 59 for x in pred))
            self.assertEqual(len(set(pred)), 6)  # All values should be unique
    
    def test_score_combinations(self):
        predictions = [
            [1, 2, 3, 4, 5, 6],
            [11, 22, 33, 44, 55, 59],
            [2, 3, 5, 7, 11, 13],  # All primes
            [10, 20, 30, 40, 50, 58]  # All even
        ]
        
        result = score_combinations(predictions, self.df)
        
        self.assertEqual(len(result), len(predictions))
        for pred, score in result:
            self.assertEqual(len(pred), 6)
            self.assertTrue(isinstance(score, float))
    
    def test_monte_carlo_simulation(self):
        n_simulations = 5
        result = monte_carlo_simulation(self.df, n_simulations)
        
        self.assertEqual(len(result), n_simulations)
        for pred in result:
            self.assertEqual(len(pred), 6)
            self.assertTrue(all(isinstance(x, int) for x in pred))
            self.assertTrue(all(1 <= x <= 59 for x in pred))
            self.assertEqual(len(set(pred)), 6)  # All values should be unique
    
    @patch('models.prediction.update_models')
    def test_backtest(self, mock_update_models):
        # Make update_models return our mock models
        mock_update_models.return_value = self.models
        
        # Use a smaller dataframe for faster testing
        df_small = self.df.iloc[:10]
        
        # Test with provided models
        accuracy, predictions = backtest(df_small, self.models)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertEqual(len(predictions), len(df_small) - 1)
        
        # Test with models being generated inside the function
        accuracy, predictions = backtest(df_small)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertEqual(len(predictions), len(df_small) - 1)
    
    def test_calculate_prediction_accuracy(self):
        # Create test data
        predictions = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]
        ]
        
        # Case 1: All correct
        actual = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
        accuracy = calculate_prediction_accuracy(predictions, actual)
        self.assertEqual(accuracy, 1.0)
        
        # Case 2: Half correct
        actual = [[1, 2, 3, 10, 20, 30], [7, 8, 9, 40, 50, 60], [13, 14, 15, 70, 80, 90]]
        accuracy = calculate_prediction_accuracy(predictions, actual)
        self.assertEqual(accuracy, 0.5)
        
        # Case 3: None correct
        actual = [[21, 22, 23, 24, 25, 26], [27, 28, 29, 30, 31, 32], [33, 34, 35, 36, 37, 38]]
        accuracy = calculate_prediction_accuracy(predictions, actual)
        self.assertEqual(accuracy, 0.0)
        
        # Case 4: Empty lists
        accuracy = calculate_prediction_accuracy([], [])
        self.assertEqual(accuracy, 0.0)

if __name__ == '__main__':
    unittest.main() 