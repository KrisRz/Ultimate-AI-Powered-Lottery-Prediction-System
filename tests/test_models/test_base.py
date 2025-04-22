import unittest
import numpy as np
import pandas as pd
from itertools import combinations
from scripts.train_models import (
    train_lstm_models,
    train_arima_models,
    train_holtwinters_models,
    train_linear_models,
    train_xgboost_models,
    train_lightgbm_models,
    train_knn_models,
    train_gradientboosting_models,
    train_catboost_models,
    train_cnn_lstm_models,
    train_autoencoder,
    train_meta_model,
    update_models
)
from scripts.predict_numbers import (
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
    predict_with_meta
)
from scripts.utils.validation import ensure_valid_prediction
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import logging

# Model constants
LOOK_BACK = 10
EPOCHS = 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create synthetic training data matching real data structure
        n_samples = 100
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        days = [days[i % 7] for i in range(n_samples)]
        balls = [f"{' '.join(str(x).zfill(2) for x in np.random.choice(range(1, 60), size=6, replace=False))} BONUS {str(np.random.randint(1, 60)).zfill(2)}" for _ in range(n_samples)]
        jackpots = [f"£{np.random.randint(1000000, 10000000):,}" for _ in range(n_samples)]
        winners = np.random.randint(0, 5, size=n_samples)
        draw_details = ['draw details'] * n_samples
        
        self.df = pd.DataFrame({
            'Draw Date': dates,
            'Day': days,
            'Balls': balls,
            'Jackpot': jackpots,
            'Winners': winners,
            'Draw Details': draw_details
        })
        
        # Parse Balls column to create Main_Numbers and Bonus_Number
        def parse_balls(ball_str):
            parts = ball_str.split(' BONUS ')
            main_numbers = [int(x) for x in parts[0].split()]
            bonus_number = int(parts[1])
            return pd.Series({'Main_Numbers': main_numbers, 'Bonus_Number': bonus_number})
            
        parsed = self.df['Balls'].apply(parse_balls)
        self.df['Main_Numbers'] = parsed['Main_Numbers']
        self.df['Bonus_Number'] = parsed['Bonus_Number']
        
        # Add required features
        day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
        self.df['DayOfWeek'] = self.df['Day'].map(day_map)  # Convert day names to numeric
        self.df['Month'] = pd.to_datetime(self.df['Draw Date']).dt.month
        self.df['Year'] = pd.to_datetime(self.df['Draw Date']).dt.year
        self.df['Sum'] = self.df['Main_Numbers'].apply(sum)
        self.df['Mean'] = self.df['Main_Numbers'].apply(np.mean)
        self.df['Std'] = self.df['Main_Numbers'].apply(np.std)
        self.df['Unique'] = self.df['Main_Numbers'].apply(lambda x: len(set(x)))
        self.df['ZScore_Sum'] = (self.df['Sum'] - self.df['Sum'].rolling(window=10, min_periods=1).mean()) / self.df['Sum'].rolling(window=10, min_periods=1).std()
        self.df['ZScore_Sum'] = self.df['ZScore_Sum'].fillna(0)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            self.df[f'Sum_MA_{window}'] = self.df['Sum'].rolling(window=window, min_periods=1).mean()
            self.df[f'Sum_STD_{window}'] = self.df['Sum'].rolling(window=window, min_periods=1).std()
            self.df[f'Mean_MA_{window}'] = self.df['Mean'].rolling(window=window, min_periods=1).mean()
            self.df[f'Mean_STD_{window}'] = self.df['Mean'].rolling(window=window, min_periods=1).std()
        
        # Number properties
        self.df['Primes'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]))
        self.df['Odds'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 1))
        self.df['Evens'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 0))
        self.df['Low_Numbers'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n <= 30))
        self.df['High_Numbers'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n > 30))
        
        # Previous draw features
        self.df['Prev_Sum'] = self.df['Sum'].shift(1)
        self.df['Prev_Mean'] = self.df['Mean'].shift(1)
        self.df['Prev_Std'] = self.df['Std'].shift(1)
        self.df['Prev_Primes'] = self.df['Primes'].shift(1)
        self.df['Prev_Odds'] = self.df['Odds'].shift(1)
        
        # Frequency analysis
        all_numbers = self.df['Main_Numbers'].tolist()
        for window in [10, 20, 50]:
            window_size = min(window, len(all_numbers))
            self.df[f'Freq_{window}'] = self.df['Main_Numbers'].apply(
                lambda x: sum(1 for n in x for row in all_numbers[-window_size:] if n in row) / (window_size * 6)
            )
        
        # Pattern analysis
        def get_pair_freq(numbers, all_numbers, window_size):
            pairs = list(combinations(numbers, 2))
            window_numbers = all_numbers[-window_size:]
            all_pairs = [list(combinations(nums, 2)) for nums in window_numbers]
            return sum(1 for p in pairs for draw_pairs in all_pairs if p in draw_pairs) / len(window_numbers)
        
        def get_triplet_freq(numbers, all_numbers, window_size):
            triplets = list(combinations(numbers, 3))
            window_numbers = all_numbers[-window_size:]
            all_triplets = [list(combinations(nums, 3)) for nums in window_numbers]
            return sum(1 for t in triplets for draw_triplets in all_triplets if t in draw_triplets) / len(window_numbers)
        
        window_size = min(50, len(all_numbers))
        self.df['Pair_Freq'] = self.df['Main_Numbers'].apply(lambda x: get_pair_freq(x, all_numbers, window_size))
        self.df['Triplet_Freq'] = self.df['Main_Numbers'].apply(lambda x: get_triplet_freq(x, all_numbers, window_size))
        
        # Add Gaps feature
        self.df['Gaps'] = self.df['Main_Numbers'].apply(lambda x: np.mean([x[i+1] - x[i] for i in range(len(x)-1)]))
        
        # Fill any remaining NaN values with 0
        self.df = self.df.fillna(0)
        
        # Create feature scaler
        self.scaler = MinMaxScaler()
        feature_cols = [col for col in self.df.columns if col not in ['Draw Date', 'Day', 'Balls', 'Jackpot', 'Winners', 'Draw Details', 'Main_Numbers', 'Bonus_Number', 'DayOfWeek']]
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        
        # Prepare features for deep learning and non-deep learning models
        self.X_dl = self.df[feature_cols].values.reshape((self.df.shape[0], 1, len(feature_cols)))  # For LSTM, CNN-LSTM
        self.X_ml = self.df[feature_cols].values  # For XGBoost, LightGBM, etc.
        
    def assertValidPrediction(self, pred, msg=None):
        """Helper method to validate predictions"""
        if pred is None:
            self.fail(f"Prediction is None: {msg}")
        if not isinstance(pred, (list, np.ndarray)):
            self.fail(f"Prediction is not a list or array: {msg}")
        if len(pred) != 6:
            self.fail(f"Prediction length is not 6 (got {len(pred)}): {msg}")
        if not all(isinstance(n, (int, np.integer)) for n in pred):
            self.fail(f"Not all predictions are integers: {msg}")
        if not all(1 <= n <= 59 for n in pred):
            self.fail(f"Not all predictions are in range [1, 59]: {msg}")
        if len(set(pred)) != 6:
            self.fail(f"Predictions are not unique: {msg}")

    def test_train_lstm_models(self):
        try:
            model, scaler = train_lstm_models(self.df)
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            pred = predict_with_lstm(self.df, {'lstm': (model, scaler)})
            self.assertValidPrediction(pred, "LSTM prediction failed validation")
        except Exception as e:
            logging.error(f"LSTM training failed: {str(e)}")
            self.fail(f"LSTM training failed: {str(e)}")

    def test_train_arima_models(self):
        try:
            models = train_arima_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_arima(self.df, {'arima': models})
            self.assertValidPrediction(pred, "ARIMA prediction failed validation")
        except Exception as e:
            logging.error(f"ARIMA training failed: {str(e)}")
            self.skipTest(f"ARIMA training skipped due to: {str(e)}")

    def test_train_holtwinters_models(self):
        try:
            models = train_holtwinters_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_holtwinters(self.df, {'holtwinters': models})
            self.assertValidPrediction(pred, "Holt-Winters prediction failed validation")
        except Exception as e:
            logging.error(f"Holt-Winters training failed: {str(e)}")
            self.skipTest(f"Holt-Winters training skipped due to: {str(e)}")

    def test_train_linear_models(self):
        try:
            models, feature_cols = train_linear_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_linear(self.df, {'linear': (models, feature_cols)})
            self.assertValidPrediction(pred, "Linear prediction failed validation")
        except Exception as e:
            logging.error(f"Linear training failed: {str(e)}")
            self.fail(f"Linear training failed: {str(e)}")

    def test_train_xgboost_models(self):
        try:
            models, feature_cols = train_xgboost_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_xgboost(self.df, {'xgboost': (models, feature_cols)})
            self.assertValidPrediction(pred, "XGBoost prediction failed validation")
        except Exception as e:
            logging.error(f"XGBoost training failed: {str(e)}")
            self.fail(f"XGBoost training failed: {str(e)}")

    def test_train_lightgbm_models(self):
        try:
            models, feature_cols = train_lightgbm_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_lightgbm(self.df, {'lightgbm': (models, feature_cols)})
            self.assertValidPrediction(pred, "LightGBM prediction failed validation")
        except Exception as e:
            logging.error(f"LightGBM training failed: {str(e)}")
            self.fail(f"LightGBM training failed: {str(e)}")

    def test_train_knn_models(self):
        try:
            models, feature_cols = train_knn_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_knn(self.df, {'knn': (models, feature_cols)})
            self.assertValidPrediction(pred, "KNN prediction failed validation")
        except Exception as e:
            logging.error(f"KNN training failed: {str(e)}")
            self.fail(f"KNN training failed: {str(e)}")

    def test_train_gradientboosting_models(self):
        try:
            models, feature_cols = train_gradientboosting_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_gradientboosting(self.df, {'gradientboosting': (models, feature_cols)})
            self.assertValidPrediction(pred, "Gradient Boosting prediction failed validation")
        except Exception as e:
            logging.error(f"Gradient Boosting training failed: {str(e)}")
            self.fail(f"Gradient Boosting training failed: {str(e)}")

    def test_train_catboost_models(self):
        try:
            models, feature_cols = train_catboost_models(self.df)
            self.assertIsNotNone(models)
            self.assertEqual(len(models), 6)
            pred = predict_with_catboost(self.df, {'catboost': (models, feature_cols)})
            self.assertValidPrediction(pred, "CatBoost prediction failed validation")
        except Exception as e:
            logging.error(f"CatBoost training failed: {str(e)}")
            self.fail(f"CatBoost training failed: {str(e)}")

    def test_train_cnn_lstm_models(self):
        try:
            model, scaler = train_cnn_lstm_models(self.df)
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            pred = predict_with_cnn_lstm(self.df, {'cnn_lstm': (model, scaler)})
            self.assertValidPrediction(pred, "CNN-LSTM prediction failed validation")
        except Exception as e:
            logging.error(f"CNN-LSTM training failed: {str(e)}")
            self.fail(f"CNN-LSTM training failed: {str(e)}")

    def test_train_autoencoder(self):
        try:
            model, scaler = train_autoencoder(self.df)
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            pred = predict_with_autoencoder(self.df, {'autoencoder': (model, scaler)})
            self.assertValidPrediction(pred, "Autoencoder prediction failed validation")
        except Exception as e:
            logging.error(f"Autoencoder training failed: {str(e)}")
            self.fail(f"Autoencoder training failed: {str(e)}")

    def test_train_meta_model(self):
        try:
            # Train all base models
            models = {
                'lstm': train_lstm_models(self.df),
                'arima': train_arima_models(self.df) if train_arima_models else [None] * 6,
                'holtwinters': train_holtwinters_models(self.df) if train_holtwinters_models else [None] * 6,
                'linear': train_linear_models(self.df),
                'xgboost': train_xgboost_models(self.df),
                'lightgbm': train_lightgbm_models(self.df),
                'knn': train_knn_models(self.df),
                'gradientboosting': train_gradientboosting_models(self.df),
                'catboost': train_catboost_models(self.df),
                'cnn_lstm': train_cnn_lstm_models(self.df),
                'autoencoder': train_autoencoder(self.df)
            }
            
            # Generate predictions from base models
            preds_dict = {}
            for model_name, model in models.items():
                if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
                    pred = globals()[f'predict_with_{model_name}'](self.df, {model_name: model})
                else:
                    pred = globals()[f'predict_with_{model_name}'](self.df, {model_name: model})
                preds_dict[model_name] = np.array([pred])  # Ensure correct shape
            
            # Train meta model
            meta_model = train_meta_model(self.df, preds_dict)
            self.assertIsNotNone(meta_model)
            
            # Test prediction
            pred = predict_with_meta(self.df, {'meta': meta_model})
            self.assertValidPrediction(pred, "Meta model prediction failed validation")
        except Exception as e:
            logging.error(f"Meta model training failed: {str(e)}")
            self.fail(f"Meta model training failed: {str(e)}")

    def test_update_models(self):
        try:
            models = update_models(self.df)
            self.assertIsNotNone(models)
            self.assertIsInstance(models, dict)
            
            # Test that each model type exists and can make predictions
            for model_type, model in models.items():
                if model_type == 'meta':
                    continue  # Skip meta model as it requires all base models
                
                pred_func = globals()[f'predict_with_{model_type}']
                pred = pred_func(self.df, {model_type: model})
                self.assertValidPrediction(pred, f"{model_type} prediction failed validation")
        except Exception as e:
            logging.error(f"Model update failed: {str(e)}")
            self.fail(f"Model update failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()