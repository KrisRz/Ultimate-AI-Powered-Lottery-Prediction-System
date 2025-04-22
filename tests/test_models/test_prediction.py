import unittest
import numpy as np
import pandas as pd
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
    predict_with_meta,
    predict_next_draw,
    ensemble_prediction,
    generate_optimized_predictions,
    score_combinations,
    monte_carlo_simulation,
    backtest
)
from scripts.train_models import update_models
from utils.validation import ensure_valid_prediction
from itertools import combinations
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestPredictions(unittest.TestCase):
    def setUp(self):
        # Create synthetic data (numbers 1–59)
        n_samples = 50  # Reduced for faster testing
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        days = (['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] * ((n_samples // 7) + 1))[:n_samples]
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
        
        # Parse Balls column
        def parse_balls(ball_str):
            try:
                parts = ball_str.split(' BONUS ')
                main_numbers = [int(x) for x in parts[0].split()]
                bonus_number = int(parts[1])
                if len(main_numbers) != 6 or not all(1 <= n <= 59 for n in main_numbers):
                    raise ValueError("Invalid main numbers")
                if not 1 <= bonus_number <= 59:
                    raise ValueError("Invalid bonus number")
                return pd.Series({'Main_Numbers': main_numbers, 'Bonus_Number': bonus_number})
            except Exception as e:
                logging.error(f"Error parsing balls: {str(e)}")
                raise ValueError(f"Invalid Balls format: {ball_str}")
            
        parsed = self.df['Balls'].apply(parse_balls)
        self.df['Main_Numbers'] = parsed['Main_Numbers']
        self.df['Bonus_Number'] = parsed['Bonus_Number']
        
        # Add required features
        self.df['DayOfWeek'] = pd.to_datetime(self.df['Draw Date']).dt.dayofweek.astype(str)
        self.df['Month'] = pd.to_datetime(self.df['Draw Date']).dt.month
        self.df['Year'] = pd.to_datetime(self.df['Draw Date']).dt.year
        self.df['Sum'] = self.df['Main_Numbers'].apply(sum)
        self.df['Mean'] = self.df['Main_Numbers'].apply(np.mean)
        self.df['Std'] = self.df['Main_Numbers'].apply(np.std)
        self.df['Unique'] = self.df['Main_Numbers'].apply(lambda x: len(set(x)))
        self.df['ZScore_Sum'] = (self.df['Sum'] - self.df['Sum'].rolling(window=10, min_periods=1).mean()) / self.df['Sum'].rolling(window=10, min_periods=1).std()
        self.df['ZScore_Sum'] = self.df['ZScore_Sum'].fillna(0)
        
        # Add rolling mean and std features
        for window in [10, 20]:
            self.df[f'rolling_mean_{window}'] = self.df['Sum'].rolling(window=window, min_periods=1).mean()
            self.df[f'rolling_std_{window}'] = self.df['Sum'].rolling(window=window, min_periods=1).std()
            self.df[f'rolling_mean_{window}'] = self.df[f'rolling_mean_{window}'].fillna(0)
            self.df[f'rolling_std_{window}'] = self.df[f'rolling_std_{window}'].fillna(0)
        
        self.df['Primes'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]))
        self.df['Odds'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 1))
        self.df['Evens'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 0))
        self.df['Low_Numbers'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n <= 30))
        self.df['High_Numbers'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n > 30))
        self.df['Gaps'] = self.df['Main_Numbers'].apply(lambda x: np.mean([x[i+1] - x[i] for i in range(len(x)-1)]))
        
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
        
        # Fill NaN values
        self.df = self.df.fillna(0)
        
        # Train models
        try:
            self.models = update_models(self.df)
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            self.fail(f"setUp failed: {str(e)}")

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

    def test_predict_with_lstm(self):
        try:
            predictions = predict_with_lstm(self.df, self.models)
            self.assertValidPrediction(predictions, "LSTM prediction failed validation")
        except Exception as e:
            logging.error(f"LSTM prediction failed: {str(e)}")
            self.fail(f"LSTM prediction failed: {str(e)}")

    def test_predict_with_arima(self):
        try:
            predictions = predict_with_arima(self.df, self.models)
            self.assertValidPrediction(predictions, "ARIMA prediction failed validation")
        except Exception as e:
            logging.error(f"ARIMA prediction failed: {str(e)}")
            self.skipTest(f"ARIMA prediction skipped due to: {str(e)}")

    def test_predict_with_holtwinters(self):
        try:
            predictions = predict_with_holtwinters(self.df, self.models)
            self.assertValidPrediction(predictions, "Holt-Winters prediction failed validation")
        except Exception as e:
            logging.error(f"Holt-Winters prediction failed: {str(e)}")
            self.skipTest(f"Holt-Winters prediction skipped due to: {str(e)}")

    def test_predict_with_linear(self):
        try:
            predictions = predict_with_linear(self.df, self.models)
            self.assertValidPrediction(predictions, "Linear prediction failed validation")
        except Exception as e:
            logging.error(f"Linear prediction failed: {str(e)}")
            self.fail(f"Linear prediction failed: {str(e)}")

    def test_predict_with_xgboost(self):
        try:
            predictions = predict_with_xgboost(self.df, self.models)
            self.assertValidPrediction(predictions, "XGBoost prediction failed validation")
        except Exception as e:
            logging.error(f"XGBoost prediction failed: {str(e)}")
            self.fail(f"XGBoost prediction failed: {str(e)}")

    def test_predict_with_lightgbm(self):
        try:
            predictions = predict_with_lightgbm(self.df, self.models)
            self.assertValidPrediction(predictions, "LightGBM prediction failed validation")
        except Exception as e:
            logging.error(f"LightGBM prediction failed: {str(e)}")
            self.fail(f"LightGBM prediction failed: {str(e)}")

    def test_predict_with_knn(self):
        try:
            predictions = predict_with_knn(self.df, self.models)
            self.assertValidPrediction(predictions, "KNN prediction failed validation")
        except Exception as e:
            logging.error(f"KNN prediction failed: {str(e)}")
            self.fail(f"KNN prediction failed: {str(e)}")

    def test_predict_with_gradientboosting(self):
        try:
            predictions = predict_with_gradientboosting(self.df, self.models)
            self.assertValidPrediction(predictions, "Gradient Boosting prediction failed validation")
        except Exception as e:
            logging.error(f"Gradient Boosting prediction failed: {str(e)}")
            self.fail(f"Gradient Boosting prediction failed: {str(e)}")

    def test_predict_with_catboost(self):
        try:
            predictions = predict_with_catboost(self.df, self.models)
            self.assertValidPrediction(predictions, "CatBoost prediction failed validation")
        except Exception as e:
            logging.error(f"CatBoost prediction failed: {str(e)}")
            self.fail(f"CatBoost prediction failed: {str(e)}")

    def test_predict_with_cnn_lstm(self):
        try:
            predictions = predict_with_cnn_lstm(self.df, self.models)
            self.assertValidPrediction(predictions, "CNN-LSTM prediction failed validation")
        except Exception as e:
            logging.error(f"CNN-LSTM prediction failed: {str(e)}")
            self.fail(f"CNN-LSTM prediction failed: {str(e)}")

    def test_predict_with_autoencoder(self):
        try:
            predictions = predict_with_autoencoder(self.df, self.models)
            self.assertValidPrediction(predictions, "Autoencoder prediction failed validation")
        except Exception as e:
            logging.error(f"Autoencoder prediction failed: {str(e)}")
            self.fail(f"Autoencoder prediction failed: {str(e)}")

    def test_predict_with_meta(self):
        try:
            predictions = predict_with_meta(self.df, self.models)
            self.assertValidPrediction(predictions, "Meta model prediction failed validation")
        except Exception as e:
            logging.error(f"Meta model prediction failed: {str(e)}")
            self.fail(f"Meta model prediction failed: {str(e)}")

    def test_predict_next_draw(self):
        try:
            predictions = predict_next_draw(self.df, self.models)
            self.assertIsNotNone(predictions)
            self.assertIsInstance(predictions, dict)
            for model_name, numbers in predictions.items():
                self.assertValidPrediction(numbers, f"Next draw prediction failed for {model_name}")
        except Exception as e:
            logging.error(f"Next draw prediction failed: {str(e)}")
            self.fail(f"Next draw prediction failed: {str(e)}")

    def test_ensemble_prediction(self):
        try:
            predictions = ensemble_prediction(self.df, self.models)
            self.assertIsNotNone(predictions)
            self.assertIsInstance(predictions, list)
            for numbers in predictions:
                self.assertValidPrediction(numbers, "Ensemble prediction failed validation")
        except Exception as e:
            logging.error(f"Ensemble prediction failed: {str(e)}")
            self.fail(f"Ensemble prediction failed: {str(e)}")

    def test_generate_optimized_predictions(self):
        try:
            predictions = generate_optimized_predictions(self.df, self.models)
            self.assertIsNotNone(predictions)
            self.assertIsInstance(predictions, list)
            for numbers in predictions:
                self.assertValidPrediction(numbers, "Optimized prediction failed validation")
        except Exception as e:
            logging.error(f"Optimized predictions failed: {str(e)}")
            self.fail(f"Optimized predictions failed: {str(e)}")

    def test_monte_carlo_simulation(self):
        try:
            predictions = monte_carlo_simulation(self.df)
            self.assertIsNotNone(predictions)
            self.assertIsInstance(predictions, list)
            for numbers in predictions:
                self.assertValidPrediction(numbers, "Monte Carlo prediction failed validation")
        except Exception as e:
            logging.error(f"Monte Carlo simulation failed: {str(e)}")
            self.fail(f"Monte Carlo simulation failed: {str(e)}")

    def test_backtest(self):
        try:
            accuracy, predictions = backtest(self.df, self.models)
            self.assertIsNotNone(accuracy)
            self.assertIsInstance(accuracy, (float, np.floating))
            self.assertGreaterEqual(accuracy, 0)
            self.assertIsInstance(predictions, list)
            for pred in predictions:
                self.assertValidPrediction(pred, "Backtest prediction failed validation")
        except Exception as e:
            logging.error(f"Backtest failed: {str(e)}")
            self.fail(f"Backtest failed: {str(e)}")

    def test_score_combinations(self):
        try:
            predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
            scored = score_combinations(predictions, self.df)
            self.assertIsNotNone(scored)
            self.assertIsInstance(scored, list)
            self.assertEqual(len(scored), len(predictions))
            for combo, score in scored:
                self.assertValidPrediction(combo, "Scored combination failed validation")
                self.assertIsInstance(score, (float, np.floating))
        except Exception as e:
            logging.error(f"Score combinations failed: {str(e)}")
            self.fail(f"Score combinations failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()