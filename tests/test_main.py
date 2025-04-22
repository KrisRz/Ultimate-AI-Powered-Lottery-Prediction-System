import unittest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.predict_numbers import predict_next_draw, ensemble_prediction, generate_optimized_predictions
from scripts.fetch_data import load_data
from scripts.train_models import update_models
from itertools import combinations
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestMain(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data (numbers 1–59)
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
        
        # Add required features (aligned with test_models.py)
        self.df['DayOfWeek'] = pd.to_datetime(self.df['Draw Date']).dt.dayofweek.astype(str)  # For CatBoost
        self.df['Month'] = pd.to_datetime(self.df['Draw Date']).dt.month
        self.df['Year'] = pd.to_datetime(self.df['Draw Date']).dt.year
        self.df['Sum'] = self.df['Main_Numbers'].apply(sum)
        self.df['Mean'] = self.df['Main_Numbers'].apply(np.mean)
        self.df['Unique'] = self.df['Main_Numbers'].apply(lambda x: len(set(x)))
        self.df['ZScore_Sum'] = (self.df['Sum'] - self.df['Sum'].rolling(window=10, min_periods=1).mean()) / self.df['Sum'].rolling(window=10, min_periods=1).std()
        self.df['ZScore_Sum'] = self.df['ZScore_Sum'].fillna(0)
        self.df['Primes'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]))
        self.df['Odds'] = self.df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 1))
        self.df['Gaps'] = self.df['Main_Numbers'].apply(lambda x: np.mean([x[i+1] - x[i] for i in range(len(x)-1)]))
        
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
        
        # Fill any remaining NaN values
        self.df = self.df.fillna(0)
        
        # Process data and train models
        try:
            # Save synthetic data to a temporary CSV for load_data
            temp_file = Path("temp_test_data.csv")
            self.df.to_csv(temp_file, index=False)
            self.df = load_data(temp_file)  # Use predict_numbers.load_data
            self.models = update_models(self.df)
            temp_file.unlink()  # Clean up
        except Exception as e:
            logging.error(f"Error in setUp: {str(e)}")
            self.fail(f"setUp failed: {str(e)}")

    def assertValidPrediction(self, numbers, msg):
        """Validate that predictions are 6 unique integers between 1 and 59."""
        self.assertIsNotNone(numbers, f"Prediction is None: {msg}")
        self.assertIsInstance(numbers, (list, np.ndarray), f"Prediction is not a list/array: {msg}")
        self.assertEqual(len(numbers), 6, f"Prediction length is not 6 (got {len(numbers)}): {msg}")
        self.assertTrue(all(isinstance(n, (int, np.integer)) for n in numbers), f"Not all predictions are integers: {msg}")
        self.assertTrue(all(1 <= n <= 59 for n in numbers), f"Not all predictions are in range [1, 59]: {msg}")
        self.assertEqual(len(set(numbers)), 6, f"Predictions are not unique: {msg}")

    def test_predict_next_draw(self):
        try:
            prediction = predict_next_draw(self.df, self.models)
            self.assertIsNotNone(prediction, "Prediction is None")
            self.assertIsInstance(prediction, dict, "Prediction is not a dictionary")
            for model_name, numbers in prediction.items():
                self.assertValidPrediction(numbers, f"Invalid prediction for model {model_name}")
        except Exception as e:
            logging.error(f"Test predict_next_draw failed: {str(e)}")
            self.fail(f"Test predict_next_draw failed: {str(e)}")

    def test_ensemble_prediction(self):
        try:
            predictions = ensemble_prediction(self.df, self.models)
            self.assertIsNotNone(predictions, "Predictions are None")
            self.assertIsInstance(predictions, list, "Predictions are not a list")
            for numbers in predictions:
                self.assertValidPrediction(numbers, "Invalid ensemble prediction")
        except Exception as e:
            logging.error(f"Test ensemble_prediction failed: {str(e)}")
            self.fail(f"Test ensemble_prediction failed: {str(e)}")

    def test_generate_optimized_predictions(self):
        try:
            predictions = generate_optimized_predictions(self.df, self.models)
            self.assertIsNotNone(predictions, "Predictions are None")
            self.assertIsInstance(predictions, list, "Predictions are not a list")
            for numbers in predictions:
                self.assertValidPrediction(numbers, "Invalid optimized prediction")
        except Exception as e:
            logging.error(f"Test generate_optimized_predictions failed: {str(e)}")
            self.fail(f"Test generate_optimized_predictions failed: {str(e)}")

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Draw Date', 'Day', 'Balls', 'Jackpot', 'Winners', 'Draw Details'])
        try:
            prediction = predict_next_draw(empty_df, self.models)
            self.assertIsNotNone(prediction, "Prediction is None for empty DataFrame")
        except Exception as e:
            logging.info(f"Expected error for empty DataFrame: {str(e)}")

    def test_invalid_balls(self):
        """Test handling of invalid Balls data."""
        invalid_df = self.df.copy()
        invalid_df.loc[0, 'Balls'] = "invalid data"
        with self.assertRaises(ValueError):
            predict_next_draw(invalid_df, self.models)

if __name__ == '__main__':
    unittest.main()