import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import json
from models.feature_engineering import (
    load_feature_config, 
    create_temporal_features,
    create_number_features,
    create_frequency_features,
    create_lag_features,
    create_combination_features,
    engineer_features
)

class TestFeatureEngineering(unittest.TestCase):
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
            'Bonus_Number': bonus_numbers
        })
        
        # Mock feature config
        self.mock_config = {
            "feature_engineering": {
                "temporal_features": {"enabled": True},
                "frequency_features": {
                    "enabled": True,
                    "rolling_windows": [5, 10, 20],
                    "number_combinations": {"enabled": True}
                }
            }
        }
    
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"feature_engineering": {"test": True}}))
    def test_load_feature_config(self, mock_file):
        # Test successful config loading
        result = load_feature_config("mock_path.json")
        self.assertEqual(result, {"test": True})
        mock_file.assert_called_once_with("mock_path.json", 'r')
        
        # Test error handling
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = load_feature_config("nonexistent_path.json")
            self.assertEqual(result, {})
    
    def test_create_temporal_features(self):
        result = create_temporal_features(self.df)
        
        # Check that original DataFrame was not modified
        self.assertNotIn('Year', self.df.columns)
        
        # Check that new features were created
        self.assertIn('Year', result.columns)
        self.assertIn('Month', result.columns)
        self.assertIn('DayOfWeek', result.columns)
        self.assertIn('Month_sin', result.columns)
        self.assertIn('Month_cos', result.columns)
        
        # Check values
        self.assertEqual(result['Year'].iloc[0], 2020)
        self.assertEqual(result['Month'].iloc[0], 1)  # January
        self.assertTrue(all(-1 <= result['Month_sin']) and all(result['Month_sin'] <= 1))
        self.assertTrue(all(-1 <= result['Month_cos']) and all(result['Month_cos'] <= 1))
    
    def test_create_number_features(self):
        result = create_number_features(self.df)
        
        # Check that new features were created
        self.assertIn('Sum', result.columns)
        self.assertIn('Mean', result.columns)
        self.assertIn('Std', result.columns)
        self.assertIn('Primes', result.columns)
        self.assertIn('Odds', result.columns)
        self.assertIn('Evens', result.columns)
        self.assertIn('Gaps', result.columns)
        
        # Check value ranges
        # Sum should be between 21 (1+2+3+4+5+6) and 324 (54+55+56+57+58+59)
        self.assertTrue(all(result['Sum'] >= 21) and all(result['Sum'] <= 324))
        
        # Mean should be between 3.5 and 54
        self.assertTrue(all(result['Mean'] >= 3.5) and all(result['Mean'] <= 54))
        
        # Odds + Evens should equal 6
        self.assertTrue(all(result['Odds'] + result['Evens'] == 6))
        
        # Decade features should exist
        self.assertIn('Decade_1_10', result.columns)
        self.assertIn('Decade_51_60', result.columns)
    
    def test_create_frequency_features(self):
        # Only test with smaller windows to avoid issues with test data size
        windows = [5, 10]
        result = create_frequency_features(self.df, windows=windows)
        
        # Check that new features were created
        self.assertIn('Number_Frequency', result.columns)
        self.assertIn('Number_Frequency_Std', result.columns)
        
        # Check that rolling window features were created for each window
        for window in windows:
            self.assertIn(f'Sum_MA_{window}', result.columns)
            self.assertIn(f'Sum_STD_{window}', result.columns)
            self.assertIn(f'Mean_MA_{window}', result.columns)
            self.assertIn(f'Mean_STD_{window}', result.columns)
            
            # Only check Hotness column for windows that can be created
            if window < len(self.df):
                self.assertIn(f'Hotness_{window}', result.columns)
    
    def test_create_lag_features(self):
        target_cols = ['Main_Numbers', 'Bonus_Number']
        lag_periods = [1, 2, 3]
        result = create_lag_features(self.df, target_cols, lag_periods)
        
        # Check that lag features were created
        for col in target_cols:
            for lag in lag_periods:
                lag_col = f'{col}_lag_{lag}'
                self.assertIn(lag_col, result.columns)
                
                # Check that lag values are correct
                for i in range(lag, len(result)):
                    np.testing.assert_equal(
                        result[lag_col].iloc[i],
                        self.df[col].iloc[i-lag]
                    )
    
    def test_create_combination_features(self):
        result = create_combination_features(self.df)
        
        # Check that pair features were created
        pair_columns = [col for col in result.columns if col.startswith('Pair_')]
        self.assertTrue(len(pair_columns) > 0)
        
        # Check that values are binary (0 or 1)
        for col in pair_columns:
            self.assertTrue(all(result[col].isin([0, 1])))
    
    @patch("models.feature_engineering.load_feature_config")
    def test_engineer_features(self, mock_load_config):
        # Mock the config loading
        mock_load_config.return_value = self.mock_config["feature_engineering"]
        
        result = engineer_features(self.df, config_path="dummy_path.json")
        
        # Check that config was loaded
        mock_load_config.assert_called_once_with("dummy_path.json")
        
        # Check that result includes features from all feature engineering steps
        # Temporal features
        self.assertIn('Year', result.columns)
        self.assertIn('Month', result.columns)
        
        # Number features
        self.assertIn('Sum', result.columns)
        self.assertIn('Mean', result.columns)
        
        # Frequency features
        self.assertIn('Number_Frequency', result.columns)
        self.assertIn('Sum_MA_5', result.columns)
        
        # Lag features
        self.assertIn('Sum_lag_1', result.columns)
        self.assertIn('Mean_lag_1', result.columns)
        
        # Combination features
        pair_columns = [col for col in result.columns if col.startswith('Pair_')]
        self.assertTrue(len(pair_columns) > 0)

if __name__ == '__main__':
    unittest.main() 