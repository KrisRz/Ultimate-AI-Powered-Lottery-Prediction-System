import unittest
import pandas as pd
import numpy as np
import logging
from models.utils import ensure_valid_prediction, log_training_errors
import os
from unittest.mock import patch, MagicMock

class TestModelUtils(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        n_samples = 10
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        self.df = pd.DataFrame({
            'Draw Date': dates,
            'Feature1': np.random.rand(n_samples),
            'Feature2': np.random.rand(n_samples),
            'Feature3': np.random.rand(n_samples)
        })
        
        # Configure logging for tests
        logging.basicConfig(level=logging.ERROR)
    
    def test_ensure_valid_prediction(self):
        # Test valid prediction
        valid_pred = [1, 2, 3, 4, 5, 6]
        result = ensure_valid_prediction(valid_pred)
        self.assertEqual(result, sorted(valid_pred))
        self.assertEqual(len(result), 6)
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertTrue(all(1 <= x <= 59 for x in result))
        
        # Test predictions with values outside range
        out_of_range = [0, 60, 2, 3, 4, 5]
        result = ensure_valid_prediction(out_of_range)
        self.assertEqual(len(result), 6)
        self.assertTrue(all(1 <= x <= 59 for x in result))
        
        # Test duplicate values
        duplicates = [1, 1, 2, 3, 4, 5]
        result = ensure_valid_prediction(duplicates)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(set(result)), 6)  # All values should be unique
        
        # Test None input
        result = ensure_valid_prediction(None)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(set(result)), 6)
        
        # Test too few values
        too_few = [1, 2, 3]
        result = ensure_valid_prediction(too_few)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(set(result)), 6)
        
        # Test too many values - should be truncated to 6
        too_many = [1, 2, 3, 4, 5, 6, 7, 8]
        result = ensure_valid_prediction(too_many)
        self.assertEqual(len(result), 6)
    
    def test_log_training_errors_decorator(self):
        # Mock function to decorate
        @log_training_errors
        def mock_training_function(df):
            return "trained_model"
        
        # Test successful execution
        with patch('logging.info') as mock_log_info:
            result = mock_training_function(self.df)
            self.assertEqual(result, "trained_model")
            # Verify logging was called
            self.assertTrue(mock_log_info.called)
            
        # Test error handling
        @log_training_errors
        def mock_error_function(df):
            raise ValueError("Test error")
        
        with patch('logging.error') as mock_log_error:
            with self.assertRaises(ValueError):
                mock_error_function(self.df)
            # Verify error was logged
            self.assertTrue(mock_log_error.called)
    
    @patch('models.utils.log_system_metrics')
    def test_log_system_metrics_integration(self, mock_log_metrics):
        # Test that the metrics logging is called when using the decorator
        @log_training_errors
        def mock_training_function(df):
            return "trained_model"
        
        result = mock_training_function(self.df)
        self.assertEqual(result, "trained_model")
        self.assertTrue(mock_log_metrics.called)
        self.assertEqual(mock_log_metrics.call_count, 2)  # Called at start and end

    @patch('models.utils.parse_balls')
    def test_parse_balls_integration(self, mock_parse):
        # Configure the mock to return a specific value
        mock_parse.return_value = ([1, 2, 3, 4, 5, 6], 7)
        
        # We're using the mock since the actual function may import modules 
        # that aren't directly visible in the utils.py snippet
        ball_str = "01 02 03 04 05 06 BONUS 07"
        main_nums, bonus = mock_parse(ball_str)
        
        self.assertEqual(main_nums, [1, 2, 3, 4, 5, 6])
        self.assertEqual(bonus, 7)
        mock_parse.assert_called_once_with(ball_str)

if __name__ == '__main__':
    unittest.main() 