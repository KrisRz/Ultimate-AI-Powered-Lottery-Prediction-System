import unittest
import numpy as np
import pandas as pd
from scripts.models.utils import (
    ensure_valid_prediction,
    parse_balls,
    setup_logging,
    log_system_metrics,
    log_memory_usage,
    log_training_errors
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({
            'Draw Date': pd.date_range(start='2020-01-01', periods=10),
            'Balls': ['1 2 3 4 5 6 7', '8 9 10 11 12 13 14', '15 16 17 18 19 20 21'],
            'Main_Numbers': [[1,2,3,4,5,6], [8,9,10,11,12,13], [15,16,17,18,19,20]]
        })

    def test_ensure_valid_prediction(self):
        # Test with valid input
        valid_pred = [1, 2, 3, 4, 5, 6]
        result = ensure_valid_prediction(valid_pred)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(set(result)), 6)
        self.assertTrue(all(1 <= x <= 59 for x in result))

        # Test with invalid input
        invalid_pred = [0, 60, 2, 2, 3, 3]
        result = ensure_valid_prediction(invalid_pred)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(set(result)), 6)
        self.assertTrue(all(1 <= x <= 59 for x in result))

        # Test with None
        result = ensure_valid_prediction(None)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(set(result)), 6)
        self.assertTrue(all(1 <= x <= 59 for x in result))

    def test_parse_balls(self):
        # Test valid input
        main_numbers, bonus = parse_balls('1 2 3 4 5 6 7')
        self.assertEqual(main_numbers, [1, 2, 3, 4, 5, 6])
        self.assertEqual(bonus, 7)

        # Test invalid input
        with self.assertRaises(ValueError):
            parse_balls('invalid input')

        # Test input with missing numbers
        main_numbers, bonus = parse_balls('1 2 3')
        self.assertEqual(len(main_numbers), 6)
        self.assertTrue(all(1 <= x <= 59 for x in main_numbers))
        self.assertTrue(1 <= bonus <= 59)

    @log_training_errors
    def dummy_training_function(self, X, y):
        """Dummy function to test log_training_errors decorator"""
        return np.array([1, 2, 3, 4, 5, 6])

    def test_log_training_errors_decorator(self):
        # Test successful execution
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 6)
        result = self.dummy_training_function(X, y)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 6)

        # Test error handling
        with self.assertRaises(Exception):
            self.dummy_training_function(None, None)

if __name__ == '__main__':
    unittest.main() 