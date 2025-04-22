import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Import modules to test
from analyze_data import (
    analyze_lottery_data,
    get_prediction_weights,
    analyze_randomness,
    analyze_patterns,
    analyze_correlations,
    find_consecutive_pairs,
    get_hot_cold_numbers,
    analyze_spatial_distribution,
    analyze_range_frequency
)

class TestAnalyzeFunctions(unittest.TestCase):
    """Test the lottery data analysis functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'Draw Date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
            'Main_Numbers': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
            'Bonus': [7, 13, 19]
        })
        
        # Larger sample for frequency testing
        self.extended_data = pd.DataFrame({
            'Draw Date': [datetime(2023, 1, i) for i in range(1, 11)],
            'Main_Numbers': [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [1, 13, 15, 17, 19, 21],
                [2, 14, 16, 18, 20, 22],
                [1, 3, 5, 7, 9, 11],
                [2, 4, 6, 8, 10, 12],
                [1, 5, 10, 15, 20, 25],
                [2, 7, 12, 17, 22, 27],
                [3, 9, 15, 21, 27, 33],
                [4, 11, 18, 25, 32, 39]
            ],
            'Bonus': [7, 13, 19, 23, 13, 14, 30, 32, 39, 45]
        })
        
    def test_analyze_lottery_data(self):
        """Test analyzing lottery data."""
        stats = analyze_lottery_data(self.sample_data, cache_results=False)
        
        # Basic checks
        self.assertEqual(stats['total_draws'], 3)
        self.assertEqual(stats['date_range']['start'], '2023-01-01')
        self.assertEqual(stats['date_range']['end'], '2023-01-03')
        
        # Content checks
        self.assertIn('number_statistics', stats)
        self.assertIn('number_frequencies', stats)
        self.assertIn('hot_numbers', stats)
        self.assertIn('cold_numbers', stats)
        self.assertIn('patterns', stats)
        self.assertIn('correlations', stats)
        self.assertIn('randomness_tests', stats)
        self.assertIn('spatial_distribution', stats)
        self.assertIn('range_frequency', stats)
        
    def test_get_prediction_weights(self):
        """Test generating prediction weights."""
        weights = get_prediction_weights(self.extended_data, recent_draws=5)
        
        # Check structure
        self.assertIn('number_weights', weights)
        self.assertIn('pair_weights', weights)
        self.assertIn('pattern_weights', weights)
        
        # Check number weights
        self.assertIsInstance(weights['number_weights'], dict)
        self.assertGreaterEqual(len(weights['number_weights']), 39)
        
        # Check pair weights
        self.assertIsInstance(weights['pair_weights'], dict)
        
        # Check pattern weights
        self.assertIn('consecutive_pairs', weights['pattern_weights'])
        self.assertIn('even_odd_ratio', weights['pattern_weights'])
        self.assertIn('sum_ranges', weights['pattern_weights'])
        
    def test_analyze_randomness(self):
        """Test randomness analysis."""
        all_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        randomness = analyze_randomness(all_numbers)
        
        # Check structure
        self.assertIn('chi_square', randomness)
        self.assertIn('distribution_test', randomness)
        self.assertIn('runs_test', randomness)
        self.assertIn('autocorrelation', randomness)
        
        # Check chi-square test
        self.assertIn('statistic', randomness['chi_square'])
        self.assertIn('p_value', randomness['chi_square'])
        
        # Check autocorrelation is a float
        self.assertIsInstance(randomness['autocorrelation'], float)
        
    def test_analyze_patterns(self):
        """Test analyzing patterns."""
        patterns = analyze_patterns(self.sample_data, recent_window=2)
        
        # Check structure
        self.assertIn('consecutive_pairs', patterns)
        self.assertIn('even_odd_ratio', patterns)
        self.assertIn('prime_numbers', patterns)
        self.assertIn('sum_ranges', patterns)
        self.assertIn('number_gaps', patterns)
        self.assertIn('pair_frequencies', patterns)
        
        # Check consecutive_pairs is a float
        self.assertIsInstance(patterns['consecutive_pairs'], float)
        
    def test_find_consecutive_pairs(self):
        """Test consecutive pairs function."""
        self.assertEqual(find_consecutive_pairs([1, 2, 3, 4, 5, 6]), 5)
        self.assertEqual(find_consecutive_pairs([1, 2, 4, 5, 10, 20]), 2)
        self.assertEqual(find_consecutive_pairs([1, 3, 5, 7, 9, 11]), 0)
        
    def test_get_hot_cold_numbers(self):
        """Test hot and cold numbers identification."""
        hot, cold = get_hot_cold_numbers(self.extended_data, recent_window=5, 
                                        hot_threshold=0.4, cold_threshold=0.1)
        
        # Check types
        self.assertIsInstance(hot, list)
        self.assertIsInstance(cold, list)
        
        # Number 1 should be hot (appears 4 times in 10 draws)
        self.assertIn(1, hot)
        
    def test_analyze_spatial_distribution(self):
        """Test spatial distribution analysis."""
        all_numbers = np.array([1, 2, 3, 4, 5, 6, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52])
        distribution = analyze_spatial_distribution(all_numbers)
        
        # Check structure
        self.assertIn('distribution', distribution)
        self.assertIn('bins', distribution)
        
        # Check length of distribution
        self.assertEqual(len(distribution['distribution']), 6)
        
    def test_analyze_range_frequency(self):
        """Test range frequency analysis."""
        frequency = analyze_range_frequency(self.sample_data)
        
        # Check structure
        self.assertIn('low', frequency)
        self.assertIn('medium', frequency)
        self.assertIn('high', frequency)
        
        # For test data, we should get 6 for each range
        self.assertEqual(frequency['low'], 6)
        self.assertEqual(frequency['medium'], 6)
        self.assertEqual(frequency['high'], 6)

if __name__ == '__main__':
    unittest.main() 