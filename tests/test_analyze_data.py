import unittest
import numpy as np
import pandas as pd
from scripts.analyze_data import analyze_lottery_data
from itertools import combinations
import logging
import pytest
from datetime import datetime
from pathlib import Path
import json
import os
import sys
from unittest.mock import patch, MagicMock, mock_open

# Configure logging for test output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path to resolve imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

try:
    from scripts.analyze_data import (
        analyze_patterns,
        analyze_correlations,
        test_randomness,
        is_prime,
        analyze_spatial_distribution,
        analyze_range_frequency,
        get_prediction_weights,
        get_hot_cold_numbers,
        find_consecutive_pairs
    )
except ImportError:
    try:
        # Try direct imports without 'scripts.' prefix
        from analyze_data import (
            analyze_patterns,
            analyze_correlations,
            test_randomness,
            is_prime,
            analyze_spatial_distribution,
            analyze_range_frequency,
            get_prediction_weights,
            get_hot_cold_numbers,
            find_consecutive_pairs
        )
    except ImportError:
        pytest.skip("Could not import analyze_data module. Skipping tests.", allow_module_level=True)

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        # Create synthetic lottery data matching real data (numbers 1–59)
        n_samples = 100  # Reduced for faster testing
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.df = pd.DataFrame({
            'Draw Date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
            'Day': [days[i % 7] for i in range(n_samples)],
            'Balls': [f"{' '.join(str(x).zfill(2) for x in np.random.choice(range(1, 60), size=6, replace=False))} BONUS {str(np.random.randint(1, 60)).zfill(2)}" for _ in range(n_samples)],
            'Jackpot': [f"£{np.random.randint(1000000, 10000000):,}" for _ in range(n_samples)],
            'Winners': np.random.randint(0, 5, size=n_samples),
            'Draw Details': ['draw details'] * n_samples
        })
        
        # Parse balls column
        def parse_balls(balls_str):
            parts = balls_str.split(' BONUS ')
            main_numbers = [int(x) for x in parts[0].split()]
            bonus = int(parts[1])
            return pd.Series({'Main_Numbers': main_numbers, 'Bonus': bonus})
            
        parsed = self.df['Balls'].apply(parse_balls)
        self.df['Main_Numbers'] = parsed['Main_Numbers']
        self.df['Bonus'] = parsed['Bonus']
        
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
        
    def test_analyze_lottery_data(self):
        try:
            analysis = analyze_lottery_data(self.df)
            self.assertIsNotNone(analysis, "Analysis result is None")
            self.assertIsInstance(analysis, dict, "Analysis result is not a dictionary")
            
            # Check required keys
            required_keys = [
                'total_draws', 'date_range', 'number_statistics',
                'number_frequencies', 'hot_numbers', 'cold_numbers',
                'patterns', 'correlations', 'randomness_tests',
                'spatial_distribution', 'range_frequency'
            ]
            for key in required_keys:
                self.assertIn(key, analysis, f"Missing key: {key}")
            
            # Validate basic properties
            self.assertEqual(analysis['total_draws'], len(self.df), "Total draws mismatch")
            self.assertIsInstance(analysis['date_range'], dict, "Date range is not a dictionary")
            self.assertIn('start', analysis['date_range'], "Missing start date")
            self.assertIn('end', analysis['date_range'], "Missing end date")
            
            # Validate number statistics
            self.assertIsInstance(analysis['number_statistics'], dict, "Number statistics is not a dictionary")
            self.assertIn('mean', analysis['number_statistics'], "Missing mean in number statistics")
            self.assertTrue(1 <= analysis['number_statistics']['mean'] <= 59, "Mean out of range")
            
            # Validate hot and cold numbers
            self.assertIsInstance(analysis['hot_numbers'], list, "Hot numbers is not a list")
            self.assertIsInstance(analysis['cold_numbers'], list, "Cold numbers is not a list")
            self.assertTrue(all(1 <= n <= 59 for n in analysis['hot_numbers']), "Hot numbers out of range")
            self.assertTrue(all(1 <= n <= 59 for n in analysis['cold_numbers']), "Cold numbers out of range")
            
            # Validate patterns
            self.assertIsInstance(analysis['patterns'], dict, "Patterns is not a dictionary")
            self.assertIn('consecutive_pairs', analysis['patterns'], "Missing consecutive_pairs in patterns")
            
            # Validate randomness tests
            self.assertIsInstance(analysis['randomness_tests'], dict, "Randomness tests is not a dictionary")
            self.assertIn('chi_square', analysis['randomness_tests'], "Missing chi_square in randomness tests")
            
        except Exception as e:
            logging.error(f"Test analyze_lottery_data failed: {str(e)}")
            self.fail(f"Test analyze_lottery_data failed: {str(e)}")

    def test_analyze_lottery_data_empty(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Draw Date', 'Day', 'Balls', 'Jackpot', 'Winners', 'Draw Details'])
        with self.assertRaises(Exception):
            analyze_lottery_data(empty_df)

    def test_analyze_lottery_data_invalid_balls(self):
        """Test handling of invalid Balls data."""
        invalid_df = self.df.copy()
        logging.debug("Original Balls value: %s", invalid_df.loc[0, 'Balls'])
        logging.debug("DataFrame columns: %s", invalid_df.columns.tolist())
        # Drop Main_Numbers column to force validation of Balls column
        invalid_df = invalid_df.drop(columns=['Main_Numbers'])
        logging.debug("Main_Numbers present after drop: %s", 'Main_Numbers' in invalid_df.columns)
        invalid_df.loc[0, 'Balls'] = "invalid data"
        logging.debug("Modified Balls value: %s", invalid_df.loc[0, 'Balls'])
        with self.assertRaises(ValueError):
            analyze_lottery_data(invalid_df)

    def test_analyze_lottery_data_missing_columns(self):
        """Test handling of missing Main_Numbers column."""
        invalid_df = self.df.drop(columns=['Main_Numbers'])
        try:
            analysis = analyze_lottery_data(invalid_df)
            self.assertIsNotNone(analysis, "Analysis result is None for missing Main_Numbers")
        except Exception as e:
            logging.error(f"Test missing columns failed: {str(e)}")
            self.fail(f"Test missing columns failed: {str(e)}")

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        'Draw Date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        'Main_Numbers': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
        'Bonus': [7, 13, 19]
    }
    return pd.DataFrame(data)

@pytest.fixture
def extended_sample_data():
    """Fixture to create a larger sample with repeated numbers for frequency testing."""
    data = {
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
    }
    return pd.DataFrame(data)

@pytest.fixture
def cache_file(tmp_path):
    """Fixture to create a sample cache file."""
    cache_path = tmp_path / 'analysis_cache.json'
    cache_data = {
        'stats': {
            'total_draws': 3,
            'date_range': {'start': '2023-01-01', 'end': '2023-01-03'},
            'number_frequencies': {i: 1 for i in range(1, 19)},
            'hot_numbers': [1, 2, 3, 4, 5, 6],
            'cold_numbers': [13, 14, 15, 16, 17, 18]
        },
        'data_hash': 'test_hash_value',
        'creation_time': '2023-01-04T00:00:00'
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    return cache_path

def test_analyze_lottery_data(sample_data):
    """Test analyzing lottery data."""
    with patch('scripts.analyze_data.calculate_data_hash', return_value='test_hash'):
        stats = analyze_lottery_data(sample_data, recent_window=2, cache_results=False)
    
    # Basic checks
    assert isinstance(stats, dict)
    assert stats['total_draws'] == 3
    assert stats['date_range']['start'] == '2023-01-01'
    assert stats['date_range']['end'] == '2023-01-03'
    
    # Content checks
    assert 'number_statistics' in stats
    assert 'number_frequencies' in stats
    assert 'hot_numbers' in stats
    assert 'cold_numbers' in stats
    assert 'patterns' in stats
    assert 'correlations' in stats
    assert 'randomness_tests' in stats
    assert 'spatial_distribution' in stats
    assert 'range_frequency' in stats
    
    # Data checks
    assert len(stats['number_frequencies']) == 18  # All numbers from 1-18
    assert all(freq == 1 for freq in stats['number_frequencies'].values())
    assert len(stats['hot_numbers']) > 0
    assert len(stats['cold_numbers']) > 0
    assert isinstance(stats['patterns'], dict)
    assert isinstance(stats['correlations'], dict)
    assert isinstance(stats['randomness_tests'], dict)

def test_analyze_lottery_data_with_cache(sample_data, cache_file):
    """Test analyzing lottery data with valid cache."""
    with patch('scripts.analyze_data.calculate_data_hash', return_value='test_hash_value'):
        with patch('scripts.analyze_data.Path.exists', return_value=True):
            m = mock_open(read_data=open(cache_file).read())
            with patch('builtins.open', m):
                stats = analyze_lottery_data(sample_data, cache_file=str(cache_file), cache_results=True)
    
    # Verify we got cached data
    assert stats['total_draws'] == 3
    assert stats['number_frequencies'] == {i: 1 for i in range(1, 19)}
    assert stats['hot_numbers'] == [1, 2, 3, 4, 5, 6]
    assert stats['cold_numbers'] == [13, 14, 15, 16, 17, 18]

def test_analyze_lottery_data_invalid_cache(sample_data, tmp_path):
    """Test analyzing data with invalid cache."""
    invalid_cache = tmp_path / 'invalid_cache.json'
    with open(invalid_cache, 'w') as f:
        f.write("This is not valid JSON")
    
    with patch('scripts.analyze_data.calculate_data_hash', return_value='test_hash'):
        with patch('scripts.analyze_data.Path.exists', return_value=True):
            stats = analyze_lottery_data(sample_data, cache_file=str(invalid_cache), cache_results=True)
    
    # Should generate new stats despite cache file existing
    assert stats['total_draws'] == 3
    assert len(stats['number_frequencies']) == 18

def test_analyze_lottery_data_missing_column():
    """Test analyzing data with missing Main_Numbers."""
    df = pd.DataFrame({'Draw Date': [datetime(2023, 1, 1)]})
    
    with pytest.raises(ValueError, match="DataFrame must contain 'Main_Numbers'"):
        analyze_lottery_data(df)

def test_analyze_patterns(sample_data):
    """Test analyzing number patterns."""
    patterns = analyze_patterns(sample_data, recent_window=2)
    
    # Check structure
    assert 'consecutive_pairs' in patterns
    assert 'even_odd_ratio' in patterns
    assert 'prime_numbers' in patterns
    assert 'sum_ranges' in patterns
    assert 'number_gaps' in patterns
    assert 'pair_frequencies' in patterns
    assert 'triplet_frequencies' in patterns
    
    # Check values
    assert patterns['consecutive_pairs'] >= 0  # Should be some consecutive pairs
    assert isinstance(patterns['even_odd_ratio'], list)
    assert len(patterns['even_odd_ratio']) == 3  # One for each draw
    assert patterns['sum_ranges'] == [21, 57, 93]  # Sum of each set of 6 numbers
    
    # Check pair frequencies
    assert isinstance(patterns['pair_frequencies'], dict)
    assert len(patterns['pair_frequencies']) > 0

def test_find_consecutive_pairs():
    """Test finding consecutive pairs in a list of numbers."""
    # Test with consecutive numbers
    assert find_consecutive_pairs([1, 2, 3, 4, 5, 6]) == 5
    
    # Test with some consecutive numbers
    assert find_consecutive_pairs([1, 2, 4, 5, 10, 20]) == 2
    
    # Test with no consecutive numbers
    assert find_consecutive_pairs([1, 3, 5, 7, 9, 11]) == 0

def test_analyze_correlations(sample_data):
    """Test analyzing correlations."""
    correlations = analyze_correlations(sample_data)
    
    # Check structure
    assert 'number_correlations' in correlations
    assert 'sum_correlations' in correlations
    assert 'pattern_correlations' in correlations
    
    # Check values
    assert isinstance(correlations['number_correlations'], dict)
    assert 'mean' in correlations['sum_correlations']
    assert 'std' in correlations['sum_correlations']
    assert correlations['sum_correlations']['mean'] == 57.0  # (21 + 57 + 93) / 3
    
    # Check pattern correlations
    assert 'consecutive_pairs' in correlations['pattern_correlations']
    assert 'even_odd_ratio' in correlations['pattern_correlations']

def test_test_randomness():
    """Test randomness tests."""
    all_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    randomness = test_randomness(all_numbers)
    
    # Check structure
    assert 'chi_square' in randomness
    assert 'distribution_test' in randomness
    assert 'runs_test' in randomness
    assert 'autocorrelation' in randomness
    
    # Check chi-square test
    assert 'statistic' in randomness['chi_square']
    assert 'p_value' in randomness['chi_square']
    assert randomness['chi_square']['statistic'] >= 0
    
    # Check distribution test
    assert 'shapiro_wilk' in randomness['distribution_test']
    assert 'statistic' in randomness['distribution_test']['shapiro_wilk']
    assert 'p_value' in randomness['distribution_test']['shapiro_wilk']
    
    # Check autocorrelation
    assert isinstance(randomness['autocorrelation'], float)

def test_is_prime():
    """Test prime number checker."""
    # Basic primes
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(5) is True
    assert is_prime(7) is True
    assert is_prime(11) is True
    assert is_prime(13) is True
    assert is_prime(17) is True
    assert is_prime(19) is True
    
    # Non-primes
    assert is_prime(1) is False
    assert is_prime(4) is False
    assert is_prime(6) is False
    assert is_prime(8) is False
    assert is_prime(9) is False
    assert is_prime(10) is False
    assert is_prime(12) is False
    
    # Edge cases
    assert is_prime(0) is False
    assert is_prime(-1) is False
    assert is_prime(-2) is False

def test_get_hot_cold_numbers(extended_sample_data):
    """Test identifying hot and cold numbers."""
    hot, cold = get_hot_cold_numbers(extended_sample_data, recent_window=5, hot_threshold=0.4, cold_threshold=0.1)
    
    # Verify types
    assert isinstance(hot, list)
    assert isinstance(cold, list)
    
    # Verify content
    assert len(hot) > 0  # Should have some hot numbers
    assert all(isinstance(n, int) for n in hot)
    assert all(1 <= n <= 59 for n in hot)
    
    # Number 1 should be hot (appears 4 times in 10 draws)
    assert 1 in hot
    
    # Numbers > 39 should be cold (appear 0-1 times)
    assert all(n > 39 for n in extended_sample_data['Main_Numbers'].iloc[-1] if n not in [1, 2, 3, 4, 5])

def test_analyze_spatial_distribution():
    """Test analyzing spatial distribution."""
    all_numbers = np.array([1, 2, 3, 4, 5, 6, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52])
    distribution = analyze_spatial_distribution(all_numbers)
    
    # Check structure
    assert 'distribution' in distribution
    assert 'bins' in distribution
    
    # Check values
    assert len(distribution['distribution']) == 6  # Bins: 1-10, 11-20, 21-30, 31-40, 41-50, 51-59
    assert distribution['distribution'][0] == 6  # 1, 2, 3, 4, 5, 6 in bin 1-10
    assert distribution['distribution'][1] == 2  # 11, 12 in bin 11-20
    assert distribution['distribution'][2] == 2  # 21, 22 in bin 21-30
    assert distribution['distribution'][3] == 2  # 31, 32 in bin 31-40
    assert distribution['distribution'][4] == 2  # 41, 42 in bin 41-50
    assert distribution['distribution'][5] == 2  # 51, 52 in bin 51-59

def test_analyze_range_frequency(sample_data):
    """Test analyzing range frequency."""
    frequency = analyze_range_frequency(sample_data)
    
    # Check structure
    assert 'low' in frequency
    assert 'medium' in frequency
    assert 'high' in frequency
    
    # Check values
    assert frequency['low'] == 6  # Numbers 1-20
    assert frequency['medium'] == 6  # Numbers 21-40
    assert frequency['high'] == 6  # Numbers 41-59

def test_get_prediction_weights(extended_sample_data):
    """Test generating prediction weights."""
    weights = get_prediction_weights(extended_sample_data, recent_draws=5)
    
    # Check structure
    assert 'number_weights' in weights
    assert 'pair_weights' in weights
    assert 'pattern_weights' in weights
    
    # Check number weights
    assert isinstance(weights['number_weights'], dict)
    assert len(weights['number_weights']) >= 39  # At least all numbers from sample data
    assert all(1 <= int(n) <= 59 for n in weights['number_weights'].keys())
    assert all(isinstance(w, float) for w in weights['number_weights'].values())
    
    # Check pair weights
    assert isinstance(weights['pair_weights'], dict)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in weights['pair_weights'].keys())
    
    # Check pattern weights
    assert 'consecutive_pairs' in weights['pattern_weights']
    assert 'even_odd_ratio' in weights['pattern_weights']
    assert 'sum_ranges' in weights['pattern_weights']
    assert 'hot_numbers' in weights['pattern_weights']
    assert 'cold_numbers' in weights['pattern_weights']
    
    # Since number 1 appears frequently, it should have a higher weight
    number_1_weight = weights['number_weights'].get('1', 0)
    assert number_1_weight > 0

@patch('scripts.analyze_data.analyze_lottery_data')
def test_get_prediction_weights_error(mock_analyze, sample_data):
    """Test get_prediction_weights with analysis error."""
    mock_analyze.side_effect = Exception("Analysis error")
    
    with pytest.raises(Exception, match="Analysis error"):
        get_prediction_weights(sample_data)
    
    mock_analyze.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])