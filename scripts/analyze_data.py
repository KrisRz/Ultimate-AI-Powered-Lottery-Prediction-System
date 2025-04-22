from scipy.stats import chisquare
from collections import Counter
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from scipy import stats
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(filename='lottery.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_lottery_data(df: pd.DataFrame, recent_window: int = 50, cache_results: bool = True) -> Dict:
    """
    Analyze lottery data and return statistics for prediction and scoring.
    
    Args:
        df: DataFrame with 'Main_Numbers' column (list of 6 integers, 1-59).
        recent_window: Number of recent draws to analyze for prediction-relevant stats.
        cache_results: Whether to cache results to avoid recomputation.
    
    Returns:
        Dictionary of statistics including frequencies, patterns, and correlations.
    """
    try:
        start_time = time.time()
        
        # Check for required column
        if 'Main_Numbers' not in df.columns:
            logger.error("Missing 'Main_Numbers' column in DataFrame")
            raise ValueError("DataFrame must contain 'Main_Numbers' column")
        
        # Check cache
        cache_file = Path('results/analysis_cache.json')
        if cache_results and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                if (cached.get('data_size') == len(df) and 
                    cached.get('last_date') == df['Draw Date'].max().strftime('%Y-%m-%d')):
                    logger.info("Returning cached analysis results")
                    return cached['stats']
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Cache read error: {str(e)}. Recomputing analysis.")
        
        # Extract main numbers
        all_numbers = np.concatenate(df['Main_Numbers'].values)
        recent_df = df.iloc[-recent_window:] if len(df) > recent_window else df
        recent_numbers = np.concatenate(recent_df['Main_Numbers'].values)
        
        # Calculate basic statistics
        stats_dict = {
            'total_draws': len(df),
            'date_range': {
                'start': df['Draw Date'].min().strftime('%Y-%m-%d'),
                'end': df['Draw Date'].max().strftime('%Y-%m-%d')
            },
            'number_statistics': {
                'mean': float(np.mean(all_numbers)),
                'median': float(np.median(all_numbers)),
                'std': float(np.std(all_numbers)),
                'min': int(np.min(all_numbers)),
                'max': int(np.max(all_numbers))
            }
        }
        
        # Calculate number frequencies
        number_counts = Counter(all_numbers)
        recent_counts = Counter(recent_numbers)
        stats_dict['number_frequencies'] = {int(k): int(v) for k, v in number_counts.items()}
        stats_dict['recent_frequencies'] = {int(k): int(v) for k, v in recent_counts.items()}
        
        # Calculate hot and cold numbers
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        stats_dict['hot_numbers'] = [int(num) for num, _ in sorted_numbers[:10]]
        stats_dict['cold_numbers'] = [int(num) for num, _ in sorted_numbers[-10:]]
        
        # Calculate recent hot/cold numbers
        sorted_recent = sorted(recent_counts.items(), key=lambda x: x[1], reverse=True)
        stats_dict['recent_hot_numbers'] = [int(num) for num, _ in sorted_recent[:10]]
        stats_dict['recent_cold_numbers'] = [int(num) for num, _ in sorted_recent[-10:]]
        
        # Calculate number probability weights
        total_numbers = len(all_numbers)
        stats_dict['number_weights'] = {int(k): float(v/total_numbers) for k, v in number_counts.items()}
        
        # Calculate patterns
        stats_dict['patterns'] = analyze_patterns(df, recent_window)
        
        # Calculate correlations
        stats_dict['correlations'] = analyze_correlations(df)
        
        # Test randomness
        stats_dict['randomness_tests'] = test_randomness(all_numbers)
        
        # Analyze spatial distribution
        stats_dict['spatial_distribution'] = analyze_spatial_distribution(all_numbers)
        
        # Analyze range frequency
        stats_dict['range_frequency'] = analyze_range_frequency(df)
        
        # Cache results
        if cache_results:
            cache_file.parent.mkdir(exist_ok=True)
            with open(cache_file, 'w') as f:
                cache_data = {
                    'stats': stats_dict, 
                    'data_size': len(df), 
                    'last_date': df['Draw Date'].max().strftime('%Y-%m-%d')
                }
                json.dump(cache_data, f, indent=2)
        
        # Log results
        duration = time.time() - start_time
        logger.info(f"Analysis completed in {duration:.2f} seconds. Total draws: {len(df)}")
        logger.info(f"Hot numbers: {stats_dict['hot_numbers']}")
        logger.info(f"Recent hot numbers (last {recent_window} draws): {stats_dict['recent_hot_numbers']}")
        
        return stats_dict
        
    except Exception as e:
        logger.error(f"Error analyzing lottery data: {e}")
        raise

def analyze_patterns(df: pd.DataFrame, recent_window: int = 50) -> Dict:
    """
    Analyze number patterns in lottery data with vectorized operations.
    
    Args:
        df: DataFrame containing lottery data
        recent_window: Number of recent draws to analyze for pair/triplet frequencies
        
    Returns:
        Dictionary containing pattern analysis
    """
    try:
        patterns = {
            'consecutive_pairs': [],
            'even_odd_ratio': [],
            'prime_numbers': [],
            'sum_ranges': [],
            'number_gaps': [],
            'pair_frequencies': {},
            'triplet_frequencies': {}
        }
        
        # Convert to numpy arrays for vectorized operations
        numbers_array = np.array([list(nums) for nums in df['Main_Numbers']])
        numbers_array.sort(axis=1)  # Ensure sorted numbers for consistent analysis
        
        # Calculate basic patterns
        for numbers in numbers_array:
            # Consecutive pairs
            consecutive = sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1)
            patterns['consecutive_pairs'].append(consecutive)
            
            # Even/odd ratio
            evens = sum(1 for n in numbers if n % 2 == 0)
            odds = len(numbers) - evens
            patterns['even_odd_ratio'].append((evens, odds))
            
            # Prime numbers
            primes = sum(1 for n in numbers if is_prime(n))
            patterns['prime_numbers'].append(primes)
            
            # Sum ranges
            total_sum = sum(numbers)
            patterns['sum_ranges'].append(total_sum)
            
            # Number gaps
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            patterns['number_gaps'].append(gaps)
        
        # Analyze pair and triplet frequencies from recent draws
        recent_draws = df['Main_Numbers'].iloc[-recent_window:] if len(df) > recent_window else df['Main_Numbers']
        
        # Extract all possible pairs and triplets
        all_pairs = []
        all_triplets = []
        
        for numbers in recent_draws:
            # Get all possible pairs
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = (min(numbers[i], numbers[j]), max(numbers[i], numbers[j]))
                    all_pairs.append(pair)
            
            # Get all possible triplets
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    for k in range(j+1, len(numbers)):
                        triplet = (numbers[i], numbers[j], numbers[k])
                        all_triplets.append(tuple(sorted(triplet)))
        
        # Count pair frequencies
        pair_counts = Counter(all_pairs)
        patterns['pair_frequencies'] = {f"{p[0]}_{p[1]}": count for p, count in pair_counts.most_common(30)}
        
        # Count triplet frequencies
        triplet_counts = Counter(all_triplets)
        patterns['triplet_frequencies'] = {f"{t[0]}_{t[1]}_{t[2]}": count 
                                          for t, count in triplet_counts.most_common(20)}
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise

def analyze_correlations(df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between lottery numbers.
    
    Args:
        df: DataFrame containing lottery data
        
    Returns:
        Dictionary containing correlation analysis
    """
    try:
        correlations = {
            'number_correlations': {},
            'sum_correlations': {},
            'pattern_correlations': {}
        }
        
        # Calculate number correlations - only for top pairs to avoid large output
        top_pairs = []
        for i in range(1, 60):
            for j in range(i+1, 60):
                count_i = sum(1 for numbers in df['Main_Numbers'] if i in numbers)
                count_j = sum(1 for numbers in df['Main_Numbers'] if j in numbers)
                count_both = sum(1 for numbers in df['Main_Numbers'] if i in numbers and j in numbers)
                
                if count_both > 0:
                    correlation = count_both / (count_i * count_j / len(df))
                    top_pairs.append((i, j, correlation))
        
        # Only keep top 30 correlated pairs
        top_pairs.sort(key=lambda x: x[2], reverse=True)
        correlations['number_correlations'] = {f"{p[0]}_{p[1]}": p[2] for p in top_pairs[:30]}
        
        # Calculate sum correlations
        sums = df['Main_Numbers'].apply(sum)
        correlations['sum_correlations'] = {
            'mean': float(np.mean(sums)),
            'std': float(np.std(sums)),
            'min': int(np.min(sums)),
            'max': int(np.max(sums))
        }
        
        # Calculate pattern correlations
        patterns = analyze_patterns(df)
        correlations['pattern_correlations'] = {
            'consecutive_pairs': float(np.mean(patterns['consecutive_pairs'])),
            'even_odd_ratio': float(np.mean([e/(e+o) if (e+o) > 0 else 0 for e, o in patterns['even_odd_ratio']])),
            'prime_numbers': float(np.mean(patterns['prime_numbers'])),
            'sum_ranges': {
                'mean': float(np.mean(patterns['sum_ranges'])),
                'std': float(np.std(patterns['sum_ranges']))
            }
        }
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error analyzing correlations: {e}")
        raise

def test_randomness(all_numbers: np.ndarray) -> Dict:
    """
    Test randomness of lottery numbers.
    
    Args:
        all_numbers: Array of all lottery numbers
        
    Returns:
        Dictionary containing randomness test results
    """
    try:
        randomness = {
            'chi_square': {},
            'distribution_test': {},
            'autocorrelation': {}
        }
        
        # Chi-square test
        observed = np.bincount(all_numbers.astype(int), minlength=60)[1:]
        expected = np.ones(59) * len(all_numbers) / 59
        chi2, p = chisquare(observed, expected)
        randomness['chi_square'] = {'statistic': float(chi2), 'p_value': float(p)}
        
        # Distribution test
        shapiro_result = stats.shapiro(all_numbers)
        randomness['distribution_test'] = {
            'shapiro_wilk': {'statistic': float(shapiro_result[0]), 'p_value': float(shapiro_result[1])},
            'anderson_darling': stats.anderson(all_numbers).tolist()
        }
        
        # Autocorrelation test
        autocorr = np.correlate(all_numbers, all_numbers, mode='full')
        randomness['autocorrelation'] = {
            'max_correlation': float(np.max(autocorr)),
            'mean_correlation': float(np.mean(autocorr))
        }
        
        return randomness
        
    except Exception as e:
        logger.error(f"Error testing randomness: {e}")
        return {'error': str(e)}

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def analyze_spatial_distribution(all_numbers: np.ndarray) -> Dict:
    """
    Analyze spatial distribution of numbers.
    
    Args:
        all_numbers: Array of all lottery numbers
        
    Returns:
        Dictionary containing spatial distribution analysis
    """
    try:
        bins = np.arange(1, 61, 10)  # Create bins of 10 numbers
        hist, _ = np.histogram(all_numbers, bins=bins)
        return {
            'distribution': hist.tolist(),
            'bins': bins.tolist()
        }
    except Exception as e:
        logger.error(f"Error analyzing spatial distribution: {e}")
        return {}

def analyze_range_frequency(df: pd.DataFrame) -> Dict:
    """
    Analyze frequency of numbers in different ranges.
    
    Args:
        df: DataFrame containing lottery data
        
    Returns:
        Dictionary containing range frequency analysis
    """
    try:
        ranges = {
            'low': (1, 20),
            'medium': (21, 40),
            'high': (41, 59)
        }
        
        frequency = {}
        for name, (start, end) in ranges.items():
            count = sum(1 for numbers in df['Main_Numbers'] for n in numbers if start <= n <= end)
            frequency[name] = count
            
        return frequency
    except Exception as e:
        logger.error(f"Error analyzing range frequency: {e}")
        return {}

def get_prediction_weights(df: pd.DataFrame, recent_draws: int = 50) -> Dict[str, Dict[int, float]]:
    """
    Generate weights for prediction scoring based on historical data.
    
    Args:
        df: DataFrame containing lottery data
        recent_draws: Number of recent draws to consider
        
    Returns:
        Dictionary of weights for individual numbers, pairs, and patterns
    """
    try:
        # Analyze data
        stats = analyze_lottery_data(df, recent_window=recent_draws)
        
        # Extract weights
        weights = {
            'number_weights': stats['number_weights'],
            'pair_weights': {tuple(map(int, k.split('_'))): v/recent_draws 
                            for k, v in stats['patterns']['pair_frequencies'].items()},
            'pattern_weights': {
                'consecutive_pairs': np.mean(stats['patterns']['consecutive_pairs']) / 6,
                'even_odd_ratio': {e: np.mean([1 if ratio[0] == e else 0 for ratio in stats['patterns']['even_odd_ratio']]) 
                                 for e in range(7)},
                'sum_ranges': stats['correlations']['sum_correlations'],
                'hot_numbers': {n: 2.0 for n in stats['hot_numbers']},
                'cold_numbers': {n: 0.5 for n in stats['cold_numbers']}
            }
        }
        
        return weights
        
    except Exception as e:
        logger.error(f"Error generating prediction weights: {e}")
        raise

if __name__ == "__main__":
    # For standalone testing
    try:
        # Load lottery data
        data_path = "data/lottery_data_1995_2025.csv"
        df = pd.read_csv(data_path)
        
        # Run analysis
        stats = analyze_lottery_data(df, cache_results=True)
        
        # Print summary
        print(f"Analysis of {len(df)} lottery draws:")
        print(f"Hot numbers: {stats['hot_numbers']}")
        print(f"Cold numbers: {stats['cold_numbers']}")
        print(f"Average sum: {stats['correlations']['sum_correlations']['mean']:.2f}")
        
        # Generate prediction weights
        weights = get_prediction_weights(df)
        print(f"Generated weights for {len(weights['number_weights'])} numbers")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error running analysis: {e}")