"""
Functions for analyzing lottery data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_file: Path) -> pd.DataFrame:
    """Load and preprocess lottery data."""
    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def analyze_lottery_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze lottery data for patterns and statistics."""
    results = {}
    
    try:
        # Basic statistics
        results['total_draws'] = len(df)
        results['date_range'] = (df['Date'].min(), df['Date'].max())
        
        # Number frequency analysis
        number_cols = [col for col in df.columns if col.startswith('Number')]
        all_numbers = df[number_cols].values.ravel()
        results['number_frequencies'] = pd.Series(all_numbers).value_counts().to_dict()
        
        # Gaps analysis
        results['max_gaps'] = {}
        for num in range(1, max(all_numbers) + 1):
            mask = df[number_cols].isin([num]).any(axis=1)
            gaps = mask.astype(int).diff().fillna(0)
            max_gap = (gaps == -1).astype(int).cumsum().max()
            results['max_gaps'][num] = max_gap
        
        # Pattern analysis
        results['common_pairs'] = find_common_pairs(df[number_cols])
        results['hot_numbers'] = find_hot_numbers(df[number_cols])
        results['cold_numbers'] = find_cold_numbers(df[number_cols])
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        raise

def find_common_pairs(numbers_df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """Find frequently occurring number pairs."""
    pairs = {}
    for i in range(len(numbers_df.columns)):
        for j in range(i + 1, len(numbers_df.columns)):
            col1, col2 = numbers_df.columns[i], numbers_df.columns[j]
            for _, row in numbers_df.iterrows():
                pair = tuple(sorted([row[col1], row[col2]]))
                pairs[pair] = pairs.get(pair, 0) + 1
    return dict(sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10])

def find_hot_numbers(numbers_df: pd.DataFrame, window: int = 10) -> List[int]:
    """Find numbers that have appeared frequently in recent draws."""
    recent = numbers_df.tail(window)
    frequencies = pd.Series(recent.values.ravel()).value_counts()
    return frequencies[frequencies > 1].index.tolist()

def find_cold_numbers(numbers_df: pd.DataFrame, window: int = 10) -> List[int]:
    """Find numbers that haven't appeared in recent draws."""
    recent = numbers_df.tail(window)
    all_numbers = set(range(1, numbers_df.max().max() + 1))
    recent_numbers = set(recent.values.ravel())
    return sorted(list(all_numbers - recent_numbers))

def test_randomness(df: pd.DataFrame) -> Dict[str, float]:
    """Test for randomness in the lottery numbers."""
    from scipy import stats
    
    results = {}
    number_cols = [col for col in df.columns if col.startswith('Number')]
    all_numbers = df[number_cols].values.ravel()
    
    # Chi-square test for uniform distribution
    observed = pd.Series(all_numbers).value_counts().sort_index()
    expected = np.ones_like(observed) * len(all_numbers) / len(observed)
    chi2, p_value = stats.chisquare(observed, expected)
    results['chi_square_p_value'] = p_value
    
    # Runs test for randomness
    median = np.median(all_numbers)
    runs = np.where(all_numbers > median, 1, 0)
    runs_stat, runs_p_value = stats.runs_test(runs)
    results['runs_test_p_value'] = runs_p_value
    
    return results 