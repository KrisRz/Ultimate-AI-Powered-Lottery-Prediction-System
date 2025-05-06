"""Prediction validation utilities for lottery prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import traceback
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import random

from utils import setup_logging, LOG_DIR

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

class LotteryValidator:
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.number_frequencies = self._calculate_frequencies()
        
    def _calculate_frequencies(self) -> Dict[int, int]:
        """Calculate frequency of each number in historical data."""
        frequencies = Counter()
        for numbers in self.historical_data['Main_Numbers']:
            frequencies.update(numbers)
        return frequencies
        
    def _get_hot_cold_numbers(self, n: int = 10) -> Tuple[List[int], List[int]]:
        """Get most and least frequent numbers."""
        items = list(self.number_frequencies.items())
        items.sort(key=lambda x: x[1], reverse=True)
        
        hot_numbers = [num for num, _ in items[:n]]
        cold_numbers = [num for num, _ in items[-n:]]
        
        return hot_numbers, cold_numbers
        
    def _calculate_backtest_metrics(self, predictions: List[List[int]]) -> Dict[str, float]:
        """Calculate backtesting metrics."""
        metrics = {}
        
        # Use last n predictions for backtesting
        n = min(len(predictions), len(self.historical_data))
        actual = self.historical_data['Main_Numbers'].iloc[-n:].tolist()
        predictions = predictions[-n:]
        
        # Calculate metrics
        total_matches = 0
        exact_matches = 0
        
        for pred, act in zip(predictions, actual):
            matches = len(set(pred) & set(act))
            total_matches += matches
            if matches == 6:
                exact_matches += 1
                
        metrics['match_rate'] = (total_matches / (n * 6)) * 100
        metrics['exact_match_rate'] = (exact_matches / n) * 100
        
        return metrics
        
    def _monte_carlo_simulation(self, n_simulations: int = 1000) -> Dict[str, float]:
        """Run Monte Carlo simulation with random guessing."""
        metrics = {}
        total_matches = 0
        exact_matches = 0
        
        for _ in range(n_simulations):
            # Random prediction
            pred = sorted(random.sample(range(1, 60), 6))
            # Random actual from historical data
            act = random.choice(self.historical_data['Main_Numbers'])
            
            matches = len(set(pred) & set(act))
            total_matches += matches
            if matches == 6:
                exact_matches += 1
                
        metrics['random_match_rate'] = (total_matches / (n_simulations * 6)) * 100
        metrics['random_exact_rate'] = (exact_matches / n_simulations) * 100
        
        return metrics
        
    def _analyze_distribution(self, predictions: List[List[int]]) -> Dict[str, float]:
        """Analyze number distribution in predictions."""
        metrics = {}
        
        # Flatten predictions
        flat_preds = [num for pred in predictions for num in pred]
        pred_freq = Counter(flat_preds)
        
        # Calculate distribution metrics
        total_numbers = len(flat_preds)
        unique_numbers = len(pred_freq)
        
        metrics['unique_ratio'] = unique_numbers / 59  # 59 possible numbers
        
        # Calculate entropy (measure of randomness)
        probs = [count/total_numbers for count in pred_freq.values()]
        entropy = -sum(p * np.log2(p) for p in probs)
        metrics['entropy'] = entropy
        
        return metrics
        
    def _analyze_consistency(self, predictions: List[List[int]]) -> Dict[str, float]:
        """Analyze prediction consistency."""
        metrics = {}
        
        # Calculate average difference between consecutive predictions
        diffs = []
        for i in range(len(predictions)-1):
            diff = len(set(predictions[i]) & set(predictions[i+1]))
            diffs.append(diff)
            
        metrics['avg_consistency'] = np.mean(diffs)
        metrics['std_consistency'] = np.std(diffs)
        
        return metrics
        
    def validate_predictions(self, predictions: List[List[int]], n_monte_carlo: int = 1000) -> Dict[str, Any]:
        """Run comprehensive validation on predictions."""
        try:
            outputs/results = {}
            
            # Get hot and cold numbers
            hot_numbers, cold_numbers = self._get_hot_cold_numbers()
            outputs/results['hot_numbers'] = hot_numbers
            outputs/results['cold_numbers'] = cold_numbers
            
            # Backtesting
            outputs/results['backtest'] = self._calculate_backtest_metrics(predictions)
            
            # Monte Carlo simulation
            outputs/results['monte_carlo'] = self._monte_carlo_simulation(n_monte_carlo)
            
            # Calculate improvement over random
            outputs/results['improvement'] = {
                'match_rate': outputs/results['backtest']['match_rate'] - outputs/results['monte_carlo']['random_match_rate'],
                'exact_rate': outputs/results['backtest']['exact_match_rate'] - outputs/results['monte_carlo']['random_exact_rate']
            }
            
            # Distribution analysis
            outputs/results['distribution'] = self._analyze_distribution(predictions)
            
            # Consistency analysis
            outputs/results['consistency'] = self._analyze_consistency(predictions)
            
            return outputs/results
            
        except Exception as e:
            logger.error(f"Error in prediction validation: {str(e)}")
            raise 