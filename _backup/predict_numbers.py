"""Generate lottery number predictions using various models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Optional
from datetime import datetime
import json
import random
import logging
from pathlib import Path
import time
import traceback
from functools import lru_cache
import math
import importlib

from utils import setup_logging, LOG_DIR
from model_bridge import (
    ensure_valid_prediction,
    import_prediction_function,
    predict_with_model,
    score_combinations,
    ensemble_prediction,
    monte_carlo_simulation,
    save_predictions,
    predict_next_draw,
    calculate_prediction_metrics,
    calculate_metrics,
    backtest
)
from validation.data_validator import validate_prediction
from analyze_data import get_prediction_weights
from performance_tracking import get_model_weights

# Model imports
from models.lstm_model import predict_lstm_model
from models.holtwinters_model import predict_holtwinters_model
from models.linear_model import predict_linear_model
from models.xgboost_model import predict_xgboost_model
from models.lightgbm_model import predict_lightgbm_model
from models.knn_model import predict_knn_model
from models.gradient_boosting_model import predict_gradient_boosting_model
from models.catboost_model import predict_catboost_model
from models.cnn_lstm_model import predict_cnn_lstm_model
from models.autoencoder_model import predict_autoencoder_model
from models.meta_model import predict_meta_model

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Model function mapping
MODEL_FUNCTIONS = {
    'lstm': predict_lstm_model,
    'holtwinters': predict_holtwinters_model,
    'linear': predict_linear_model,
    'xgboost': predict_xgboost_model,
    'lightgbm': predict_lightgbm_model,
    'knn': predict_knn_model,
    'gradient_boosting': predict_gradient_boosting_model,
    'catboost': predict_catboost_model,
    'cnn_lstm': predict_cnn_lstm_model,
    'autoencoder': predict_autoencoder_model,
    'meta': predict_meta_model
}

# Only define these functions if we couldn't import from the bridge
if not IMPORTED_FROM_BRIDGE:
    def ensure_valid_prediction(prediction: List[int], min_val: int = 1, max_val: int = 59) -> List[int]:
        """
        Ensure a prediction contains 6 unique integers in the valid range.
        If the prediction is invalid, generate a valid random one.
        
        Args:
            prediction: The prediction to validate
            min_val: Minimum valid value (default: 1)
            max_val: Maximum valid value (default: 59)
            
        Returns:
            A valid prediction with 6 unique integers between min_val and max_val
        """
        # Check if prediction is already valid
        if (isinstance(prediction, list) and 
            len(prediction) == 6 and 
            all(isinstance(n, int) and min_val <= n <= max_val for n in prediction) and
            len(set(prediction)) == 6):
            return sorted(prediction)
        
        # Generate a random valid prediction
        valid_numbers = set()
        while len(valid_numbers) < 6:
            valid_numbers.add(random.randint(min_val, max_val))
        
        return sorted(list(valid_numbers))

    def import_prediction_function(model_name: str):
        """
        Dynamically import a prediction function for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Prediction function or None if import fails
        """
        if model_name not in MODEL_FUNCTIONS:
            logger.warning(f"Unknown model: {model_name}")
            return None
        
        try:
            # Try to import from scripts subdirectory first
            try:
                module = __import__(f"scripts.{model_name}", fromlist=[model_name])
            except ImportError:
                # Then try to import from root
                module = __import__(model_name, fromlist=[model_name])
            
            return getattr(module, model_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {model_name}: {e}")
            return None

    def predict_with_model(model_name: str, model: Any, data: pd.DataFrame) -> List[int]:
        """
        Generate a prediction using a specific model.
        
        Args:
            model_name: Name of the model
            model: The trained model object
            data: DataFrame with historical lottery data
            
        Returns:
            List of 6 predicted numbers
        """
        try:
            # Import the prediction function
            predict_func = import_prediction_function(model_name)
            
            if predict_func is None:
                logger.warning(f"No prediction function for {model_name}. Using random prediction.")
                return ensure_valid_prediction([])
            
            # Generate prediction
            prediction = predict_func(model, data)
            
            # Validate prediction
            is_valid, _ = validate_prediction(prediction)
            if not is_valid:
                logger.warning(f"Invalid prediction from {model_name}: {prediction}. Using corrected version.")
                prediction = ensure_valid_prediction(prediction)
            
            return sorted(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            return ensure_valid_prediction([])

    def score_combinations(combinations: List[List[int]], data: pd.DataFrame, weights: Dict) -> List[Tuple[List[int], float]]:
        """
        Score combinations based on historical patterns and frequencies.
        
        Args:
            combinations: List of number combinations to score
            data: DataFrame with historical lottery data
            weights: Dictionary of weights from get_prediction_weights
            
        Returns:
            List of (combination, score) tuples, sorted by score in descending order
        """
        try:
            scored_combinations = []
            number_weights = weights.get('number_weights', {})
            pair_weights = weights.get('pair_weights', {})
            pattern_weights = weights.get('pattern_weights', {})
            
            for numbers in combinations:
                score = 0.0
                
                # Individual number frequency score
                for num in numbers:
                    score += number_weights.get(num, 0) * 2.0
                
                # Pair frequency score
                for i, a in enumerate(numbers):
                    for b in numbers[i+1:]:
                        pair = (min(a, b), max(a, b))
                        score += pair_weights.get(pair, 0) * 0.5
                
                # Pattern-based scores
                # Consecutive pairs
                consecutive_pairs = sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1)
                score += consecutive_pairs * pattern_weights.get('consecutive_pairs', 0.1)
                
                # Even/odd distribution
                even_count = sum(1 for n in numbers if n % 2 == 0)
                score += pattern_weights.get('even_odd_ratio', {}).get(even_count, 0) * 0.5
                
                # Sum of numbers
                total_sum = sum(numbers)
                sum_mean = pattern_weights.get('sum_ranges', {}).get('mean', 150)
                sum_std = pattern_weights.get('sum_ranges', {}).get('std', 20)
                if sum_std > 0:
                    score += (1 - abs(total_sum - sum_mean) / (sum_std * 3)) * 0.5
                
                # Number range
                number_range = max(numbers) - min(numbers)
                score += (1 - abs(number_range - 40) / 40) * 0.3
                
                scored_combinations.append((numbers, score))
            
            # Sort by score in descending order
            return sorted(scored_combinations, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error scoring combinations: {str(e)}")
            logger.debug(traceback.format_exc())
            return [(combo, 0) for combo in combinations]

    def ensemble_prediction(individual_predictions: List[List[int]], 
                        data: pd.DataFrame, 
                        model_weights: Dict[str, float],
                        prediction_count: int = 10) -> List[List[int]]:
        """
        Generate ensemble predictions by combining individual model predictions.
        
        Args:
            individual_predictions: List of individual model predictions
            data: DataFrame with historical lottery data
            model_weights: Dictionary of model weights
            prediction_count: Number of predictions to generate
            
        Returns:
            List of prediction lists, each containing 6 unique integers
        """
        try:
            # Get weights for scoring combinations
            weights = get_prediction_weights(data)
            
            # Generate a pool of candidate combinations
            candidates = []
            # Add original model predictions
            candidates.extend(individual_predictions)
            
            # Generate additional candidates using Monte Carlo
            for _ in range(max(100, prediction_count * 5)):
                # Either combine existing predictions or generate new ones
                if random.random() < 0.7 and individual_predictions:
                    # Combine two random predictions
                    pred1 = random.choice(individual_predictions)
                    pred2 = random.choice(individual_predictions)
                    combined = list(set(pred1 + pred2))
                    
                    # Ensure exactly 6 numbers
                    while len(combined) > 6:
                        combined.pop(random.randrange(len(combined)))
                    while len(combined) < 6:
                        new_num = random.randint(1, 59)
                        if new_num not in combined:
                            combined.append(new_num)
                    
                    candidates.append(sorted(combined))
                else:
                    # Generate a random prediction
                    candidates.append(ensure_valid_prediction([]))
            
            # Ensure all candidates are valid and unique
            valid_candidates = []
            seen = set()
            for candidate in candidates:
                candidate_tuple = tuple(sorted(candidate))
                if candidate_tuple not in seen and validate_prediction(candidate)[0]:
                    valid_candidates.append(candidate)
                    seen.add(candidate_tuple)
            
            # Score all candidates
            scored_candidates = score_combinations(valid_candidates, data, weights)
            
            # Return top N predictions
            return [pred for pred, _ in scored_candidates[:prediction_count]]
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            logger.debug(traceback.format_exc())
            return [ensure_valid_prediction([]) for _ in range(prediction_count)]

    def predict_next_draw(models: Dict[str, Any], data: pd.DataFrame, n_predictions: int = 10) -> List[List[int]]:
        """
        Generate predictions for the next lottery draw using a combination of models.
        
        Args:
            models: Dictionary of model objects by name
            data: DataFrame with historical lottery data
            n_predictions: Number of predictions to generate
            
        Returns:
            List of predictions, each containing 6 unique integers
        """
        try:
            start_time = time.time()
            logger.info(f"Generating {n_predictions} predictions using {len(models)} models")
            
            # Get model weights
            model_weights = get_model_weights(models)
            logger.info(f"Model weights: {model_weights}")
            
            # Generate individual predictions from each model
            individual_predictions = []
            for model_name, model in models.items():
                if model is None:
                    logger.warning(f"Model {model_name} is None, skipping")
                    continue
                    
                # Generate prediction with the model
                prediction = predict_with_model(model_name, model, data)
                
                # Add to collection
                individual_predictions.append(prediction)
                logger.info(f"Model {model_name} prediction: {prediction}")
            
            # Generate ensemble predictions
            ensemble_predictions = ensemble_prediction(
                individual_predictions, data, model_weights, 
                prediction_count=n_predictions
            )
            
            # Calculate metrics
            metrics = calculate_prediction_metrics(ensemble_predictions, data)
            
            # Save predictions to file
            save_predictions(ensemble_predictions, metrics)
            
            duration = time.time() - start_time
            logger.info(f"Generated {len(ensemble_predictions)} predictions in {duration:.2f} seconds")
            
            return ensemble_predictions
            
        except Exception as e:
            logger.error(f"Error predicting next draw: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Return random valid predictions as fallback
            return [ensure_valid_prediction([]) for _ in range(n_predictions)]

    def save_predictions(predictions: List[List[int]], metrics: Dict[str, float] = None) -> None:
        """
        Save predictions to a JSON file.
        
        Args:
            predictions: List of prediction lists
            metrics: Optional dictionary of prediction metrics
        """
        try:
            # Prepare output structure
            output = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'count': len(predictions)
            }
            
            # Add metrics if provided
            if metrics:
                output['metrics'] = metrics
            
            # Create directories if needed
            prediction_file = Path('results/predictions.json')
            prediction_file.parent.mkdir(exist_ok=True)
            
            # Save to file
            with open(prediction_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Saved {len(predictions)} predictions to {prediction_file}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            logger.debug(traceback.format_exc())

    def calculate_prediction_metrics(predictions: List[List[int]], data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate metrics for the predictions.
        
        Args:
            predictions: List of prediction lists
            data: DataFrame with historical lottery data
            
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}
            
            # Get the most recent actual draw
            try:
                latest_draw = data['Main_Numbers'].iloc[-1]
                
                # Calculate matches (partial accuracy)
                total_matches = 0
                for prediction in predictions:
                    matches = len(set(prediction) & set(latest_draw))
                    total_matches += matches
                
                # Calculate average matches (out of 6)
                avg_matches = total_matches / (len(predictions) * 6) if predictions else 0
                metrics['accuracy'] = avg_matches
                
                # Match rate (average number of matched numbers per prediction)
                metrics['match_rate'] = total_matches / len(predictions) if predictions else 0
                
                # Calculate perfect matches (full accuracy)
                perfect_matches = sum(1 for p in predictions if set(p) == set(latest_draw))
                metrics['perfect_match_rate'] = perfect_matches / len(predictions) if predictions else 0
                
                # Calculate RMSE
                square_errors = []
                for prediction in predictions:
                    # Calculate squared error for each prediction
                    error = sum((1 if n in latest_draw else 0) - (1 if n in prediction else 0) for n in range(1, 60))
                    square_errors.append(error ** 2)
                
                metrics['rmse'] = math.sqrt(sum(square_errors) / len(square_errors)) if square_errors else 0
                
            except Exception as e:
                logger.warning(f"Could not calculate metrics against latest draw: {str(e)}")
                raise  # Re-raise to be caught by outer try-except
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating prediction metrics: {str(e)}")
            return {
                'error': str(e),
                'accuracy': 0,
                'match_rate': 0,
                'perfect_match_rate': 0,
                'rmse': 0
            }

    def monte_carlo_simulation(data: pd.DataFrame, n_simulations: int = 1000) -> List[List[int]]:
        """
        Generate predictions using Monte Carlo simulation.
        
        Args:
            data: DataFrame with historical lottery data
            n_simulations: Number of simulations to run
            
        Returns:
            List of prediction lists
        """
        try:
            weights = get_prediction_weights(data)
            number_weights = weights.get('number_weights', {})
            
            # Normalize weights for numpy.random.choice
            numbers = list(range(1, 60))
            probabilities = [number_weights.get(n, 1/59) for n in numbers]
            total = sum(probabilities)
            probabilities = [p/total for p in probabilities]
            
            # Generate predictions
            predictions = []
            for _ in range(n_simulations):
                prediction = []
                while len(prediction) < 6:
                    num = np.random.choice(numbers, p=probabilities)
                    if num not in prediction:
                        prediction.append(num)
                predictions.append(sorted(prediction))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            logger.debug(traceback.format_exc())
            return [ensure_valid_prediction([]) for _ in range(n_simulations)]

    def backtest(models: Dict[str, Any], df: pd.DataFrame, test_size: int = 10) -> Dict[str, float]:
        """
        Backtest models on historical data.
        
        Args:
            models: Dictionary of model objects by name
            df: DataFrame with historical lottery data
            test_size: Number of past draws to test on
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Ensure we have enough data
            if len(df) <= test_size:
                raise ValueError(f"Not enough data for backtesting. Need more than {test_size} draws.")
            
            # Initialize metrics
            metrics = {
                'total_predictions': 0,
                'matches': 0,
                'perfect_matches': 0,
                'test_size': -test_size  # Will be negative to indicate relative indexing
            }
            
            # Iterate through test data in reverse (most recent first)
            for i in range(test_size):
                # Use data up to this point
                train_df = df.iloc[:-test_size+i]
                # Get actual draw
                actual_draw = df.iloc[-test_size+i]['Main_Numbers']
                
                # Generate prediction
                predictions = predict_next_draw(models, train_df, n_predictions=1)
                
                if predictions and len(predictions) > 0:
                    prediction = predictions[0]
                    # Count matches
                    matches = len(set(prediction) & set(actual_draw))
                    metrics['matches'] += matches
                    metrics['total_predictions'] += 1
                    
                    # Check for perfect match
                    if set(prediction) == set(actual_draw):
                        metrics['perfect_matches'] += 1
            
            # Calculate aggregate metrics
            if metrics['total_predictions'] > 0:
                metrics['accuracy'] = metrics['matches'] / (metrics['total_predictions'] * 6)
                metrics['match_rate'] = metrics['matches'] / metrics['total_predictions']
                metrics['perfect_match'] = metrics['perfect_matches'] / metrics['total_predictions']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'error': str(e),
                'total_predictions': 0,
                'matches': 0,
                'perfect_matches': 0,
                'test_size': -test_size
            }

if __name__ == "__main__":
    # Example usage when run directly
    try:
        # Import needed modules
        import argparse
        from fetch_data import load_data
        
        # Configure argument parser
        parser = argparse.ArgumentParser(description='Generate lottery predictions')
        parser.add_argument('--data', type=str, default='data/lottery_data_1995_2025.csv',
                          help='Path to lottery data CSV file')
        parser.add_argument('--count', type=int, default=10,
                          help='Number of predictions to generate')
        parser.add_argument('--mode', type=str, default='monte_carlo',
                          choices=['monte_carlo', 'backtest'],
                          help='Prediction mode (monte_carlo or backtest)')
        parser.add_argument('--test-size', type=int, default=10,
                          help='Number of historical draws to test against in backtest mode')
        
        args = parser.parse_args()
        
        # Load data
        print(f"Loading data from {args.data}...")
        df = load_data(args.data)
        print(f"Loaded {len(df)} lottery draws")
        
        if args.mode == 'monte_carlo':
            # Generate predictions with Monte Carlo
            print(f"Generating {args.count} predictions with Monte Carlo simulation...")
            predictions = monte_carlo_simulation(df, n_simulations=args.count)
            
            # Print predictions
            print("\nPredictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"{i:2d}: {pred}")
            
            # Save predictions
            save_predictions(predictions)
            print(f"Saved predictions to results/predictions.json")
            
        elif args.mode == 'backtest':
            # No models available in standalone mode, run monte carlo
            print(f"Running backtest with monte carlo against {args.test_size} historical draws...")
            test_size = min(args.test_size, len(df) - 10)
            
            predictions = []
            actual_draws = []
            matches = []
            
            # Simple monte carlo backtest
            for i in range(test_size):
                train_data = df.iloc[:-test_size+i]
                test_draw = df.iloc[-test_size+i]['Main_Numbers']
                
                # Generate monte carlo prediction
                pred = monte_carlo_simulation(train_data, n_simulations=1)[0]
                
                # Calculate matches
                match_count = len(set(pred) & set(test_draw))
                
                # Store results
                predictions.append(pred)
                actual_draws.append(test_draw)
                matches.append(match_count)
                
                print(f"Draw {i+1}: Prediction {pred}, Actual {test_draw}, Matches: {match_count}/6")
            
            # Print summary
            if predictions:
                avg_matches = sum(matches) / len(matches)
                print(f"\nAverage matches: {avg_matches:.2f}/6 ({avg_matches/6:.1%})")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc() 