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

# Try to import from utils and other modules
try:
    from utils import setup_logging
    from data_validation import validate_prediction
    from analyze_data import get_prediction_weights
except ImportError:
    # Default implementation if imports fail
    def setup_logging():
        logging.basicConfig(filename='lottery.log', level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def validate_prediction(pred):
        """Basic prediction validation"""
        if not isinstance(pred, list) or len(pred) != 6:
            return False
        if not all(isinstance(n, int) and 1 <= n <= 59 for n in pred):
            return False
        if len(set(pred)) != 6:
            return False
        return True
    
    def get_prediction_weights(df, recent_draws=50):
        """Fallback implementation for prediction weights"""
        return {
            'number_weights': {n: 1/59 for n in range(1, 60)},
            'pair_weights': {},
            'pattern_weights': {
                'consecutive_pairs': 0.1,
                'even_odd_ratio': {n: 1/7 for n in range(7)},
                'sum_ranges': {'mean': 150, 'std': 20, 'min': 100, 'max': 200},
                'hot_numbers': {},
                'cold_numbers': {}
            }
        }

# Model imports (will be dynamically imported when needed)
MODEL_FUNCTIONS = {
    'lstm': ('lstm_model', 'predict_lstm_model'),
    'holtwinters': ('holtwinters_model', 'predict_holtwinters_model'),
    'linear': ('linear_model', 'predict_linear_model'),
    'xgboost': ('xgboost_model', 'predict_xgboost_model'),
    'lightgbm': ('lightgbm_model', 'predict_lightgbm_model'),
    'knn': ('knn_model', 'predict_knn_model'),
    'gradient_boosting': ('gradient_boosting_model', 'predict_gradient_boosting_model'),
    'catboost': ('catboost_model', 'predict_catboost_model'),
    'cnn_lstm': ('cnn_lstm_model', 'predict_cnn_lstm_model'),
    'autoencoder': ('autoencoder_model', 'predict_autoencoder_model'),
    'meta': ('meta_model', 'predict_meta_model')
}

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

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
    
    module_name, function_name = MODEL_FUNCTIONS[model_name]
    
    try:
        # Try to import from scripts subdirectory first
        try:
            module = __import__(f"scripts.{module_name}", fromlist=[function_name])
        except ImportError:
            # Then try to import from root
            module = __import__(module_name, fromlist=[function_name])
        
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {function_name} from {module_name}: {e}")
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
        if not validate_prediction(prediction):
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
            if candidate_tuple not in seen and validate_prediction(candidate):
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

def get_model_weights(models: Dict[str, Any]) -> Dict[str, float]:
    """
    Get weights for each model based on their historical performance.
    Placeholder for more sophisticated model weighting.
    
    Args:
        models: Dictionary of model objects
        
    Returns:
        Dictionary of model weights
    """
    try:
        # Default weights if performance data isn't available
        default_weights = {
            'lstm': 0.15,
            'holtwinters': 0.10,
            'linear': 0.05,
            'xgboost': 0.15,
            'lightgbm': 0.15,
            'knn': 0.05,
            'gradient_boosting': 0.10,
            'catboost': 0.10,
            'cnn_lstm': 0.10,
            'autoencoder': 0.05,
            'meta': 0.00  # Meta model is used separately
        }
        
        # Try to load performance data
        try:
            performance_file = Path('results/model_performance.json')
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance = json.load(f)
                
                # Extract accuracy metrics
                weights = {}
                for model_name in models.keys():
                    if model_name in performance:
                        # Use accuracy as weight, with minimum of 0.01
                        weights[model_name] = max(performance[model_name].get('accuracy', 0), 0.01)
                    else:
                        weights[model_name] = default_weights.get(model_name, 0.05)
                
                # Normalize weights to sum to 1
                total = sum(weights.values())
                if total > 0:
                    return {k: v/total for k, v in weights.items()}
            
            return default_weights
            
        except Exception as e:
            logger.warning(f"Error loading model performance data: {str(e)}")
            return default_weights
    
    except Exception as e:
        logger.error(f"Error getting model weights: {str(e)}")
        return {model: 1.0 / len(models) for model in models}

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
            
            # Calculate perfect matches (full accuracy)
            perfect_matches = sum(1 for p in predictions if set(p) == set(latest_draw))
            metrics['perfect_match'] = perfect_matches / len(predictions) if predictions else 0
            
            # Add other metrics of interest
            metrics['exact_match_count'] = perfect_matches
            metrics['average_match_count'] = total_matches / len(predictions) if predictions else 0
            
        except Exception as e:
            logger.warning(f"Could not calculate metrics against latest draw: {str(e)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating prediction metrics: {str(e)}")
        return {'accuracy': 0, 'perfect_match': 0}

def monte_carlo_simulation(df: pd.DataFrame, n_simulations: int = 1000) -> List[List[int]]:
    """
    Generate predictions using Monte Carlo simulation.
    
    Args:
        df: DataFrame with historical lottery data
        n_simulations: Number of simulations to run
        
    Returns:
        List of prediction lists
    """
    try:
        # Get number weights from prediction weights
        weights = get_prediction_weights(df)
        number_weights = weights.get('number_weights', {})
        
        # Normalize weights if needed
        total_weight = sum(number_weights.values())
        if total_weight == 0:
            # Use uniform weights if all weights are zero
            numbers = list(range(1, 60))
            probs = [1/59] * 59
        else:
            # Normalize weights
            numbers = list(number_weights.keys())
            probs = [number_weights[n] / total_weight for n in numbers]
        
        # Generate predictions
        predictions = []
        for _ in range(n_simulations):
            # Generate a prediction with weighted sampling
            pred = []
            while len(pred) < 6:
                num = np.random.choice(numbers, p=probs)
                if num not in pred:
                    pred.append(num)
            
            predictions.append(sorted(pred))
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        logger.debug(traceback.format_exc())
        return [ensure_valid_prediction([]) for _ in range(n_simulations)]

def backtest(models: Dict[str, Any], df: pd.DataFrame, test_size: int = 10) -> Dict[str, float]:
    """
    Perform backtesting on historical data.
    
    Args:
        models: Dictionary of model objects
        df: DataFrame with historical lottery data
        test_size: Number of historical draws to test against
        
    Returns:
        Dictionary of metrics
    """
    try:
        start_time = time.time()
        logger.info(f"Running backtest with {test_size} historical draws")
        
        # Ensure test_size is not larger than the dataset
        test_size = min(test_size, len(df) - 10)  # Need at least 10 draws for training
        
        predictions = []
        actual_draws = []
        
        # Iterate over test draws
        for i in range(test_size):
            # Split data for this iteration
            train_data = df.iloc[:-test_size+i]
            test_draw = df.iloc[-test_size+i]['Main_Numbers']
            
            # Skip if training data is too small
            if len(train_data) < 10:
                continue
            
            # Generate prediction
            pred = predict_next_draw(models, train_data, n_predictions=1)[0]
            
            # Store prediction and actual draw
            predictions.append(pred)
            actual_draws.append(test_draw)
        
        # Calculate metrics
        metrics = {
            'total_predictions': len(predictions),
            'test_size': test_size
        }
        
        # Calculate match metrics
        total_matches = 0
        perfect_matches = 0
        
        for pred, actual in zip(predictions, actual_draws):
            matches = len(set(pred) & set(actual))
            total_matches += matches
            if matches == 6:
                perfect_matches += 1
        
        if predictions:
            metrics['accuracy'] = total_matches / (len(predictions) * 6)
            metrics['perfect_match_rate'] = perfect_matches / len(predictions)
            metrics['average_matched_numbers'] = total_matches / len(predictions)
        
        duration = time.time() - start_time
        logger.info(f"Backtest completed in {duration:.2f} seconds: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}

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