import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

# Try to import from utils
try:
    from utils import setup_logging
except ImportError:
    # Default implementation if imports fail
    def setup_logging():
        logging.basicConfig(filename='lottery.log', level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def calculate_metrics(predictions: Union[np.ndarray, List[List[int]]], 
                     actual: Union[np.ndarray, List[List[int]]]) -> Dict[str, float]:
    """
    Calculate performance metrics for lottery predictions.
    
    Args:
        predictions: Array/List of shape (n_predictions, 6) with predicted numbers
        actual: Array/List of shape (n_actual, 6) with actual drawn numbers
        
    Returns:
        Dictionary with metrics (accuracy, match_rate, rmse, mae)
    """
    try:
        # Ensure inputs are numpy arrays
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(actual, np.ndarray):
            actual = np.array(actual)
        
        # Validate shapes
        if len(predictions.shape) != 2 or predictions.shape[1] != 6:
            logger.warning(f"Invalid predictions shape: {predictions.shape}. Expected (n, 6)")
            if len(predictions.shape) == 1 and len(predictions) == 6:
                # Handle single prediction case
                predictions = predictions.reshape(1, 6)
            else:
                raise ValueError(f"Predictions must have shape (n, 6), got {predictions.shape}")
        
        if len(actual.shape) != 2 or actual.shape[1] != 6:
            logger.warning(f"Invalid actual shape: {actual.shape}. Expected (n, 6)")
            if len(actual.shape) == 1 and len(actual) == 6:
                # Handle single actual case
                actual = actual.reshape(1, 6)
            else:
                raise ValueError(f"Actual must have shape (n, 6), got {actual.shape}")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Match-based accuracy (proportion of matching numbers)
        total_matches = 0
        total_numbers = predictions.shape[0] * predictions.shape[1]
        
        for pred_row, actual_row in zip(predictions, actual):
            # Count matches between prediction and actual draw
            matches = len(set(pred_row) & set(actual_row))
            total_matches += matches
        
        # Calculate accuracy as proportion of matches
        if total_numbers > 0:
            metrics['accuracy'] = total_matches / total_numbers
        else:
            metrics['accuracy'] = 0.0
        
        # Calculate match rate (average number of matching numbers per prediction)
        if predictions.shape[0] > 0:
            metrics['match_rate'] = total_matches / predictions.shape[0]
        else:
            metrics['match_rate'] = 0.0
        
        # Calculate perfect match rate (proportion of exact matches)
        perfect_matches = 0
        for pred_row, actual_row in zip(predictions, actual):
            if set(pred_row) == set(actual_row):
                perfect_matches += 1
        
        if predictions.shape[0] > 0:
            metrics['perfect_match_rate'] = perfect_matches / predictions.shape[0]
        else:
            metrics['perfect_match_rate'] = 0.0
        
        # Calculate RMSE (Root Mean Squared Error)
        squared_errors = []
        for pred_row, actual_row in zip(predictions, actual):
            # Sort both arrays for consistent comparison
            pred_sorted = np.sort(pred_row)
            actual_sorted = np.sort(actual_row)
            # Calculate squared errors
            squared_error = np.mean((pred_sorted - actual_sorted) ** 2)
            squared_errors.append(squared_error)
        
        if squared_errors:
            metrics['rmse'] = float(np.sqrt(np.mean(squared_errors)))
        else:
            metrics['rmse'] = 0.0
        
        # Calculate MAE (Mean Absolute Error)
        abs_errors = []
        for pred_row, actual_row in zip(predictions, actual):
            # Sort both arrays for consistent comparison
            pred_sorted = np.sort(pred_row)
            actual_sorted = np.sort(actual_row)
            # Calculate absolute errors
            abs_error = np.mean(np.abs(pred_sorted - actual_sorted))
            abs_errors.append(abs_error)
        
        if abs_errors:
            metrics['mae'] = float(np.mean(abs_errors))
        else:
            metrics['mae'] = 0.0
        
        # Add top-k accuracy as optional metric
        metrics['top_3_accuracy'] = calculate_top_k_accuracy(predictions, actual, k=3)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        logger.debug(traceback.format_exc())
        # Return default metrics in case of error
        return {
            'accuracy': 0.0,
            'match_rate': 0.0,
            'perfect_match_rate': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'top_3_accuracy': 0.0
        }

def calculate_top_k_accuracy(predictions: np.ndarray, actual: np.ndarray, k: int = 3) -> float:
    """
    Calculate top-k accuracy (number within k of actual).
    
    Args:
        predictions: Array of shape (n_predictions, 6) with predicted numbers
        actual: Array of shape (n_actual, 6) with actual drawn numbers
        k: Maximum deviation allowed to consider a match
        
    Returns:
        Proportion of numbers within k of an actual number
    """
    try:
        # Initialize counter
        within_k_count = 0
        total_count = 0
        
        # Iterate through predictions
        for pred_row, actual_row in zip(predictions, actual):
            for p_num in pred_row:
                total_count += 1
                # Check if any actual number is within k of predicted number
                if any(abs(p_num - a_num) <= k for a_num in actual_row):
                    within_k_count += 1
        
        # Calculate accuracy
        if total_count > 0:
            return within_k_count / total_count
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating top-k accuracy: {str(e)}")
        logger.debug(traceback.format_exc())
        return 0.0

def track_model_performance(
    model_name: str,
    predictions: Union[np.ndarray, List[List[int]]],
    actual: Union[np.ndarray, List[List[int]]],
    draw_date: Optional[Union[str, datetime]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Track and log model performance.
    
    Args:
        model_name: Name of the model
        predictions: Predicted numbers
        actual: Actual drawn numbers
        draw_date: Date of the draw (default: current date)
        metadata: Additional information to log (optional)
        
    Returns:
        Dictionary with performance data
    """
    try:
        start_time = time.time()
        
        # Set timestamp if not provided
        if draw_date is None:
            timestamp = datetime.now().isoformat()
        elif isinstance(draw_date, datetime):
            timestamp = draw_date.isoformat()
        else:
            timestamp = str(draw_date)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, actual)
        
        # Create performance data
        performance_data = {
            'model': model_name,
            'timestamp': timestamp,
            'metrics': metrics,
            'prediction_count': len(predictions) if hasattr(predictions, '__len__') else 0
        }
        
        # Add metadata if provided
        if metadata:
            performance_data['metadata'] = metadata
        
        # Save to log file
        log_file = Path('results/performance_log.json')
        log_file.parent.mkdir(exist_ok=True)
        
        # Read existing history
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read performance log: {e}. Starting new log.")
            history = []
        
        # Append new performance data
        history.append(performance_data)
        
        # Trim history if too long (keep latest 100 records)
        if len(history) > 100:
            history = history[-100:]
        
        # Write updated history
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Log performance summary
        duration = time.time() - start_time
        logger.info(f"Tracked performance for {model_name}: accuracy={metrics['accuracy']:.4f}, "
                   f"match_rate={metrics['match_rate']:.4f}, rmse={metrics['rmse']:.4f}")
        
        # Log performance trends
        log_performance_trends(model_name, history)
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error tracking performance for {model_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': 0.0,
                'match_rate': 0.0,
                'perfect_match_rate': 0.0,
                'rmse': 0.0,
                'mae': 0.0
            },
            'error': str(e)
        }

def log_performance_trends(model_name: str, history: List[Dict[str, Any]]) -> None:
    """
    Log performance trends for a specific model.
    
    Args:
        model_name: Name of the model
        history: List of performance records
    """
    try:
        # Filter history for this model
        model_history = [h for h in history if h.get('model') == model_name]
        
        # Need at least 2 records to calculate trends
        if len(model_history) < 2:
            return
        
        # Get most recent and previous metrics
        recent = model_history[-1]['metrics']
        prev_metrics = [h['metrics'] for h in model_history[:-1]]
        
        # Calculate averages
        avg_accuracy = sum(m.get('accuracy', 0) for m in prev_metrics) / len(prev_metrics)
        avg_rmse = sum(m.get('rmse', 0) for m in prev_metrics) / len(prev_metrics)
        
        # Detect significant changes
        accuracy_change = recent.get('accuracy', 0) - avg_accuracy
        rmse_change = recent.get('rmse', 0) - avg_rmse
        
        # Log trends
        if abs(accuracy_change) > 0.05:  # 5% change threshold
            if accuracy_change > 0:
                logger.info(f"Model {model_name} accuracy improved by {accuracy_change:.4f} above average")
            else:
                logger.warning(f"Model {model_name} accuracy decreased by {abs(accuracy_change):.4f} below average")
        
        if abs(rmse_change) > 2.0:  # Arbitrary threshold for RMSE
            if rmse_change < 0:
                logger.info(f"Model {model_name} RMSE improved by {-rmse_change:.4f} below average")
            else:
                logger.warning(f"Model {model_name} RMSE increased by {rmse_change:.4f} above average")
                
    except Exception as e:
        logger.error(f"Error logging performance trends for {model_name}: {str(e)}")

def get_model_weights(models: Optional[List[str]] = None, 
                     history_length: int = 10, 
                     decay_factor: float = 0.9,
                     min_weight: float = 0.01) -> Dict[str, float]:
    """
    Calculate model weights based on their historical performance.
    
    Args:
        models: List of model names to include (default: None for all models)
        history_length: Number of recent performance records to consider
        decay_factor: Factor to prioritize recent performance (0-1)
        min_weight: Minimum weight for any model
        
    Returns:
        Dictionary of model weights normalized to sum to 1.0
    """
    try:
        # Get default weights as fallback
        default_weights = get_default_weights()
        
        # If no specific models requested, use all models from defaults
        if models is None:
            models = list(default_weights.keys())
        
        # Read performance history
        log_file = Path('results/performance_log.json')
        if not log_file.exists():
            logger.warning(f"No performance log found at {log_file}. Using default weights.")
            # Filter defaults to only include requested models
            weights = {model: default_weights.get(model, min_weight) for model in models}
            return normalize_weights(weights, min_weight)
        
        try:
            with open(log_file, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading performance log: {e}. Using default weights.")
            # Filter defaults to only include requested models
            weights = {model: default_weights.get(model, min_weight) for model in models}
            return normalize_weights(weights, min_weight)
        
        # Collect recent history by model
        model_history = {}
        for entry in history[-history_length*len(models):]:  # Get enough history for all models
            model = entry.get('model')
            if model in models:
                if model not in model_history:
                    model_history[model] = []
                model_history[model].append(entry)
        
        # Keep only most recent entries for each model
        for model in model_history:
            model_history[model] = sorted(model_history[model], 
                                        key=lambda x: x.get('timestamp', ''))[-history_length:]
        
        # Calculate weighted performance for each model
        weights = {}
        for model in models:
            if model not in model_history or not model_history[model]:
                # Use default if no history
                weights[model] = default_weights.get(model, min_weight)
                continue
            
            # Calculate weighted score from metrics
            weighted_scores = []
            for i, entry in enumerate(model_history[model]):
                metrics = entry.get('metrics', {})
                
                # Get accuracy and error metrics
                accuracy = metrics.get('accuracy', 0)
                rmse = metrics.get('rmse', 1)  # Default to 1 if missing
                match_rate = metrics.get('match_rate', 0)
                
                # Combine metrics into a single score
                # Higher is better: accuracy and match_rate
                # Lower is better: rmse, so we invert it
                score = (accuracy * 0.4) + (match_rate * 0.4) + (1.0 / (1.0 + rmse) * 0.2)
                
                # Apply decay factor - more recent entries get higher weight
                rec_idx = len(model_history[model]) - i - 1  # Reverse index
                decay = decay_factor ** rec_idx
                weighted_scores.append(score * decay)
            
            # Average the weighted scores
            weights[model] = sum(weighted_scores) / len(weighted_scores) if weighted_scores else min_weight
        
        # Ensure all models have at least the minimum weight
        return normalize_weights(weights, min_weight)
        
    except Exception as e:
        logger.error(f"Error calculating model weights: {str(e)}")
        logger.debug(traceback.format_exc())
        return get_default_weights()

def normalize_weights(weights: Dict[str, float], min_weight: float = 0.01) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0 while ensuring minimum weight.
    
    Args:
        weights: Dictionary of unnormalized weights
        min_weight: Minimum weight for any model
        
    Returns:
        Dictionary of normalized weights
    """
    try:
        # Ensure all weights are at least the minimum
        adjusted_weights = {k: max(v, min_weight) for k, v in weights.items()}
        
        # Calculate sum
        total = sum(adjusted_weights.values())
        
        # Normalize
        if total > 0:
            return {k: v / total for k, v in adjusted_weights.items()}
        else:
            # Equal weights if total is zero
            return {k: 1.0 / len(adjusted_weights) for k in adjusted_weights}
            
    except Exception as e:
        logger.error(f"Error normalizing weights: {str(e)}")
        # Equal weights as fallback
        return {k: 1.0 / len(weights) for k in weights} if weights else {}

def get_default_weights() -> Dict[str, float]:
    """
    Get default model weights when no performance history is available.
    
    Returns:
        Dictionary of model weights
    """
    return {
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
        'meta': 0.00  # Meta model is considered separately
    }

def get_model_performance_summary(model_name: Optional[str] = None, 
                                history_length: int = 10) -> Dict[str, Any]:
    """
    Get a summary of model performance.
    
    Args:
        model_name: Name of model to summarize (None for all models)
        history_length: Number of recent entries to include in summary
        
    Returns:
        Dictionary with performance summary
    """
    try:
        # Read performance history
        log_file = Path('results/performance_log.json')
        if not log_file.exists():
            return {'error': 'No performance log found'}
        
        with open(log_file, 'r') as f:
            history = json.load(f)
        
        # Filter by model if specified
        if model_name:
            model_history = [entry for entry in history if entry.get('model') == model_name]
        else:
            model_history = history
        
        # Take only the most recent entries
        recent_history = model_history[-history_length:]
        
        # Group by model
        model_metrics = {}
        for entry in recent_history:
            model = entry.get('model')
            if model not in model_metrics:
                model_metrics[model] = []
            model_metrics[model].append(entry.get('metrics', {}))
        
        # Calculate average metrics for each model
        summary = {}
        for model, metrics_list in model_metrics.items():
            # Skip if no metrics
            if not metrics_list:
                continue
                
            avg_metrics = {}
            for metric in ['accuracy', 'match_rate', 'perfect_match_rate', 'rmse', 'mae']:
                values = [m.get(metric, 0) for m in metrics_list]
                avg_metrics[metric] = sum(values) / len(values) if values else 0
            
            summary[model] = {
                'avg_metrics': avg_metrics,
                'entries': len(metrics_list),
                'timestamp': datetime.now().isoformat()
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {str(e)}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}

def calculate_ensemble_weights(model_names: List[str], data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Calculate ensemble weights optimized for the given data.
    
    Args:
        model_names: List of model names to include in ensemble
        data: Optional DataFrame with historical data for advanced weighting
        
    Returns:
        Dictionary of model weights for ensemble
    """
    try:
        # Basic weights from performance history
        basic_weights = get_model_weights(model_names)
        
        # If no data provided, just return basic weights
        if data is None:
            return basic_weights
        
        # Get model diversity based on correlations (if available)
        try:
            diversity_bonus = calculate_diversity_bonus(model_names)
            
            # Apply diversity bonus
            adjusted_weights = {}
            for model in basic_weights:
                bonus = diversity_bonus.get(model, 1.0)
                adjusted_weights[model] = basic_weights[model] * bonus
            
            # Normalize adjusted weights
            return normalize_weights(adjusted_weights)
            
        except Exception as e:
            logger.warning(f"Error calculating diversity bonus: {str(e)}. Using basic weights.")
            return basic_weights
            
    except Exception as e:
        logger.error(f"Error calculating ensemble weights: {str(e)}")
        logger.debug(traceback.format_exc())
        return get_default_weights()

def calculate_diversity_bonus(model_names: List[str]) -> Dict[str, float]:
    """
    Calculate diversity bonus for each model based on prediction correlation.
    
    Args:
        model_names: List of model names
        
    Returns:
        Dictionary of diversity bonuses (higher for more diverse models)
    """
    # This is a placeholder implementation - in a real system, we would
    # analyze actual prediction correlations to determine diversity
    
    # Default bonuses based on model type
    default_bonuses = {
        'lstm': 1.1,        # Neural networks get bonus for diversity
        'holtwinters': 1.2, # Statistical models get higher bonus
        'linear': 1.0,      # Basic models get no bonus
        'xgboost': 0.9,     # Tree models tend to be similar
        'lightgbm': 0.9,
        'knn': 1.2,         # Instance-based gets bonus for diversity
        'gradient_boosting': 0.9,
        'catboost': 0.9,
        'cnn_lstm': 1.1,
        'autoencoder': 1.3, # Unique approach gets highest bonus
        'meta': 1.0         # Meta model gets no bonus
    }
    
    return {model: default_bonuses.get(model, 1.0) for model in model_names}

if __name__ == "__main__":
    # Example usage when run directly
    try:
        # Simple test of metrics calculation
        predictions = np.array([
            [1, 2, 3, 4, 5, 6],
            [10, 20, 30, 40, 50, 59],
            [7, 14, 21, 28, 35, 42]
        ])
        
        actual = np.array([
            [1, 3, 5, 7, 9, 11],
            [10, 20, 30, 40, 50, 60],
            [7, 14, 21, 28, 35, 49]
        ])
        
        print("Testing metrics calculation...")
        metrics = calculate_metrics(predictions, actual)
        print("Metrics:", json.dumps(metrics, indent=2))
        
        # Test model weights
        print("\nTesting model weights calculation...")
        weights = get_model_weights()
        print("Model weights:", json.dumps(weights, indent=2))
        
        # Test performance summary
        print("\nTesting performance summary...")
        summary = get_model_performance_summary()
        print("Summary:", json.dumps(summary, indent=2))
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc() 