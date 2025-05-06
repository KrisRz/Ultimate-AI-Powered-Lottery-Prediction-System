"""
Consolidated prediction utilities for lottery prediction system.

This module combines functionality from predict_numbers.py and predict_next_draw.py,
providing comprehensive tools for generating, validating, and saving lottery predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
import json
from datetime import datetime
from pathlib import Path
import os
import time
import traceback
import random
import matplotlib.pyplot as plt
import pickle

# Setup logging
logger = logging.getLogger(__name__)

def generate_next_draw_predictions(
    models: Dict,
    last_data: Union[pd.DataFrame, np.ndarray],
    n_predictions: int = 10,
    min_number: int = 1,
    max_number: int = 59,
    n_numbers: int = 6,
    weights: Optional[Dict[str, float]] = None
) -> List[List[int]]:
    """
    Generate predictions for next lottery draw using ensemble of models.
    
    Args:
        models: Dictionary of trained models
        last_data: DataFrame with recent lottery data or numpy array with preprocessed sequences
        n_predictions: Number of predictions to generate
        min_number: Minimum valid lottery number
        max_number: Maximum valid lottery number
        n_numbers: Number of numbers in each prediction
        weights: Optional dictionary of model weights
        
    Returns:
        List of prediction lists
    """
    predictions = []
    
    # Get model weights if not provided
    if weights is None:
        weights = get_model_weights(models.keys())
    
    # Generate predictions from each model
    for _ in range(n_predictions):
        ensemble_prediction = np.zeros(n_numbers)
        total_weight = 0.0
        
        try:
            for model_name, model in models.items():
                try:
                    # Use the predict_with_model function to get prediction for this model
                    pred = predict_with_model(model_name, model, last_data)
                    
                    # Skip if prediction failed
                    if pred is None:
                        logger.warning(f"Model {model_name} returned None prediction, skipping")
                        continue
                        
                    pred = np.array(pred)  # Convert to numpy array
                    
                    # Validate and normalize prediction
                    if pred.size == 0:
                        logger.warning(f"Model {model_name} returned empty prediction, skipping")
                        continue
                        
                    # Ensure prediction has correct shape
                    if len(pred.shape) > 1:
                        pred = pred[0]  # Take first prediction if multiple
                        
                    # Normalize to [0, 1] range if needed
                    if np.max(pred) > 1.0:
                        pred = pred / max_number
                        
                    # Make sure prediction has expected length
                    if len(pred) != n_numbers:
                        logger.warning(f"Model {model_name} returned prediction with wrong length {len(pred)}, expected {n_numbers}")
                        # Pad or truncate to correct length
                        if len(pred) < n_numbers:
                            pred = np.pad(pred, (0, n_numbers - len(pred)), 'constant')
                        else:
                            pred = pred[:n_numbers]
                    
                    # Apply model weight
                    model_weight = weights.get(model_name, 0)
                    if model_weight > 0:
                        ensemble_prediction += pred * model_weight
                        total_weight += model_weight
                        logger.debug(f"Added prediction from {model_name} with weight {model_weight}")
                
                except Exception as model_error:
                    logger.error(f"Error getting prediction from {model_name}: {str(model_error)}")
                    continue  # Skip this model and continue with others
            
            # Normalize if some models failed
            if total_weight > 0 and total_weight < 1.0:
                ensemble_prediction /= total_weight
                
            # Round and clip predictions
            # Scale back to original range
            numbers = np.round(ensemble_prediction * max_number).astype(int)
            numbers = np.clip(numbers, min_number, max_number)
            
            # Ensure unique numbers
            unique_numbers = []
            for num in numbers:
                if num not in unique_numbers and min_number <= num <= max_number:
                    unique_numbers.append(num)
            
            # If we don't have enough unique numbers, add random ones
            while len(unique_numbers) < n_numbers:
                new_number = np.random.randint(min_number, max_number + 1)
                if new_number not in unique_numbers:
                    unique_numbers.append(new_number)
            
            # Sort numbers
            unique_numbers.sort()
            predictions.append(unique_numbers)
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            # Generate a random prediction as fallback
            fallback = list(sorted(np.random.choice(range(min_number, max_number + 1), n_numbers, replace=False)))
            predictions.append(fallback)
    
    return predictions

def predict_with_model(model_name: str, model: Any, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Generate prediction using a specific model.
    
    Args:
        model_name: Name of the model to use
        model: Trained model object
        data: Input data for prediction
        
    Returns:
        Prediction array
    """
    try:
        # Handle different model types
        if model_name == 'lstm':
            # Ensure data is in correct format for LSTM
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to numpy array
                if 'Main_Numbers' in data.columns:
                    sequences = np.array(data['Main_Numbers'].tolist())
                    # Normalize between 0 and 1
                    sequences = sequences / 59.0  # Assuming max number is 59
                    # Reshape for LSTM input (samples, timesteps, features)
                    if len(sequences.shape) == 2:  # Already (timesteps, features)
                        sequences = np.expand_dims(sequences, axis=0)  # Add batch dimension
                    prediction = model.predict(sequences)
                else:
                    raise ValueError("DataFrame must contain 'Main_Numbers' column")
            else:
                # Ensure numpy array has correct shape for LSTM
                if len(data.shape) == 2:  # (timesteps, features)
                    data = np.expand_dims(data, axis=0)  # Add batch dimension
                # Assume data is already properly formatted for LSTM
                prediction = model.predict(data)
            
        elif model_name in ['xgboost', 'lightgbm', 'catboost']:
            # Ensure data is in correct format for tree-based models (2D)
            if isinstance(data, pd.DataFrame):
                # Extract features
                X = extract_features(data)
                prediction = model.predict(X)
            else:
                # Flatten 3D data to 2D if needed
                if len(data.shape) == 3:
                    X_2d = data.reshape(data.shape[0], -1)
                else:
                    X_2d = data
                # Assume data is already properly formatted
                prediction = model.predict(X_2d)
                
        elif model_name == 'ensemble':
            # Ensemble model has its own prediction method
            prediction = model.predict(data)
            
        else:
            # Generic model prediction
            prediction = model.predict(data)
        
        # Ensure prediction is in correct format
        if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
            prediction = prediction[0]  # Take first prediction if multiple
            
        # Ensure the prediction has the correct format (array of numbers)
        return np.array(prediction)
        
    except Exception as e:
        logger.error(f"Error predicting with {model_name} model: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return None to indicate failure
        return None

def extract_features(data: pd.DataFrame, lookback: int = 5) -> np.ndarray:
    """
    Extract features from lottery data for prediction.
    
    Args:
        data: DataFrame with lottery data
        lookback: Number of previous draws to use for features
        
    Returns:
        Feature array
    """
    try:
        # Get the most recent draws
        recent_draws = data.tail(lookback)
        
        # Extract Main_Numbers
        if 'Main_Numbers' not in recent_draws.columns:
            raise ValueError("DataFrame must contain 'Main_Numbers' column")
            
        # Convert to numpy array
        numbers = np.array(recent_draws['Main_Numbers'].tolist())
        
        # Create feature vector (flattened sequence)
        features = numbers.flatten()
        
        # Add derived features
        for i in range(lookback):
            row = numbers[i]
            # Add mean, std, min, max
            features = np.append(features, [np.mean(row), np.std(row), np.min(row), np.max(row)])
            
        return features.reshape(1, -1)  # Reshape for prediction
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return empty feature array as fallback
        return np.zeros((1, lookback * 10))

def format_predictions(predictions: List[List[int]]) -> str:
    """
    Format predictions for display.
    
    Args:
        predictions: List of prediction lists
        
    Returns:
        Formatted string
    """
    result = "Next Draw Predictions:\n\n"
    for i, pred in enumerate(predictions, 1):
        result += f"Prediction {i}: {', '.join(map(str, pred))}\n"
    return result 

def validate_prediction(prediction: List[int], min_val: int = 1, max_val: int = 59, n_numbers: int = 6) -> Tuple[bool, str]:
    """
    Validate a single prediction.
    
    Args:
        prediction: List of predicted numbers
        min_val: Minimum valid value
        max_val: Maximum valid value
        n_numbers: Required number of numbers
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if prediction is a list or array
        if not isinstance(prediction, (list, np.ndarray)):
            return False, f"Prediction must be a list or array, got {type(prediction)}"
            
        # Check length
        if len(prediction) != n_numbers:
            return False, f"Prediction must contain {n_numbers} numbers, got {len(prediction)}"
            
        # Check if all elements are integers
        if not all(isinstance(num, (int, np.integer)) for num in prediction):
            return False, "All elements must be integers"
            
        # Check if all elements are within valid range
        if not all(min_val <= num <= max_val for num in prediction):
            return False, f"All numbers must be between {min_val} and {max_val}"
            
        # Check for duplicates
        if len(set(prediction)) != len(prediction):
            return False, "Prediction must not contain duplicate numbers"
            
        return True, ""
        
    except Exception as e:
        return False, f"Error validating prediction: {str(e)}"

def ensure_valid_prediction(prediction: List[int], min_val: int = 1, max_val: int = 59, n_numbers: int = 6) -> List[int]:
    """
    Ensure prediction contains unique integers in the valid range.
    
    Args:
        prediction: List of predicted numbers
        min_val: Minimum valid value
        max_val: Maximum valid value
        n_numbers: Required number of numbers
        
    Returns:
        Valid prediction with unique integers
    """
    try:
        # Check if prediction is already valid
        is_valid, _ = validate_prediction(prediction, min_val, max_val, n_numbers)
        if is_valid:
            return sorted(prediction)
        
        # If prediction has some valid numbers, try to keep them
        valid_numbers = []
        if isinstance(prediction, (list, np.ndarray)):
            for num in prediction:
                if isinstance(num, (int, np.integer)) and min_val <= num <= max_val and num not in valid_numbers:
                    valid_numbers.append(int(num))
        
        # Add random numbers if needed
        while len(valid_numbers) < n_numbers:
            num = random.randint(min_val, max_val)
            if num not in valid_numbers:
                valid_numbers.append(num)
        
        return sorted(valid_numbers[:n_numbers])
        
    except Exception as e:
        logger.error(f"Error ensuring valid prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Generate completely random prediction as fallback
        valid_numbers = set()
        while len(valid_numbers) < n_numbers:
            valid_numbers.add(random.randint(min_val, max_val))
        return sorted(list(valid_numbers))

def validate_predictions(predictions: List[List[int]], min_val: int = 1, max_val: int = 59, n_numbers: int = 6) -> Tuple[List[List[int]], List[int]]:
    """
    Validate a list of predictions, replacing invalid ones.
    
    Args:
        predictions: List of prediction lists
        min_val: Minimum valid value
        max_val: Maximum valid value
        n_numbers: Required number of numbers
        
    Returns:
        Tuple of (validated_predictions, invalid_indices)
    """
    validated = []
    invalid_indices = []
    
    for i, pred in enumerate(predictions):
        is_valid, error_msg = validate_prediction(pred, min_val, max_val, n_numbers)
        if is_valid:
            validated.append(sorted(pred))
        else:
            logger.warning(f"Invalid prediction at index {i}: {error_msg}. Replacing with valid prediction.")
            validated.append(ensure_valid_prediction(pred, min_val, max_val, n_numbers))
            invalid_indices.append(i)
    
    return validated, invalid_indices

def calculate_prediction_metrics(predictions: List[List[int]], data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for predictions against the most recent actual draw.
    
    Args:
        predictions: List of prediction lists
        data: DataFrame with historical lottery data
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Get the most recent actual draw
        if 'Main_Numbers' not in data.columns:
            logger.error("No 'Main_Numbers' column in data")
            return {'error': 'No Main_Numbers column in data'}
        
        last_draw = data['Main_Numbers'].iloc[-1]
        
        # Create numpy arrays for metrics calculation
        pred_array = np.array(predictions)
        actual_array = np.array([last_draw] * len(predictions))
        
        # Calculate metrics
        metrics = calculate_metrics(pred_array, actual_array)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating prediction metrics: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'accuracy': 0.0,
            'match_rate': 0.0,
            'error': str(e)
        }

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for predictions against actuals.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        
    Returns:
        Dictionary of metrics
    """
    try:
        metrics = {}
        
        # Calculate exact match rate
        exact_matches = 0
        for pred, actual in zip(predictions, actuals):
            if np.array_equal(np.sort(pred), np.sort(actual)):
                exact_matches += 1
        metrics['exact_match_rate'] = exact_matches / len(predictions) if len(predictions) > 0 else 0
        
        # Calculate partial match rates
        partial_3plus = 0
        partial_4plus = 0
        partial_5plus = 0
        match_counts = []
        
        for pred, actual in zip(predictions, actuals):
            # Count matching numbers
            matching = len(set(pred).intersection(set(actual)))
            match_counts.append(matching)
            
            if matching >= 3:
                partial_3plus += 1
            if matching >= 4:
                partial_4plus += 1
            if matching >= 5:
                partial_5plus += 1
        
        metrics['partial_3plus_rate'] = partial_3plus / len(predictions) if len(predictions) > 0 else 0
        metrics['partial_4plus_rate'] = partial_4plus / len(predictions) if len(predictions) > 0 else 0
        metrics['partial_5plus_rate'] = partial_5plus / len(predictions) if len(predictions) > 0 else 0
        metrics['avg_matches'] = sum(match_counts) / len(match_counts) if len(match_counts) > 0 else 0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'error': str(e),
            'exact_match_rate': 0.0,
            'partial_3plus_rate': 0.0
        }

def get_model_weights(model_names: List[str]) -> Dict[str, float]:
    """
    Get weights for each model for ensemble prediction.
    
    Args:
        model_names: List of model names
        
    Returns:
        Dictionary mapping model names to weights
    """
    # Default weights if performance history not available
    default_weights = {
        'lstm': 0.3,
        'xgboost': 0.25,
        'lightgbm': 0.25,
        'cnn_lstm': 0.2,
        'ensemble': 1.0
    }
    
    # Initialize weights
    weights = {}
    for name in model_names:
        weights[name] = default_weights.get(name, 0.1)
    
    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        for name in weights:
            weights[name] /= total
    
    return weights

def save_predictions(
    predictions: List[List[int]], 
    metrics: Optional[Dict[str, float]] = None, 
    output_path: str = 'outputs/predictions/predictions.json',
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save predictions and metrics to a JSON file.
    
    Args:
        predictions: List of prediction lists
        metrics: Optional dictionary of metrics
        output_path: Path to save the predictions
        metadata: Optional additional metadata to include
        
    Returns:
        True if successful, False otherwise
    """
    try:
        start_time = time.time()
        
        # Prepare output data
        output = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'count': len(predictions)
        }
        
        # Add metrics if provided
        if metrics:
            output['metrics'] = metrics
        
        # Add metadata if provided
        if metadata:
            output['metadata'] = metadata
        
        # Create directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        duration = time.time() - start_time
        logger.info(f"Saved {len(predictions)} predictions to {output_file} in {duration:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving predictions to {output_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def load_predictions(input_path: str = 'outputs/predictions/predictions.json') -> Dict[str, Any]:
    """
    Load predictions from a JSON file.
    
    Args:
        input_path: Path to the predictions file
        
    Returns:
        Dictionary with predictions and metrics
    """
    try:
        input_file = Path(input_path)
        if not input_file.exists():
            logger.warning(f"Predictions file not found: {input_path}")
            return {'error': 'File not found', 'predictions': []}
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data.get('predictions', []))} predictions from {input_path}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading predictions from {input_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return {'error': str(e), 'predictions': []}

def validate_and_save_predictions(
    predictions: List[List[int]], 
    data: pd.DataFrame, 
    output_path: str = 'outputs/predictions/predictions.json',
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate predictions, calculate metrics, and save to file.
    
    Args:
        predictions: List of prediction lists
        data: DataFrame with historical lottery data
        output_path: Path to save the predictions
        metadata: Optional additional metadata to include
        
    Returns:
        Dictionary with validated predictions and metrics
    """
    try:
        start_time = time.time()
        logger.info(f"Processing {len(predictions)} predictions...")
        
        # Validate predictions
        validated_predictions, invalid_indices = validate_predictions(predictions)
        if invalid_indices:
            logger.warning(f"Fixed {len(invalid_indices)} invalid predictions at indices: {invalid_indices}")
        
        # Calculate metrics
        metrics = calculate_prediction_metrics(validated_predictions, data)
        
        # Save to file
        success = save_predictions(
            validated_predictions, 
            metrics, 
            output_path,
            metadata
        )
        
        if not success:
            logger.warning("Failed to save predictions to file")
        
        # Prepare return data
        result = {
            'predictions': validated_predictions,
            'metrics': metrics,
            'invalid_count': len(invalid_indices),
            'success': success
        }
        
        # Add formatted display
        result['display'] = format_predictions_for_display(validated_predictions, metrics)
        
        duration = time.time() - start_time
        logger.info(f"Processed predictions in {duration:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in validate_and_save_predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'predictions': [ensure_valid_prediction([]) for _ in range(len(predictions) if predictions else 10)],
            'metrics': {'error': str(e)},
            'invalid_count': len(predictions) if predictions else 0,
            'success': False,
            'error': str(e)
        }

def format_predictions_for_display(
    predictions: List[List[int]],
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Lottery Predictions"
) -> str:
    """
    Format predictions for display in console or UI.
    
    Args:
        predictions: List of prediction lists
        metrics: Optional dictionary of metrics
        title: Title for the display
        
    Returns:
        Formatted string
    """
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"\n{'-'*50}\n{title.upper()}\n{'-'*50}\n"
        formatted += f"Generated: {now}\n"
        formatted += f"Number of predictions: {len(predictions)}\n\n"
        
        # Add predictions
        for i, pred in enumerate(predictions, 1):
            formatted += f"Prediction {i}: {' - '.join(f'{num:02d}' for num in sorted(pred))}\n"
        
        # Add metrics if provided
        if metrics:
            formatted += f"\n{'-'*20} METRICS {'-'*20}\n"
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
                else:
                    formatted += f"{metric.replace('_', ' ').title()}: {value}\n"
        
        formatted += f"\n{'-'*50}\n"
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        return f"Error formatting predictions: {str(e)}"

def plot_predictions(predictions: List[List[int]], output_path: str = 'outputs/visualizations/predictions.png') -> bool:
    """
    Create a visualization of prediction frequencies.
    
    Args:
        predictions: List of prediction lists
        output_path: Path to save the plot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Flatten predictions and count frequencies
        all_numbers = [num for pred in predictions for num in pred]
        max_number = max(all_numbers)
        
        # Count frequencies
        frequencies = np.zeros(max_number + 1)
        for num in all_numbers:
            frequencies[num] += 1
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, max_number + 1), frequencies[1:])
        plt.title('Prediction Frequencies')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(range(1, max_number + 1, 5))
        
        # Create directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved prediction frequencies plot to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating prediction plot: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

# Define constants
max_number = 59  # Maximum lottery number
n_numbers = 6    # Number of numbers in a prediction 

# Additional functions merged from prediction.py
# ======================================================================

def __call__(self, df: pd.DataFrame, models: Dict[str, Any]) -> Union[List[int], np.ndarray]: ...



def recover_failed_model(model_name: str, df: pd.DataFrame) -> Optional[Any]:
    """
    Attempt to recover a failed model
    
    Args:
        model_name: Name of the failed model
        df: Training data
        
    Returns:
        Recovered model or None
    """
    try:
        # Try to load from checkpoint
        model_path = f"models/checkpoints/{model_name}_latest.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Recovered {model_name} from checkpoint")
            return model
            
        # If no checkpoint, try retraining
        if model_name == 'lstm':
            from models.lstm_model import train_lstm_model
            model = train_lstm_model(df)
        elif model_name == 'xgboost':
            from models.xgboost_model import train_xgboost_model
            model = train_xgboost_model(df)
        # Add other model types...
        
        if model is not None:
            logging.info(f"Successfully retrained {model_name}")
            return model
            
    except Exception as e:
        logging.error(f"Failed to recover {model_name}: {str(e)}")
        
    return None



def generate_optimized_predictions(df: pd.DataFrame, models: Dict, n_predictions: int = 10) -> List[List[int]]:
    """Generate optimized predictions using multiple models and scoring."""
    predictions = []
    for _ in range(n_predictions):
        model_predictions = []
        for model_name, model in models.items():
            if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
                pred = model[0].predict(df)
            else:
                pred = model.predict(df)
            model_predictions.append(pred)
        
        # Score and select best predictions
        scored = score_combinations(model_predictions, df)
        best_pred = max(scored, key=lambda x: x[1])[0]
        predictions.append(ensure_valid_prediction(best_pred))
    
    return predictions



def score_combinations(predictions: List[List[int]], df: pd.DataFrame) -> List[Tuple[List[int], float]]:
    """Score combinations of predictions based on historical patterns."""
    scored = []
    for pred in predictions:
        score = 0
        
        # Check number properties
        score += sum(1 for n in pred if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])  # Primes
        score += sum(1 for n in pred if n % 2 == 1)  # Odds
        score += sum(1 for n in pred if n <= 30)  # Low numbers
        
        # Check historical patterns
        all_numbers = df['Main_Numbers'].tolist()
        window_size = min(50, len(all_numbers))
        window_numbers = all_numbers[-window_size:]
        
        # Check pair frequency
        pairs = list(combinations(pred, 2))
        pair_freq = sum(1 for p in pairs for draw in window_numbers if p[0] in draw and p[1] in draw)
        score += pair_freq / len(window_numbers)
        
        # Check triplet frequency
        triplets = list(combinations(pred, 3))
        triplet_freq = sum(1 for t in triplets for draw in window_numbers if t[0] in draw and t[1] in draw and t[2] in draw)
        score += triplet_freq / len(window_numbers)
        
        scored.append((pred, score))
    
    return scored



def backtest(df: pd.DataFrame, models: Dict = None) -> Tuple[float, List[List[int]]]:
    """Backtest models on historical data."""
    if models is None:
        models = update_models(df)
    
    predictions = []
    actual = df['Main_Numbers'].tolist()
    
    for i in range(len(df) - 1):
        train_df = df.iloc[:i+1]
        test_df = df.iloc[i+1:i+2]
        
        if i % 10 == 0:  # Retrain models every 10 draws
            models = update_models(train_df)
        
        pred = predict_next_draw(train_df, models)
        predictions.append(list(pred.values())[0])  # Use first model's prediction
    
    accuracy = calculate_prediction_accuracy(predictions, actual[1:])
    return accuracy, predictions



def calculate_prediction_accuracy(predictions: List[List[int]], actual: List[int]) -> float:
    """Calculate prediction accuracy."""
    correct = 0
    total = 0
    
    for pred, act in zip(predictions, actual):
        matches = len(set(pred) & set(act))
        correct += matches
        total += 6
    
    return correct / total if total > 0 else 0 