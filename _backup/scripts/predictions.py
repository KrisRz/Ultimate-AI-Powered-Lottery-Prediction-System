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
    last_data: pd.DataFrame,
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
        last_data: DataFrame with recent lottery data
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
        
        for model_name, model in models.items():
            # Use the predict_with_model function to get prediction for this model
            pred = predict_with_model(model_name, model, last_data)
            pred = np.array(pred)  # Convert to numpy array
            
            # Apply model weight
            model_weight = weights.get(model_name, 0)
            ensemble_prediction += pred * model_weight
        
        # Round and clip predictions
        numbers = np.round(ensemble_prediction).astype(int)
        numbers = np.clip(numbers, min_number, max_number)
        
        # Ensure unique numbers
        numbers = np.unique(numbers)
        while len(numbers) < n_numbers:
            new_number = np.random.randint(min_number, max_number + 1)
            if new_number not in numbers:
                numbers = np.append(numbers, new_number)
        
        # Sort numbers
        numbers = np.sort(numbers)
        predictions.append(numbers.tolist())
    
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
                    sequences = sequences / max_number
                    prediction = model.predict(np.expand_dims(sequences[-1], axis=0))
                else:
                    raise ValueError("DataFrame must contain 'Main_Numbers' column")
            else:
                # Assume data is already properly formatted for LSTM
                prediction = model.predict(data)
            
        elif model_name in ['xgboost', 'lightgbm', 'catboost']:
            # Ensure data is in correct format for tree-based models
            if isinstance(data, pd.DataFrame):
                # Extract features
                X = extract_features(data)
                prediction = model.predict(X)
            else:
                # Assume data is already properly formatted
                prediction = model.predict(data)
                
        elif model_name == 'ensemble':
            # Ensemble model has its own prediction method
            prediction = model.predict(data)
            
        else:
            # Generic model prediction
            prediction = model.predict(data)
        
        # Ensure prediction is in correct format
        if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
            prediction = prediction[0]  # Take first prediction if multiple
            
        # Convert prediction to correct range (1-max_number)
        if np.max(prediction) <= 1.0:  # Normalized prediction
            prediction = np.round(prediction * max_number).astype(int)
            
        return prediction
        
    except Exception as e:
        logger.error(f"Error predicting with {model_name} model: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return random prediction as fallback
        return np.sort(np.random.choice(range(1, max_number + 1), n_numbers, replace=False))

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