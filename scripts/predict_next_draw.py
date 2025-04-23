import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path
import os
import sys
import time
import traceback

# Try to import from project modules with multiple approaches
try:
    # Try absolute imports first
    from scripts.performance_tracking import get_model_weights
    from scripts.model_bridge import predict_with_model
    from scripts.utils import setup_logging
    from scripts.data_validation import validate_prediction
    from scripts.performance_tracking import calculate_metrics
except ImportError:
    try:
        # Try relative imports
        from .performance_tracking import get_model_weights
        from .model_bridge import predict_with_model
        from .utils import setup_logging
        from .data_validation import validate_prediction
        from .performance_tracking import calculate_metrics
    except ImportError:
        # Default implementations if imports fail
        def setup_logging():
            logging.basicConfig(filename='lottery.log', level=logging.INFO,
                               format='%(asctime)s - %(levelname)s - %(message)s')
        
        def validate_prediction(pred):
            """Simple prediction validation"""
            if not isinstance(pred, list) or len(pred) != 6:
                return False, "Prediction must be a list of 6 numbers"
            if not all(isinstance(n, int) and 1 <= n <= 59 for n in pred):
                return False, "All numbers must be integers between 1 and 59"
            if len(set(pred)) != 6:
                return False, "Numbers must be unique"
            return True, ""
        
        def calculate_metrics(predictions, actual):
            """Simple metrics calculation"""
            metrics = {'accuracy': 0.0, 'match_rate': 0.0}
            return metrics
            
        def get_model_weights():
            """Default model weights"""
            return {'lstm': 0.2, 'xgboost': 0.2, 'lightgbm': 0.2, 'meta': 0.4}
            
        def predict_with_model(model_name, model, data):
            """Simple prediction function"""
            import random
            return sorted([random.randint(1, 59) for _ in range(6)])

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def generate_next_draw_predictions(
    models: Dict,
    last_data: pd.DataFrame,
    n_predictions: int = 10,
    min_number: int = 1,
    max_number: int = 59,
    n_numbers: int = 6
) -> List[List[int]]:
    """Generate predictions for next lottery draw"""
    predictions = []
    
    # Get model weights
    weights = get_model_weights()
    
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

def format_predictions(predictions: List[List[int]]) -> str:
    """Format predictions for display"""
    result = "Next Draw Predictions:\n\n"
    for i, pred in enumerate(predictions, 1):
        result += f"Prediction {i}: {', '.join(map(str, pred))}\n"
    return result 

def ensure_valid_prediction(prediction: List[int], min_val: int = 1, max_val: int = 59) -> List[int]:
    """
    Ensure prediction contains 6 unique integers in the valid range.
    
    Args:
        prediction: List of predicted numbers
        min_val: Minimum valid value (default: 1)
        max_val: Maximum valid value (default: 59)
        
    Returns:
        Valid prediction with 6 unique integers
    """
    try:
        # Check if prediction is already valid
        is_valid, _ = validate_prediction(prediction)
        if is_valid:
            return sorted(prediction)
        
        # If prediction has some valid numbers, try to keep them
        valid_numbers = []
        if isinstance(prediction, list):
            for num in prediction:
                if isinstance(num, int) and min_val <= num <= max_val and num not in valid_numbers:
                    valid_numbers.append(num)
        
        # Add random numbers if needed
        import random
        while len(valid_numbers) < 6:
            num = random.randint(min_val, max_val)
            if num not in valid_numbers:
                valid_numbers.append(num)
        
        return sorted(valid_numbers)
        
    except Exception as e:
        logger.error(f"Error ensuring valid prediction: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Generate completely random prediction as fallback
        import random
        valid_numbers = set()
        while len(valid_numbers) < 6:
            valid_numbers.add(random.randint(min_val, max_val))
        return sorted(list(valid_numbers))

def validate_predictions(predictions: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
    """
    Validate a list of predictions, replacing invalid ones.
    
    Args:
        predictions: List of prediction lists
        
    Returns:
        Tuple of (validated_predictions, invalid_indices)
    """
    validated = []
    invalid_indices = []
    
    for i, pred in enumerate(predictions):
        is_valid, error_msg = validate_prediction(pred)
        if is_valid:
            validated.append(sorted(pred))
        else:
            logger.warning(f"Invalid prediction at index {i}: {error_msg}. Replacing with valid prediction.")
            validated.append(ensure_valid_prediction(pred))
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

def save_predictions(
    predictions: List[List[int]], 
    metrics: Optional[Dict[str, float]] = None, 
    output_path: str = 'results/predictions.json',
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
        output_file.parent.mkdir(exist_ok=True)
        
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

def load_predictions(input_path: str = 'results/predictions.json') -> Dict[str, Any]:
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

def validate_and_save_predictions(
    predictions: List[List[int]], 
    data: pd.DataFrame, 
    output_path: str = 'results/predictions.json',
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

if __name__ == "__main__":
    # Example usage when run directly
    try:
        import argparse
        from scripts.fetch_data import load_data
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Process lottery predictions')
        parser.add_argument('--input', type=str, help='Path to input predictions JSON file')
        parser.add_argument('--data', type=str, default='data/lottery_data_1995_2025.csv',
                          help='Path to lottery data CSV file')
        parser.add_argument('--output', type=str, default='results/predictions.json',
                          help='Path to output predictions JSON file')
        
        args = parser.parse_args()
        
        # Load data
        print(f"Loading lottery data from {args.data}...")
        lottery_data = load_data(args.data)
        print(f"Loaded {len(lottery_data)} lottery draws")
        
        # Process predictions
        if args.input:
            # Load and process existing predictions
            print(f"Loading predictions from {args.input}...")
            loaded_data = load_predictions(args.input)
            predictions = loaded_data.get('predictions', [])
            
            if not predictions:
                print("No predictions found in input file. Generating random predictions...")
                predictions = [ensure_valid_prediction([]) for _ in range(10)]
        else:
            # Generate random predictions
            print("Generating random test predictions...")
            predictions = [ensure_valid_prediction([]) for _ in range(10)]
        
        # Validate and save
        print(f"Validating {len(predictions)} predictions...")
        result = validate_and_save_predictions(predictions, lottery_data, args.output)
        
        # Display results
        print(result['display'])
        
        if result['invalid_count'] > 0:
            print(f"Warning: Fixed {result['invalid_count']} invalid predictions")
        
        if result['success']:
            print(f"Successfully saved validated predictions to {args.output}")
        else:
            print(f"Failed to save predictions to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc() 