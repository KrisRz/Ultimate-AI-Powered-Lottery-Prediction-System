import numpy as np
from typing import List, Union, Tuple
import logging

def ensure_valid_prediction(prediction: Union[List[int], Tuple[int, ...]], 
                           numbers_count: int = 6, 
                           min_value: int = 1, 
                           max_value: int = 59) -> List[int]:
    """
    Ensure prediction is a valid lottery prediction.
    
    Args:
        prediction: List or tuple of predicted numbers
        numbers_count: Expected count of numbers (default: 6)
        min_value: Minimum valid number (default: 1)
        max_value: Maximum valid number (default: 59)
        
    Returns:
        List of validated and sorted lottery numbers
        
    Raises:
        ValueError: If the prediction is invalid
    """
    # Convert to list if it's a tuple
    if isinstance(prediction, tuple):
        prediction = list(prediction)
    
    # Check if it's a list
    if not isinstance(prediction, list):
        raise ValueError(f"Prediction must be a list or tuple, got {type(prediction).__name__}")
    
    # Check length
    if len(prediction) != numbers_count:
        raise ValueError(f"Prediction must contain exactly {numbers_count} numbers, got {len(prediction)}")
    
    # Check all are integers
    if not all(isinstance(n, int) for n in prediction):
        raise ValueError("All predicted numbers must be integers")
    
    # Check range
    if not all(min_value <= n <= max_value for n in prediction):
        out_of_range = [n for n in prediction if not (min_value <= n <= max_value)]
        raise ValueError(f"All numbers must be between {min_value} and {max_value}, got {out_of_range}")
    
    # Check for duplicates
    if len(set(prediction)) != len(prediction):
        duplicates = [n for n in prediction if prediction.count(n) > 1]
        raise ValueError(f"Prediction contains duplicate numbers: {duplicates}")
    
    # Return sorted prediction
    return sorted(prediction)

def ensure_valid_prediction_old(prediction: Union[List[int], np.ndarray], min_number: int = 1, max_number: int = 59, n_numbers: int = 6) -> List[int]:
    """
    Ensure prediction is a valid list of lottery numbers.
    If not valid, generate a valid list of numbers.
    """
    try:
        # Convert to list if numpy array
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        
        # Ensure we have integers
        prediction = [int(round(x)) for x in prediction]
        
        # Clip values to valid range
        prediction = [max(min(x, max_number), min_number) for x in prediction]
        
        # Ensure unique numbers
        prediction = list(set(prediction))
        
        # If we don't have enough numbers, add random ones
        while len(prediction) < n_numbers:
            new_number = np.random.randint(min_number, max_number + 1)
            if new_number not in prediction:
                prediction.append(new_number)
        
        # If we have too many numbers, take first n_numbers
        if len(prediction) > n_numbers:
            prediction = prediction[:n_numbers]
        
        # Sort numbers
        prediction.sort()
        
        return prediction
    except Exception as e:
        logging.error(f"Error in ensure_valid_prediction: {str(e)}")
        # If anything goes wrong, return random valid numbers
        return sorted(np.random.choice(range(min_number, max_number + 1), size=n_numbers, replace=False).tolist()) 