import numpy as np
from typing import List, Union
import logging

def ensure_valid_prediction(prediction: Union[List[int], np.ndarray], min_number: int = 1, max_number: int = 59, n_numbers: int = 6) -> List[int]:
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