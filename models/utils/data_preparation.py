from typing import Tuple
import pandas as pd
import numpy as np

def prepare_sequence_data(data: pd.DataFrame, sequence_length: int = 3, target_col: str = 'winning_numbers') -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequence data for time series models.
    
    Args:
        data: DataFrame containing lottery data
        sequence_length: Length of sequences to create
        target_col: Column containing target values
        
    Returns:
        Tuple of (X, y) arrays for sequence modeling
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if target_col not in data.columns:
        raise ValueError(f"target_col '{target_col}' not found in data")
    if len(data) <= sequence_length:
        raise ValueError("Not enough data points for the requested sequence length")

    # Validate date column
    if 'Draw Date' in data.columns:
        try:
            pd.to_datetime(data['Draw Date'])
        except (ValueError, TypeError):
            raise ValueError("Invalid date format in Draw Date column")

    # Convert winning numbers to sequences
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[target_col].iloc[i:i+sequence_length].values
        target = data[target_col].iloc[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(targets) 