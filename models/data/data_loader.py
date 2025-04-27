import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from models.utils.validation import validate_prediction_format

class DataLoader:
    """Handles loading and processing of lottery data from CSV files."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the DataLoader with a path to the data file.
        
        Args:
            file_path: Path to the CSV file containing lottery data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is not a CSV
        """
        self.file_path = Path(file_path)
        self.data = None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix.lower() != ".csv":
            raise ValueError("File must be a CSV")
    
    def load_data(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data from the CSV file.
        
        Args:
            columns: Optional list of columns to load. If None, loads all columns.
            
        Returns:
            DataFrame containing the lottery data
        """
        self.data = pd.read_csv(self.file_path, usecols=columns)
        return self.data
    
    def get_winning_numbers(self, return_dates: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, pd.Series]]:
        """
        Get the winning numbers from the data.
        
        Args:
            return_dates: If True, also returns the dates for each set of numbers
            
        Returns:
            If return_dates is False, returns numpy array of winning numbers
            If return_dates is True, returns tuple of (winning numbers array, dates series)
            
        Raises:
            ValueError: If any winning numbers are invalid
        """
        if self.data is None:
            self.load_data()
            
        # Convert string representation of list to numpy array
        numbers = np.array([eval(x) for x in self.data['winning_numbers']])
        
        # Validate each set of numbers
        for i, nums in enumerate(numbers):
            is_valid, error = validate_prediction_format(nums)
            if not is_valid:
                raise ValueError(f"Invalid winning numbers at index {i}: {error}")
        
        if return_dates:
            dates = pd.to_datetime(self.data['date'])
            return numbers, dates
        return numbers
    
    def get_bonus_numbers(self) -> np.ndarray:
        """
        Get the bonus numbers from the data.
        
        Returns:
            Numpy array of bonus numbers
        """
        if self.data is None:
            self.load_data()
        return self.data['bonus_number'].to_numpy()
    
    def get_prize_pools(self) -> np.ndarray:
        """
        Get the prize pools from the data.
        
        Returns:
            Numpy array of prize pools
        """
        if self.data is None:
            self.load_data()
        return self.data['prize_pool'].to_numpy() 