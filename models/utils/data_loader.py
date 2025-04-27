from typing import List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Class for loading and preprocessing lottery data."""
    
    def __init__(self, data_path: Union[str, Path]):
        """Initialize DataLoader with path to data file.
        
        Args:
            data_path: Path to CSV file containing lottery data
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        if self.data_path.suffix.lower() != '.csv':
            raise ValueError("Data file must be a CSV file")
            
    def load_data(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load lottery data from CSV file.
        
        Args:
            columns: Optional list of columns to load. If None, loads all columns.
            
        Returns:
            DataFrame containing lottery data
        """
        try:
            if columns:
                data = pd.read_csv(self.data_path, usecols=columns)
            else:
                data = pd.read_csv(self.data_path)
                
            if data.empty:
                raise ValueError("Data file is empty")
                
            return data
            
        except pd.errors.EmptyDataError:
            raise ValueError("Data file is empty")
        except pd.errors.ParserError:
            raise ValueError("Invalid CSV file format")
            
    def get_winning_numbers(self) -> np.ndarray:
        """Extract winning numbers from data.
        
        Returns:
            Array of winning number sequences
        """
        data = self.load_data(['winning_numbers'])
        if 'winning_numbers' not in data.columns:
            raise ValueError("winning_numbers column not found in data")
            
        return data['winning_numbers'].values 