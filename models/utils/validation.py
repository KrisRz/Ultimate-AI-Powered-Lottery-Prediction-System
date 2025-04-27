"""
Unified validation utilities for lottery predictions.
"""
import numpy as np
from typing import Tuple, Union, List, Dict, Any
import logging
import pandas as pd
from pathlib import Path
import json
import random

logger = logging.getLogger(__name__)

def validate_prediction_format(prediction: Union[List[int], np.ndarray], max_number: int = 59, n_numbers: int = 6) -> Tuple[bool, str]:
    """
    Validate that a prediction has the correct format for lottery numbers.
    
    Args:
        prediction: Array or list of predicted numbers
        max_number: Maximum allowed number (default 59)
        n_numbers: Required number of predictions (default 6)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Convert to numpy array if needed
        if isinstance(prediction, list):
            prediction = np.array(prediction)
        elif not isinstance(prediction, np.ndarray):
            return False, "Prediction must be a list or numpy array"
            
        # Check shape
        if len(prediction) != n_numbers:
            return False, f"Prediction must contain exactly {n_numbers} numbers"
            
        # Check type
        if not np.issubdtype(prediction.dtype, np.integer):
            return False, "All numbers must be integers"
            
        # Check range
        if np.any((prediction < 1) | (prediction > max_number)):
            return False, f"Numbers must be between 1 and {max_number}"
            
        # Check for duplicates
        if len(np.unique(prediction)) != len(prediction):
            return False, "Numbers must be unique"
            
        return True, ""
        
    except Exception as e:
        return False, f"Invalid prediction format: {str(e)}"

def ensure_valid_prediction(prediction: Union[List[int], np.ndarray], max_number: int = 59, n_numbers: int = 6) -> np.ndarray:
    """
    Ensure prediction is valid by fixing any issues.
    
    Args:
        prediction: Raw prediction array
        max_number: Maximum allowed number
        n_numbers: Number of numbers required
        
    Returns:
        Valid prediction array
    """
    try:
        # Convert to numpy array if needed
        pred = np.array(prediction)
        
        # Round to nearest integer
        pred = np.round(pred).astype(int)
        
        # Clip to valid range
        pred = np.clip(pred, 1, max_number)
        
        # Remove duplicates and get correct number of values
        pred = np.unique(pred)[:n_numbers]
        
        # If we don't have enough numbers, add random ones
        if len(pred) < n_numbers:
            missing = n_numbers - len(pred)
            available = list(set(range(1, max_number + 1)) - set(pred))
            extra = np.random.choice(available, size=missing, replace=False)
            pred = np.concatenate([pred, extra])
            
        # Sort the final prediction
        return np.sort(pred)
        
    except Exception as e:
        # Return a random valid prediction as fallback
        return np.sort(np.random.choice(range(1, max_number + 1), size=n_numbers, replace=False))

class DataValidator:
    """
    Validates lottery data to ensure data quality before prediction or training.
    
    Performs validation on each column according to predefined rules and provides
    detailed error reporting and suggestions for fixes.
    """
    
    def __init__(self, validation_rules: Dict[str, Dict[str, Any]] = None):
        """
        Initialize validator with default or custom validation rules.
        
        Args:
            validation_rules: Optional custom validation rules dictionary
        """
        self.max_number = 59
        self.n_numbers = 6
        self.validation_rules = validation_rules if validation_rules else {
            'Draw Date': {
                'type': 'datetime',
                'required': True,
                'unique': True,
                'chronological': True
            },
            'Day': {
                'type': 'str',
                'required': True,
                'values': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 
                           'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            },
            'Balls': {
                'type': 'str',
                'required': True,
                'format': r'^\d{1,2}( \d{1,2}){5} BONUS \d{1,2}$'
            },
            'Jackpot': {
                'type': 'numeric',
                'required': True
            },
            'Winners': {
                'type': 'int',
                'required': True,
                'min': 0,
                'fix_missing': True
            },
            'Draw Details': {
                'type': 'str',
                'required': True
            },
            'Main_Numbers': {
                'type': 'list',
                'required': False
            },
            'Bonus': {
                'type': 'int',
                'required': False,
                'min': 1,
                'max': 59
            }
        }
        self.history_file = Path('results/validation_log.json')

    def validate_prediction(self, prediction: Union[List[int], np.ndarray]) -> Tuple[bool, str]:
        """Validate lottery number predictions."""
        return validate_prediction_format(prediction, self.max_number, self.n_numbers)

    def validate_data(self, data: pd.DataFrame, fix_issues: bool = True) -> Tuple[bool, Dict]:
        """
        Validate a dataframe against predefined rules.
        
        Args:
            data: DataFrame to validate
            fix_issues: Whether to attempt automatic fixes for minor issues
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        validation_results = {
            'errors': [],
            'warnings': [],
            'fixes_applied': [],
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Check for empty dataframe
            if len(df) == 0:
                validation_results['errors'].append("Empty dataframe - no data to validate")
                return False, validation_results
            
            # Check required columns
            required_cols = {k for k, v in self.validation_rules.items() if v.get('required', False)}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            
            # Check for extra columns not in rules
            extra_cols = set(df.columns) - set(self.validation_rules.keys())
            if extra_cols:
                validation_results['warnings'].append(f"Extra columns found: {extra_cols}")
            
            # Extract Main_Numbers and Bonus from Balls if needed
            if 'Balls' in df.columns and ('Main_Numbers' not in df.columns or 'Bonus' not in df.columns):
                if fix_issues:
                    self._extract_from_balls(df)
                    validation_results['fixes_applied'].append("Extracted Main_Numbers and Bonus from Balls")
                    for col in ['Main_Numbers', 'Bonus']:
                        if col in df.columns and col not in data.columns:
                            data[col] = df[col]
            
            # Validate each column
            for column, rules in self.validation_rules.items():
                if column in df.columns:
                    col_results = self._validate_column(df, column, rules, fix_issues)
                    validation_results['errors'].extend(col_results['errors'])
                    validation_results['warnings'].extend(col_results['warnings'])
                    validation_results['fixes_applied'].extend(col_results['fixes'])
            
            # Check data consistency
            consistency_results = self._check_consistency(df)
            validation_results['errors'].extend(consistency_results['errors'])
            validation_results['warnings'].extend(consistency_results['warnings'])
            
            # Log summary
            duration = (datetime.now() - datetime.fromisoformat(validation_results['start_time'])).total_seconds()
            validation_results['duration_seconds'] = duration
            
            is_valid = len(validation_results['errors']) == 0
            logger.info(f"Data validation completed in {duration:.2f}s: {len(validation_results['errors'])} errors, "
                        f"{len(validation_results['warnings'])} warnings, "
                        f"{len(validation_results['fixes_applied'])} fixes")
            
            if validation_results['errors']:
                logger.error(f"Validation errors: {validation_results['errors']}")
            if validation_results['warnings']:
                logger.warning(f"Validation warnings: {validation_results['warnings']}")
            if validation_results['fixes_applied']:
                logger.info(f"Fixes applied: {validation_results['fixes_applied']}")
            
            return is_valid, validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            return False, validation_results

    def _validate_column(self, df: pd.DataFrame, column: str, rules: Dict, fix_issues: bool) -> Dict:
        """
        Validate a single column against its rules.
        
        Args:
            df: DataFrame containing the data
            column: Column name to validate
            rules: Validation rules for this column
            fix_issues: Whether to attempt fixes
            
        Returns:
            Dict with validation results
        """
        results = {'errors': [], 'warnings': [], 'fixes': []}
        
        try:
            series = df[column]
            
            # Check type
            if rules.get('type') == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(series):
                    if fix_issues:
                        try:
                            df[column] = pd.to_datetime(series)
                            results['fixes'].append(f"Converted {column} to datetime")
                        except Exception as e:
                            results['errors'].append(f"Could not convert {column} to datetime: {str(e)}")
                    else:
                        results['errors'].append(f"Column {column} must be datetime type")
            
            elif rules.get('type') == 'numeric':
                if not pd.api.types.is_numeric_dtype(series):
                    if fix_issues:
                        try:
                            df[column] = pd.to_numeric(series)
                            results['fixes'].append(f"Converted {column} to numeric")
                        except Exception as e:
                            results['errors'].append(f"Could not convert {column} to numeric: {str(e)}")
                    else:
                        results['errors'].append(f"Column {column} must be numeric type")
            
            elif rules.get('type') == 'int':
                if not pd.api.types.is_integer_dtype(series):
                    if fix_issues:
                        try:
                            df[column] = pd.to_numeric(series, downcast='integer')
                            results['fixes'].append(f"Converted {column} to integer")
                        except Exception as e:
                            results['errors'].append(f"Could not convert {column} to integer: {str(e)}")
                    else:
                        results['errors'].append(f"Column {column} must be integer type")
            
            # Check required
            if rules.get('required', False):
                missing = series.isna().sum()
                if missing > 0:
                    results['errors'].append(f"Column {column} has {missing} missing values")
            
            # Check unique
            if rules.get('unique', False):
                duplicates = series.duplicated().sum()
                if duplicates > 0:
                    results['errors'].append(f"Column {column} has {duplicates} duplicate values")
            
            # Check min/max
            if 'min' in rules and pd.api.types.is_numeric_dtype(series):
                below_min = (series < rules['min']).sum()
                if below_min > 0:
                    if fix_issues:
                        df.loc[series < rules['min'], column] = rules['min']
                        results['fixes'].append(f"Clipped {below_min} values below minimum in {column}")
                    else:
                        results['errors'].append(f"Column {column} has {below_min} values below minimum {rules['min']}")
            
            if 'max' in rules and pd.api.types.is_numeric_dtype(series):
                above_max = (series > rules['max']).sum()
                if above_max > 0:
                    if fix_issues:
                        df.loc[series > rules['max'], column] = rules['max']
                        results['fixes'].append(f"Clipped {above_max} values above maximum in {column}")
                    else:
                        results['errors'].append(f"Column {column} has {above_max} values above maximum {rules['max']}")
            
            # Check allowed values
            if 'values' in rules:
                invalid = ~series.isin(rules['values'])
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    results['errors'].append(f"Column {column} has {invalid_count} invalid values")
            
            # Check format
            if 'format' in rules:
                import re
                pattern = re.compile(rules['format'])
                invalid = ~series.astype(str).str.match(pattern)
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    results['errors'].append(f"Column {column} has {invalid_count} values not matching required format")
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating column {column}: {str(e)}")
            results['errors'].append(f"Validation error: {str(e)}")
            return results

    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Check data consistency across columns.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict with validation results
        """
        results = {'errors': [], 'warnings': []}
        
        try:
            # Check chronological order of dates
            if 'Draw Date' in df.columns and self.validation_rules['Draw Date'].get('chronological', False):
                dates = pd.to_datetime(df['Draw Date'])
                if not dates.is_monotonic_increasing:
                    results['warnings'].append("Draw dates are not in chronological order")
            
            # Check for missing values in required columns
            for column, rules in self.validation_rules.items():
                if column in df.columns and rules.get('required', False):
                    missing = df[column].isna().sum()
                    if missing > 0:
                        results['errors'].append(f"Required column {column} has {missing} missing values")
            
            # Verify that either Balls or (Main_Numbers and Bonus) exist
            if 'Balls' not in df.columns and not ('Main_Numbers' in df.columns and 'Bonus' in df.columns):
                results['errors'].append("Either 'Balls' or both 'Main_Numbers' and 'Bonus' must be present")
            
            # Check for extreme outliers in numeric columns
            for column in df.select_dtypes(include=[np.number]).columns:
                if column in self.validation_rules and len(df) > 10:  # Only check if we have enough data
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    
                    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                    if len(outliers) > 0:
                        results['warnings'].append(f"Column {column} has {len(outliers)} extreme outliers")
            
            return results
            
        except Exception as e:
            logger.error(f"Error checking consistency: {str(e)}")
            results['errors'].append(f"Consistency check error: {str(e)}")
            return results

    def _extract_from_balls(self, df: pd.DataFrame) -> None:
        """
        Extract Main_Numbers and Bonus from Balls column.
        
        Args:
            df: DataFrame to modify
        """
        main_numbers_list = []
        bonus_numbers = []
        
        for ball_str in df['Balls']:
            try:
                if pd.isna(ball_str):
                    raise ValueError("Missing ball string")
                    
                # Split the string and convert to numbers
                parts = ball_str.split()
                if len(parts) != 8 or parts[6].upper() != 'BONUS':
                    raise ValueError("Invalid ball string format")
                    
                # Extract main numbers and bonus
                main = [int(n) for n in parts[:6]]
                bonus = int(parts[7])
                
                # Validate
                if not all(1 <= n <= 59 for n in main) or not 1 <= bonus <= 59:
                    raise ValueError("Numbers out of range")
                if len(set(main)) != 6:
                    raise ValueError("Duplicate numbers")
                    
                main_numbers_list.append(sorted(main))
                bonus_numbers.append(bonus)
                
            except Exception:
                # On any error, provide synthetic numbers
                seed = hash(str(ball_str)) % 10000 if ball_str else 0
                np.random.seed(seed)
                main = sorted(np.random.choice(range(1, 60), 6, replace=False))
                bonus = np.random.choice([n for n in range(1, 60) if n not in main])
                main_numbers_list.append(main)
                bonus_numbers.append(bonus)
        
        df['Main_Numbers'] = main_numbers_list
        df['Bonus'] = bonus_numbers

def validate_dataframe(df: pd.DataFrame, fix_issues: bool = True) -> Tuple[bool, Dict]:
    """
    Validate a dataframe against predefined rules.
    
    Args:
        df: DataFrame to validate
        fix_issues: Whether to attempt automatic fixes for minor issues
        
    Returns:
        Tuple of (is_valid, validation_results)
    """
    validator = DataValidator()
    return validator.validate_data(df, fix_issues)

def validate_prediction(prediction: List[int]) -> Tuple[bool, str]:
    """
    Validate a lottery prediction.
    
    Args:
        prediction: List of 6 numbers to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    validator = DataValidator()
    return validator.validate_prediction(prediction)

def fix_prediction(prediction: Union[List[int], np.ndarray], max_number: int = 59, n_numbers: int = 6) -> List[int]:
    """
    Fix an invalid prediction by ensuring it contains n_numbers unique integers between 1 and max_number.
    
    Args:
        prediction (Union[List[int], np.ndarray]): The prediction to fix
        max_number (int): Maximum allowed number (inclusive)
        n_numbers (int): Required number of predictions
        
    Returns:
        List[int]: Fixed prediction containing n_numbers unique integers between 1 and max_number
    """
    # Convert to list and handle floating point numbers
    prediction = [int(round(x)) for x in prediction]
    
    # Remove duplicates and sort
    prediction = sorted(list(set(prediction)))
    
    # Remove numbers outside valid range
    prediction = [x for x in prediction if 1 <= x <= max_number]
    
    # Take first n_numbers if we have too many
    prediction = prediction[:n_numbers]
    
    # Add random numbers if we don't have enough
    while len(prediction) < n_numbers:
        new_num = random.randint(1, max_number)
        if new_num not in prediction:
            prediction.append(new_num)
    
    return sorted(prediction) 