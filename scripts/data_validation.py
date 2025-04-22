import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import re
import sys
import traceback

# Setup logging
try:
    from utils import setup_logging
    setup_logging()
except ImportError:
    logging.basicConfig(filename='lottery.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates lottery data to ensure data quality before prediction or training.
    
    Performs validation on each column according to predefined rules and provides
    detailed error reporting and suggestions for fixes.
    """
    
    def __init__(self, validation_rules: Optional[Dict] = None):
        """
        Initialize validator with default or custom validation rules.
        
        Args:
            validation_rules: Optional custom validation rules dictionary
        """
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
                'values': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            },
            'Balls': {
                'type': 'str',
                'required': True,
                'format': r'^\d{1,2}( \d{1,2}){5} BONUS \d{1,2}$',
                'validate': self._validate_balls_format
            },
            'Jackpot': {
                'type': 'str',
                'required': True,
                'format': r'^£[\d,]+$'
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
                'required': False,  # Optional if Balls is present
                'validate': self._validate_main_numbers
            },
            'Bonus': {
                'type': 'int',
                'required': False,  # Optional if Balls is present
                'min': 1,
                'max': 59
            }
        }
        
        # Initialize validation history
        self.history_file = Path('results/validation_log.json')
    
    def validate_data(self, data: pd.DataFrame, fix_issues: bool = True) -> Tuple[bool, Dict]:
        """
        Validate lottery data against rules.
        
        Args:
            data: DataFrame with lottery data
            fix_issues: Whether to attempt to fix minor issues automatically
        
        Returns:
            Tuple of (is_valid: bool, results: Dict) indicating validation success and details
        """
        start_time = datetime.now()
        validation_results = {
            'timestamp': start_time.isoformat(),
            'data_shape': {
                'rows': len(data),
                'columns': list(data.columns)
            },
            'checks': {},
            'errors': [],
            'warnings': [],
            'fixes_applied': []
        }
        
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Check required columns
            required_cols = {k for k, v in self.validation_rules.items() if v.get('required', False)}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            
            # Validate each column
            for column, rules in self.validation_rules.items():
                if column not in df.columns:
                    if rules.get('required', False):
                        continue  # Already reported in missing columns
                    else:
                        validation_results['warnings'].append(f"Optional column {column} missing")
                        continue
                
                column_results = self._validate_column(df, column, rules, fix_issues)
                validation_results['checks'][column] = column_results
                
                if column_results.get('errors', []):
                    validation_results['errors'].extend([f"{column}: {e}" for e in column_results['errors']])
                if column_results.get('warnings', []):
                    validation_results['warnings'].extend([f"{column}: {w}" for w in column_results['warnings']])
                if column_results.get('fixes', []):
                    validation_results['fixes_applied'].extend([f"{column}: {f}" for f in column_results['fixes']])
            
            # Additional cross-column and consistency checks
            consistency_results = self._check_consistency(df)
            validation_results['checks']['consistency'] = consistency_results
            if consistency_results.get('errors', []):
                validation_results['errors'].extend([f"Consistency: {e}" for e in consistency_results['errors']])
            if consistency_results.get('warnings', []):
                validation_results['warnings'].extend([f"Consistency: {w}" for w in consistency_results['warnings']])
            
            # Save results and return
            self._save_validation_results(validation_results)
            
            # Log summary and update with duration
            duration = (datetime.now() - start_time).total_seconds()
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
            logger.error(traceback.format_exc())
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            return False, validation_results
    
    def _validate_column(self, df: pd.DataFrame, column: str, rules: Dict, fix_issues: bool) -> Dict:
        """
        Validate a single column against its rules.
        
        Args:
            df: DataFrame containing the data
            column: Column name to validate
            rules: Validation rules for this column
            fix_issues: Whether to attempt fixing issues
            
        Returns:
            Dictionary with validation results
        """
        series = df[column]
        results = {
            'errors': [],
            'warnings': [],
            'fixes': []
        }
        
        try:
            # Apply type validation
            if rules['type'] == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(series):
                    if fix_issues:
                        try:
                            df[column] = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
                            results['fixes'].append(f"Converted to datetime (format auto-detected)")
                        except Exception as e:
                            results['errors'].append(f"Failed to convert to datetime: {str(e)}")
                    else:
                        results['errors'].append("Column must be datetime type")
            
            elif rules['type'] == 'int':
                if not pd.api.types.is_integer_dtype(series):
                    if fix_issues:
                        try:
                            df[column] = pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
                            results['fixes'].append(f"Converted to integer (missing/invalid values set to 0)")
                        except Exception as e:
                            results['errors'].append(f"Failed to convert to integer: {str(e)}")
                    else:
                        results['errors'].append("Column must be integer type")
            
            elif rules['type'] == 'list':
                # Check if values are lists/arrays
                non_list_indices = [i for i, x in enumerate(series) if not isinstance(x, (list, np.ndarray))]
                if non_list_indices:
                    results['errors'].append(f"Found {len(non_list_indices)} values that are not lists at indices: "
                                           f"{non_list_indices[:5]}{'...' if len(non_list_indices) > 5 else ''}")
            
            # Check uniqueness
            if rules.get('unique', False):
                duplicates = series.duplicated()
                if duplicates.any():
                    dup_values = series[duplicates].unique()
                    results['warnings'].append(f"Found {duplicates.sum()} duplicate values: {dup_values[:3]}"
                                             f"{'...' if len(dup_values) > 3 else ''}")
            
            # Check for valid values if specified
            if 'values' in rules:
                invalid_mask = ~series.isin(rules['values'])
                if invalid_mask.any():
                    invalid_values = series[invalid_mask].unique()
                    results['errors'].append(f"Found {invalid_mask.sum()} invalid values: {invalid_values[:3]}"
                                           f"{'...' if len(invalid_values) > 3 else ''}")
            
            # Check regex format for string columns
            if 'format' in rules and pd.api.types.is_string_dtype(series):
                invalid_mask = ~series.str.match(rules['format'], na=False)
                if invalid_mask.any():
                    invalid_examples = series[invalid_mask].iloc[:3].tolist()
                    results['errors'].append(f"Found {invalid_mask.sum()} values with invalid format. Examples: "
                                           f"{invalid_examples}")
            
            # Check min/max for numeric columns
            if pd.api.types.is_numeric_dtype(series):
                if 'min' in rules:
                    below_min = series < rules['min']
                    if below_min.any():
                        if fix_issues and rules.get('fix_min', False):
                            df.loc[below_min, column] = rules['min']
                            results['fixes'].append(f"Set {below_min.sum()} values below minimum to {rules['min']}")
                        else:
                            results['errors'].append(f"Found {below_min.sum()} values below minimum {rules['min']}")
                
                if 'max' in rules:
                    above_max = series > rules['max']
                    if above_max.any():
                        if fix_issues and rules.get('fix_max', False):
                            df.loc[above_max, column] = rules['max']
                            results['fixes'].append(f"Set {above_max.sum()} values above maximum to {rules['max']}")
                        else:
                            results['errors'].append(f"Found {above_max.sum()} values above maximum {rules['max']}")
            
            # Fix missing values if specified
            if fix_issues and rules.get('fix_missing', False) and series.isna().any():
                missing_count = series.isna().sum()
                if rules['type'] == 'int':
                    df[column] = series.fillna(0).astype(int)
                    results['fixes'].append(f"Filled {missing_count} missing values with 0")
                elif rules['type'] == 'str':
                    df[column] = series.fillna('')
                    results['fixes'].append(f"Filled {missing_count} missing values with empty string")
            
            # Custom validation function
            if 'validate' in rules:
                custom_errors, custom_warnings, custom_fixes = rules['validate'](df, column, fix_issues)
                results['errors'].extend(custom_errors)
                results['warnings'].extend(custom_warnings)
                results['fixes'].extend(custom_fixes)
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating column {column}: {str(e)}")
            logger.error(traceback.format_exc())
            results['errors'].append(f"Validation error: {str(e)}")
            return results
    
    def _validate_main_numbers(self, df: pd.DataFrame, column: str, fix_issues: bool) -> Tuple[List[str], List[str], List[str]]:
        """
        Custom validation for Main_Numbers column.
        
        Args:
            df: DataFrame containing the data
            column: Column name (should be 'Main_Numbers')
            fix_issues: Whether to attempt fixing issues
            
        Returns:
            Tuple of (errors, warnings, fixes)
        """
        errors = []
        warnings = []
        fixes = []
        
        series = df[column]
        
        # Check for incorrect types, lengths, ranges
        for idx, numbers in series.items():
            try:
                if not isinstance(numbers, (list, np.ndarray)):
                    errors.append(f"Row {idx}: Main_Numbers must be a list, got {type(numbers).__name__}")
                    continue
                    
                if len(numbers) != 6:
                    errors.append(f"Row {idx}: Main_Numbers must have exactly 6 values, got {len(numbers)}")
                    continue
                
                if not all(isinstance(n, (int, np.integer)) for n in numbers):
                    if fix_issues:
                        try:
                            # Try to convert to integers
                            fixed_numbers = [int(float(n)) for n in numbers]
                            df.at[idx, column] = fixed_numbers
                            fixes.append(f"Row {idx}: Converted non-integer values to integers")
                        except Exception:
                            errors.append(f"Row {idx}: Main_Numbers must contain integers")
                    else:
                        errors.append(f"Row {idx}: Main_Numbers must contain integers")
                
                # Check number range
                out_of_range = [n for n in df.at[idx, column] if not (1 <= n <= 59)]
                if out_of_range:
                    errors.append(f"Row {idx}: Found {len(out_of_range)} numbers outside range 1-59: {out_of_range}")
                
                # Check for duplicates
                if len(set(df.at[idx, column])) != 6:
                    warnings.append(f"Row {idx}: Main_Numbers contains duplicates")
                
                # Check bonus number isn't in main numbers
                if 'Bonus' in df.columns and idx in df.index:
                    bonus = df.at[idx, 'Bonus']
                    if isinstance(bonus, (int, np.integer)) and bonus in df.at[idx, column]:
                        errors.append(f"Row {idx}: Bonus number {bonus} must not be in Main_Numbers")
            
            except Exception as e:
                errors.append(f"Row {idx}: Error validating Main_Numbers: {str(e)}")
        
        return errors, warnings, fixes
    
    def _validate_balls_format(self, df: pd.DataFrame, column: str, fix_issues: bool) -> Tuple[List[str], List[str], List[str]]:
        """
        Custom validation for Balls column.
        
        Args:
            df: DataFrame containing the data
            column: Column name (should be 'Balls')
            fix_issues: Whether to attempt fixing issues
            
        Returns:
            Tuple of (errors, warnings, fixes)
        """
        errors = []
        warnings = []
        fixes = []
        
        # Regex pattern for valid ball format - allowing single or double digits
        ball_pattern = re.compile(r'^\d{1,2}( \d{1,2}){5} BONUS \d{1,2}$')
        
        # Process each row
        for idx, ball_str in df[column].items():
            if not isinstance(ball_str, str):
                errors.append(f"Row {idx}: Balls must be a string, got {type(ball_str).__name__}")
                continue
            
            if not ball_pattern.match(ball_str):
                if fix_issues:
                    try:
                        # Try to fix common formatting issues
                        fixed = self._try_fix_ball_format(ball_str)
                        if ball_pattern.match(fixed):
                            df.at[idx, column] = fixed
                            fixes.append(f"Row {idx}: Fixed ball format '{ball_str}' to '{fixed}'")
                        else:
                            errors.append(f"Row {idx}: Invalid Balls format '{ball_str}' (couldn't fix)")
                    except Exception as e:
                        errors.append(f"Row {idx}: Error fixing Balls format: {str(e)}")
                else:
                    errors.append(f"Row {idx}: Invalid Balls format '{ball_str}'")
                continue
            
            # Check for valid number ranges and duplicates
            try:
                parts = ball_str.split(' BONUS ')
                main_numbers = [int(n) for n in parts[0].split()]
                bonus = int(parts[1])
                
                # Check ranges
                if not all(1 <= n <= 59 for n in main_numbers):
                    out_of_range = [n for n in main_numbers if not (1 <= n <= 59)]
                    errors.append(f"Row {idx}: Main numbers contain values outside range 1-59: {out_of_range}")
                
                if not 1 <= bonus <= 59:
                    errors.append(f"Row {idx}: Bonus number {bonus} outside range 1-59")
                
                # Check for duplicates
                if len(set(main_numbers)) != 6:
                    warnings.append(f"Row {idx}: Main numbers contain duplicates: {main_numbers}")
                
                # Check if bonus is in main numbers
                if bonus in main_numbers:
                    errors.append(f"Row {idx}: Bonus number {bonus} should not be in main numbers {main_numbers}")
                
            except Exception as e:
                errors.append(f"Row {idx}: Error parsing Balls values: {str(e)}")
        
        return errors, warnings, fixes
    
    def _try_fix_ball_format(self, ball_str: str) -> str:
        """
        Attempt to fix common ball format issues.
        
        Args:
            ball_str: Original ball string
            
        Returns:
            Fixed ball string
        """
        # Handle missing "BONUS" keyword
        if 'BONUS' not in ball_str:
            parts = ball_str.split()
            if len(parts) == 7:
                return ' '.join(parts[:6]) + ' BONUS ' + parts[6]
        
        # Handle extra spaces
        if '  ' in ball_str:
            return re.sub(r'\s+', ' ', ball_str)
        
        # Handle missing spaces
        if re.search(r'\d\d', ball_str):
            numbers = re.findall(r'\d+', ball_str)
            if len(numbers) == 7:
                return ' '.join(numbers[:6]) + ' BONUS ' + numbers[6]
        
        # If we can't fix it, return original
        return ball_str
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Check data consistency across columns and rows.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with consistency check results
        """
        results = {
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check chronological order of draw dates
            if 'Draw Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Draw Date']):
                if not df['Draw Date'].is_monotonic_increasing:
                    unsorted_indices = np.where(np.diff(df['Draw Date']) < pd.Timedelta(0))[0]
                    results['warnings'].append(f"Draw Date is not chronological at indices: {unsorted_indices[:5]}"
                                             f"{'...' if len(unsorted_indices) > 5 else ''}")
            
            # Check day of week matches draw date
            if 'Draw Date' in df.columns and 'Day' in df.columns:
                day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
                reverse_day_map = {v: k for k, v in day_map.items()}
                
                if pd.api.types.is_datetime64_any_dtype(df['Draw Date']):
                    day_mismatches = []
                    for idx, row in df.iterrows():
                        if row['Day'] in day_map:
                            expected_day = reverse_day_map.get(row['Draw Date'].dayofweek)
                            if expected_day and expected_day != row['Day']:
                                day_mismatches.append(idx)
                    
                    if day_mismatches:
                        results['warnings'].append(f"Day of week doesn't match Draw Date for {len(day_mismatches)} rows: "
                                                f"indices {day_mismatches[:5]}{'...' if len(day_mismatches) > 5 else ''}")
            
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
    
    def validate_prediction(self, prediction: Union[List[int], np.ndarray]) -> Tuple[bool, str]:
        """
        Validate a single prediction (6 unique integers, 1-59).
        
        Args:
            prediction: List of predicted numbers
        
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        try:
            error_msg = ""
            
            # Check type
            if not isinstance(prediction, (list, np.ndarray)):
                error_msg = f"Prediction must be a list, got {type(prediction).__name__}"
                logger.error(error_msg)
                return False, error_msg
            
            # Check length
            if len(prediction) != 6:
                error_msg = f"Prediction must contain exactly 6 numbers, got {len(prediction)}"
                logger.error(error_msg)
                return False, error_msg
            
            # Check integer type
            if not all(isinstance(n, (int, np.integer)) for n in prediction):
                error_msg = "Prediction must contain only integers"
                logger.error(error_msg)
                return False, error_msg
            
            # Check range
            out_of_range = [n for n in prediction if not (1 <= n <= 59)]
            if out_of_range:
                error_msg = f"Prediction contains numbers outside range 1-59: {out_of_range}"
                logger.error(error_msg)
                return False, error_msg
            
            # Check uniqueness
            if len(set(prediction)) != 6:
                error_msg = f"Prediction must contain 6 unique numbers, got {prediction}"
                logger.error(error_msg)
                return False, error_msg
            
            return True, ""
            
        except Exception as e:
            error_msg = f"Error validating prediction: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _save_validation_results(self, results: Dict):
        """
        Save validation results to history file.
        
        Args:
            results: Validation results to save
        """
        try:
            self.history_file.parent.mkdir(exist_ok=True)
            
            history = self.get_validation_history()
            history.append(results)
            
            # Keep only the last 10 validation results to prevent file growth
            if len(history) > 10:
                history = history[-10:]
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving validation results: {str(e)}")
    
    def get_validation_history(self) -> List[Dict]:
        """
        Get validation history from log file.
        
        Returns:
            List of validation result dictionaries
        """
        try:
            if not self.history_file.exists():
                return []
            
            with open(self.history_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error reading validation history: {str(e)}")
            return []


def validate_dataframe(df: pd.DataFrame, fix_issues: bool = True) -> Tuple[bool, Dict]:
    """
    Validate a DataFrame using the DataValidator class.
    
    Args:
        df: DataFrame to validate
        fix_issues: Whether to attempt fixing minor issues
        
    Returns:
        Tuple of (is_valid: bool, results: Dict) indicating validation success and details
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


if __name__ == "__main__":
    # For running validation directly from command line
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description='Validate lottery data CSV file')
        parser.add_argument('file', help='Path to CSV file with lottery data')
        parser.add_argument('--no-fix', dest='fix', action='store_false', help='Do not attempt to fix issues')
        parser.set_defaults(fix=True)
        
        args = parser.parse_args()
        
        print(f"Validating data in {args.file}...")
        df = pd.read_csv(args.file)
        
        # Convert date column if present
        if 'Draw Date' in df.columns:
            df['Draw Date'] = pd.to_datetime(df['Draw Date'], errors='coerce')
        
        is_valid, results = validate_dataframe(df, fix_issues=args.fix)
        
        print(f"\nValidation {'PASSED' if is_valid else 'FAILED'}")
        print(f"Found {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        
        if results['errors']:
            print("\nERRORS:")
            for e in results['errors']:
                print(f"  - {e}")
        
        if results['warnings']:
            print("\nWARNINGS:")
            for w in results['warnings']:
                print(f"  - {w}")
        
        if args.fix and results['fixes_applied']:
            print("\nFIXES APPLIED:")
            for f in results['fixes_applied']:
                print(f"  - {f}")
        
        print(f"\nValidation log saved to {Path('results/validation_log.json')}")
        sys.exit(0 if is_valid else 1)
        
    except Exception as e:
        print(f"Error running validation: {str(e)}")
        traceback.print_exc()
        sys.exit(2) 