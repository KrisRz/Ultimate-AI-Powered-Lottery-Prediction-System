import logging
import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add scripts directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try direct imports first
    from scripts.utils import setup_logging
    from scripts.fetch_data import load_data
    from scripts.data_validation import DataValidator
    from scripts.train_models import train_all_models, load_trained_models
    from scripts.predict_next_draw import validate_and_save_predictions, format_predictions_for_display
    from scripts.predict_numbers import predict_next_draw
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("data/lottery_data_1995_2025.csv")
RESULTS_PATH = Path("results/predictions.json")

def main(retrain: str = 'no', data_path: str = str(DATA_PATH), 
         n_predictions: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Main function for the lottery prediction system.
    
    Args:
        retrain: Whether to retrain models ('yes'/'no' or 'true'/'false')
        data_path: Path to lottery data CSV
        n_predictions: Number of predictions to generate
        verbose: Whether to print results to console
    
    Returns:
        Dictionary with results (predictions, metrics, etc.)
    """
    start_time = time.time()
    result = {'success': False}
    
    try:
        # Normalize retrain input
        retrain_flag = str(retrain).lower() in ('yes', 'true', 'y', '1')
        if verbose:
            print(f"Starting lottery prediction system (retrain: {retrain_flag})")
        logger.info(f"Starting lottery prediction system (retrain: {retrain_flag})")
        
        # Load and validate data
        if verbose:
            print(f"Loading data from {data_path}...")
        logger.info(f"Loading data from {data_path}...")
        
        try:
            # Modify load_data call to disable validation
            data = load_data(data_path, validate=False)
            if verbose:
                print(f"Loaded {len(data)} lottery draws")
            logger.info(f"Loaded {len(data)} lottery draws")
            
            # Extract Main_Numbers from Balls if needed
            if 'Balls' in data.columns and ('Main_Numbers' not in data.columns or data['Main_Numbers'].isna().all()):
                if verbose:
                    print("Extracting Main_Numbers from Balls column...")
                # Process each row
                main_numbers_list = []
                bonus_numbers = []
                valid_indices = []
                
                for idx, ball_str in enumerate(data['Balls']):
                    try:
                        if not isinstance(ball_str, str):
                            # Handle non-string values
                            ball_str = str(ball_str)
                            if ball_str.lower() == 'nan':
                                continue
                        
                        # Strip quotes
                        ball_str = ball_str.strip('"').strip("'")
                        
                        # Check if it's proper format with "BONUS"
                        if ' BONUS ' not in ball_str:
                            continue
                            
                        # Split by BONUS
                        parts = ball_str.split(' BONUS ')
                        if len(parts) != 2:
                            continue
                            
                        main_str, bonus_str = parts
                        main = sorted([int(n) for n in main_str.split()])
                        bonus = int(bonus_str)
                        
                        if len(main) != 6:
                            continue
                            
                        main_numbers_list.append(main)
                        bonus_numbers.append(bonus)
                        valid_indices.append(idx)
                    except Exception as e:
                        if verbose and idx < 10:  # Only show errors for first few rows
                            print(f"Skipping row {idx} due to parsing error: {e}")
                        logger.error(f"Error parsing row {idx}, Balls: '{ball_str}': {str(e)}")
                
                # Keep only valid rows
                if valid_indices:
                    if verbose:
                        print(f"Keeping {len(valid_indices)} valid rows out of {len(data)}")
                    data = data.iloc[valid_indices].copy()
                    data['Main_Numbers'] = main_numbers_list
                    data['Bonus'] = bonus_numbers
                    data = data.reset_index(drop=True)
                else:
                    error_msg = "No valid lottery data found in the CSV file."
                    if verbose:
                        print(f"Error: {error_msg}")
                    logger.error(error_msg)
                    result['error'] = error_msg
                    return result
            
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            if verbose:
                print(f"Error: {error_msg}")
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result['error'] = error_msg
            return result
        
        # Validate data
        try:
            validator = DataValidator()
            validation_result = validator.validate_dataframe(data)
            
            if not validation_result[0]:  # First element is boolean valid/invalid
                error_msg = f"Data validation failed: {validation_result[1]['errors']}"
                if verbose:
                    print(f"Error: {error_msg}")
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            if 'warnings' in validation_result[1] and validation_result[1]['warnings']:
                warning_msg = f"Data validation warnings: {validation_result[1]['warnings']}"
                if verbose:
                    print(f"Warning: {warning_msg}")
                logger.warning(warning_msg)
        except Exception as e:
            logger.warning(f"Data validation could not be performed: {str(e)}")
            logger.debug(traceback.format_exc())
            if verbose:
                print(f"Warning: Data validation could not be performed: {str(e)}")
        
        # Train or load models
        models = {}
        if retrain_flag:
            if verbose:
                print("Training models (this may take a while)...")
            logger.info("Training models with retrain flag set to True")
            
            try:
                models = train_all_models(data, force_retrain=True)
                if not models:
                    error_msg = "No models were successfully trained"
                    if verbose:
                        print(f"Error: {error_msg}")
                    logger.error(error_msg)
                    result['error'] = error_msg
                    return result
                
                if verbose:
                    print(f"Successfully trained {len(models)} models")
                logger.info(f"Successfully trained {len(models)} models")
            except Exception as e:
                error_msg = f"Error training models: {str(e)}"
                if verbose:
                    print(f"Error: {error_msg}")
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                result['error'] = error_msg
                return result
        else:
            if verbose:
                print("Using existing trained models...")
            logger.info("Loading existing trained models")
            
            try:
                models = load_trained_models()
                if not models:
                    error_msg = "No trained models found. Please train models first (use --retrain yes)"
                    if verbose:
                        print(f"Error: {error_msg}")
                    logger.error(error_msg)
                    result['error'] = error_msg
                    return result
                
                if verbose:
                    print(f"Loaded {len(models)} trained models")
                logger.info(f"Loaded {len(models)} trained models")
            except Exception as e:
                error_msg = f"Error loading models: {str(e)}"
                if verbose:
                    print(f"Error: {error_msg}")
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                result['error'] = error_msg
                return result
        
        # Generate predictions
        if verbose:
            print(f"Generating {n_predictions} predictions...")
        logger.info(f"Generating {n_predictions} predictions...")
        
        try:
            # Generate predictions using predict_numbers.py
            predictions = predict_next_draw(models, data, n_predictions=n_predictions)
            
            # Validate and save predictions
            validation_result = validate_and_save_predictions(
                predictions=predictions,
                data=data,
                output_path=str(RESULTS_PATH),
                metadata={
                    'generated_by': 'main.py',
                    'retrain': retrain_flag,
                    'models_used': list(models.keys())
                }
            )
            
            if not validation_result['success']:
                error_msg = f"Error saving predictions: {validation_result.get('error', 'Unknown error')}"
                if verbose:
                    print(f"Warning: {error_msg}")
                logger.warning(error_msg)
            
            # Display predictions
            if verbose:
                print(validation_result['display'])
            
            # Return results
            result.update({
                'success': True,
                'predictions': validation_result['predictions'],
                'metrics': validation_result['metrics'],
                'invalid_count': validation_result['invalid_count']
            })
            
        except Exception as e:
            error_msg = f"Error generating predictions: {str(e)}"
            if verbose:
                print(f"Error: {error_msg}")
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result['error'] = error_msg
            return result
        
        # Log completion time
        duration = time.time() - start_time
        if verbose:
            print(f"\nCompleted in {duration:.2f} seconds")
        logger.info(f"Prediction process completed in {duration:.2f} seconds")
        
        return result
    
    except Exception as e:
        error_msg = f"Unexpected error in main function: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        result['error'] = error_msg
        return result

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lottery Prediction System")
    parser.add_argument("--retrain", type=str, default="no", choices=["yes", "no", "true", "false"],
                      help="Whether to retrain models (yes/no)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                      help=f"Path to lottery data CSV (default: {DATA_PATH})")
    parser.add_argument("--predictions", type=int, default=10,
                      help="Number of predictions to generate (default: 10)")
    parser.add_argument("--quiet", action="store_true",
                      help="Suppress console output (default: False)")
    
    args = parser.parse_args()
    
    # Run main function
    result = main(
        retrain=args.retrain,
        data_path=args.data,
        n_predictions=args.predictions,
        verbose=not args.quiet
    )
    
    if not result['success']:
        sys.exit(1) 