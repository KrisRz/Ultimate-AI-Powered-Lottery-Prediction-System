import logging
import argparse
import sys
import time
import traceback
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'lstm_training.log'),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger('lstm_trainer')

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # Import necessary modules
    from models.lstm_model import train_lstm_model, predict_lstm_model
    from models.utils import ensure_valid_prediction, LOOK_BACK
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def load_data(file_path='data/fixed_lottery_data.csv'):
    """
    Load lottery data from CSV file
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with lottery data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} lottery draws")
        
        # Process Main_Numbers if needed
        if 'Balls' in df.columns and ('Main_Numbers' not in df.columns or df['Main_Numbers'].isna().all()):
            logger.info("Extracting Main_Numbers from Balls column")
            main_numbers_list = []
            bonus_numbers = []
            valid_indices = []
            
            for idx, ball_str in enumerate(df['Balls']):
                try:
                    if not isinstance(ball_str, str):
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
                    logger.error(f"Error parsing row {idx}, Balls: '{ball_str}': {str(e)}")
            
            # Keep only valid rows
            if valid_indices:
                logger.info(f"Keeping {len(valid_indices)} valid rows out of {len(df)}")
                df = df.iloc[valid_indices].copy()
                df['Main_Numbers'] = main_numbers_list
                df['Bonus'] = bonus_numbers
                df = df.reset_index(drop=True)
            else:
                logger.error("No valid lottery data found in the CSV file.")
                return None
        
        # If Main_Numbers is still not in right format (possibly as string representation of lists)
        if 'Main_Numbers' in df.columns and isinstance(df.iloc[0]['Main_Numbers'], str):
            logger.info("Converting Main_Numbers from string to list format")
            df['Main_Numbers'] = df['Main_Numbers'].apply(eval)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prepare_lstm_data(df, look_back=None):
    """
    Prepare data in format suitable for LSTM model
    
    Args:
        df: DataFrame with lottery data
        look_back: Number of previous draws to include
        
    Returns:
        Tuple of (X_train, y_train)
    """
    if look_back is None:
        look_back = LOOK_BACK
    
    logger.info(f"Preparing LSTM training data with look_back={look_back}")
    
    try:
        # Extract array of Main_Numbers
        y_array = np.array(df['Main_Numbers'].tolist())
        
        # Create sequences for LSTM
        X = []
        y = []
        
        for i in range(len(df) - look_back):
            X.append(y_array[i:i+look_back])
            y.append(y_array[i+look_back])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created LSTM training data with shape X:{X.shape}, y:{y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error preparing LSTM data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main(data_path='data/fixed_lottery_data.csv', look_back=None, epochs=20):
    """
    Main function to train the LSTM model
    
    Args:
        data_path: Path to the lottery data CSV
        look_back: Number of timesteps to use
        epochs: Number of training epochs
    """
    start_time = time.time()
    
    try:
        # Set default look_back if not specified
        if look_back is None:
            look_back = LOOK_BACK
        
        logger.info(f"Starting LSTM model training with look_back={look_back}, epochs={epochs}")
        
        # Load data
        df = load_data(data_path)
        if df is None:
            logger.error("Failed to load data, exiting.")
            return False
        
        # Create models directory for checkpoints
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Prepare data for LSTM
        X, y = prepare_lstm_data(df, look_back)
        if X is None or y is None:
            logger.error("Failed to prepare training data, exiting.")
            return False
        
        # Train the LSTM model
        logger.info(f"Training LSTM model with {epochs} epochs...")
        try:
            model, scaler, look_back = train_lstm_model(
                X, y, 
                epochs=epochs, 
                batch_size=64, 
                look_back=look_back, 
                validation_split=0.2
            )
            
            logger.info("LSTM model training completed successfully")
            
            # Make a test prediction
            latest_data = X[-1:]
            prediction = predict_lstm_model(model, scaler, latest_data, look_back)
            
            logger.info(f"LSTM model test prediction: {prediction}")
            
            duration = time.time() - start_time
            logger.info(f"Total runtime: {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for lottery prediction")
    parser.add_argument("--data", type=str, default="data/fixed_lottery_data.csv",
                        help="Path to lottery data CSV")
    parser.add_argument("--look_back", type=int, default=None,
                        help="Number of timesteps to use (default: uses LOOK_BACK from utils)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    
    args = parser.parse_args()
    
    success = main(
        data_path=args.data,
        look_back=args.look_back,
        epochs=args.epochs
    )
    
    if not success:
        sys.exit(1) 