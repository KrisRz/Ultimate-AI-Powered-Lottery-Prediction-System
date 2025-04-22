import logging
import os
from pathlib import Path
from .validation import ensure_valid_prediction

# Constants for lottery analysis
LOOK_BACK = 200  # Number of past draws to consider
DEFAULT_NUMBERS = 6  # Number of lottery balls to draw
MAX_LOTTERY_NUMBER = 59  # Maximum lottery number
MIN_LOTTERY_NUMBER = 1  # Minimum lottery number

def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file, defaults to logs/lottery.log
        level: Logging level, defaults to INFO
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'lottery.log'
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Log basic information
    logging.info(f"Logging initialized at level {logging._levelToName[level]}")
    logging.info(f"Log file: {log_file}")
    
    return logger

__all__ = ['ensure_valid_prediction', 'setup_logging', 'LOOK_BACK', 
           'DEFAULT_NUMBERS', 'MAX_LOTTERY_NUMBER', 'MIN_LOTTERY_NUMBER'] 