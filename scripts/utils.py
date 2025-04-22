import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import sys

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as auto_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback implementation if tqdm is not available
    class DummyTQDM:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable is not None else None)
            self.n = 0
            self.kwargs = kwargs
        
        def update(self, n=1):
            self.n += n
            
        def close(self):
            pass
            
        def set_description(self, desc=None):
            pass
            
        def __iter__(self):
            if self.iterable is None:
                return range(self.total).__iter__()
            return self.iterable.__iter__()
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args, **kwargs):
            self.close()
    
    tqdm = DummyTQDM
    auto_tqdm = DummyTQDM

# Constants
LOOK_BACK = 200  # Number of previous draws to consider for prediction
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "lottery.log"

def setup_logging(log_file: Optional[str] = None, 
                  log_level: int = logging.INFO, 
                  console: bool = True) -> None:
    """
    Configure logging for the lottery prediction system.
    
    Args:
        log_file: Path to log file (default: logs/lottery.log)
        log_level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_file = LOG_FILE
    
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler with detailed formatting
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Console handler with simplified formatting
    if console:
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    # Log initial setup
    logging.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    logging.info(f"Log file: {os.path.abspath(log_file)}")

def create_progress_bar(iterable=None, total=None, desc="Processing", leave=True, 
                       position=None, **kwargs) -> Any:
    """
    Create a progress bar using tqdm if available, or a dummy implementation if not.
    
    Args:
        iterable: Iterable to wrap with progress bar
        total: Total number of items (required if iterable is None)
        desc: Description to show in the progress bar
        leave: Whether to leave the progress bar after completion
        position: Position of the progress bar (for multiple bars)
        **kwargs: Additional arguments to pass to tqdm
        
    Returns:
        tqdm progress bar or dummy implementation
    """
    if not HAS_TQDM:
        return DummyTQDM(
            iterable=iterable,
            total=total,
            desc=desc,
            **kwargs
        )
    
    # Use auto_tqdm for automatic terminal/notebook detection
    if iterable is not None:
        return auto_tqdm(
            iterable, 
            desc=desc,
            leave=leave,
            position=position,
            **kwargs
        )
    else:
        return auto_tqdm(
            total=total, 
            desc=desc,
            leave=leave,
            position=position,
            **kwargs
        )

def create_model_progress_tracker(models: List[str], desc="Training models") -> Dict[str, Any]:
    """
    Create a progress tracker for multiple models.
    
    Args:
        models: List of model names
        desc: Description for the main progress bar
        
    Returns:
        Dictionary with progress tracking objects
    """
    if not HAS_TQDM:
        logging.warning("tqdm not installed - progress bars disabled")
        return {
            "main": DummyTQDM(total=len(models), desc=desc),
            "models": {model: None for model in models},
            "current": None
        }
    
    return {
        "main": auto_tqdm(total=len(models), desc=desc, position=0, leave=True),
        "models": {model: None for model in models},
        "current": None
    }

def update_model_progress(tracker: Dict[str, Any], model_name: str, 
                         desc: str = None, reset: bool = False) -> None:
    """
    Update the progress tracker for a specific model.
    
    Args:
        tracker: Progress tracker dictionary
        model_name: Name of the current model
        desc: Description for the current model's progress bar
        reset: Whether to reset the progress bar
    """
    if not HAS_TQDM or tracker is None:
        return
    
    # Update main progress bar
    if reset and "current" in tracker and tracker["current"] == model_name:
        tracker["main"].update()
    
    # Set description for main progress bar
    if desc is not None:
        current_desc = f"Training models: {model_name} - {desc}"
        tracker["main"].set_description(current_desc)
    
    # Update current model
    tracker["current"] = model_name

def ensure_valid_prediction(prediction: list, min_val: int = 1, max_val: int = 59, required_length: int = 6) -> list:
    """
    Ensure a prediction contains the required number of unique integers within the valid range.
    
    Args:
        prediction: List of predicted numbers
        min_val: Minimum valid value (default: 1)
        max_val: Maximum valid value (default: 59)
        required_length: Required number of unique values (default: 6)
        
    Returns:
        Valid prediction with required_length unique integers
    """
    import random
    
    # Filter valid numbers from the prediction
    valid_numbers = [
        num for num in prediction 
        if isinstance(num, int) and min_val <= num <= max_val
    ]
    
    # Remove duplicates
    valid_numbers = list(set(valid_numbers))
    
    # Add random numbers if needed
    while len(valid_numbers) < required_length:
        num = random.randint(min_val, max_val)
        if num not in valid_numbers:
            valid_numbers.append(num)
    
    # Trim if too many
    if len(valid_numbers) > required_length:
        valid_numbers = valid_numbers[:required_length]
    
    # Sort the numbers
    return sorted(valid_numbers)

def get_timestamp_str(include_time: bool = True) -> str:
    """
    Get current timestamp as a string.
    
    Args:
        include_time: Whether to include time in addition to date (default: True)
        
    Returns:
        Timestamp string in format "YYYY-MM-DD" or "YYYY-MM-DD_HH-MM-SS"
    """
    from datetime import datetime
    now = datetime.now()
    if include_time:
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    return now.strftime("%Y-%m-%d")

if __name__ == "__main__":
    # Example usage
    setup_logging()
    logging.info("Testing utils.py logging setup")
    
    test_prediction = [1, 3, 5, 7, 9, 11]
    valid_prediction = ensure_valid_prediction(test_prediction)
    logging.info(f"Test prediction: {test_prediction}")
    logging.info(f"Valid prediction: {valid_prediction}")
    
    invalid_prediction = [0, 1, 60, 3, 3, 3]
    valid_prediction = ensure_valid_prediction(invalid_prediction)
    logging.info(f"Invalid prediction: {invalid_prediction}")
    logging.info(f"Fixed prediction: {valid_prediction}")
    
    timestamp = get_timestamp_str()
    logging.info(f"Current timestamp: {timestamp}")
    
    # Test progress bar
    if HAS_TQDM:
        print("Testing progress bar functionality:")
        with create_progress_bar(range(10), desc="Testing") as pbar:
            for i in pbar:
                import time
                time.sleep(0.1)
                pbar.set_description(f"Processing item {i}") 