"""
Utility functions for the lottery prediction system.

This package contains various utilities for data processing,
model evaluation, cross-validation, memory monitoring, and validation.
"""

# Version
__version__ = '0.1.0'

# Import common utilities
from pathlib import Path
import os
import logging

# Define common paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True, parents=True)

# Create logger
logger = logging.getLogger(__name__)

# Make cross-validation functions available at package level
try:
    from .cross_validation import (
        perform_rolling_window_cv,
        perform_time_series_cv,
        calculate_lottery_matches,
        cross_validate_models
    )
except ImportError:
    # Module might not exist yet
    logger.warning("cross_validation module not available")
    pass

# Import memory monitoring
try:
    from .memory_monitor import (
        get_memory_usage,
        log_memory_usage,
        get_memory_monitor,
        configure_memory_growth,
        memory_callback,
        optimize_batch_size,
        MemoryMonitor
    )
except ImportError:
    # Module might not exist yet
    logger.warning("memory_monitor module not available")
    
    # Define fallback functions
    def log_memory_usage(label=""):
        logger.info(f"Memory monitoring not available ({label})")
        return {}
    
    def get_memory_monitor(**kwargs):
        logger.info("Memory monitoring not available")
        return None

try:
    from .model_utils import (
        setup_gpu_memory_growth,
        set_random_seed,
        create_sequences,
        normalize_data,
        denormalize_data,
        create_callbacks,
        validate_predictions,
        calculate_metrics,
        log_training_errors
    )
except ImportError:
    logger.warning("model_utils module not available")

# Import validations, trying the newer path first
try:
    from ..validations import DataValidator, validate_dataframe, validate_prediction, LotteryValidator
except ImportError:
    try:
        # Fallback to old path if needed
        from ..validations import DataValidator, validate_dataframe, validate_prediction
        from ..validations import LotteryValidator
    except ImportError:
        logging.warning("Failed to import validation utilities")

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'lottery.log'),
            logging.StreamHandler()
        ]
    )

# Define public API
__all__ = [
    # Cross-validation
    'perform_rolling_window_cv',
    'perform_time_series_cv',
    'cross_validate_models',
    
    # Memory monitoring
    'log_memory_usage',
    'get_memory_monitor',
    'configure_memory_growth',
    'memory_callback',
    'optimize_batch_size',
    'MemoryMonitor',
    
    # Model utilities
    'setup_gpu_memory_growth',
    'set_random_seed',
    'create_sequences',
    'normalize_data',
    'denormalize_data',
    'create_callbacks',
    'validate_predictions',
    'calculate_metrics',
    'log_training_errors',
    
    # Validation
    'validate_dataframe',
    'validate_prediction',
    'LotteryValidator',
    
    # Logging
    'setup_logging',
    
    # Constants
    'ROOT_DIR',
    'DATA_DIR',
    'MODEL_DIR',
    'OUTPUT_DIR',
    'LOG_DIR',
    'DataValidator'
] 