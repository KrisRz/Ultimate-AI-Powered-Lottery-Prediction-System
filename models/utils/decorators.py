"""
Decorators for model training and utilities
"""
import functools
import logging
import traceback
from typing import Callable, Any

logger = logging.getLogger(__name__)

def log_training_errors(func: Callable) -> Callable:
    """
    Decorator to log errors during model training
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that logs errors
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    return wrapper 