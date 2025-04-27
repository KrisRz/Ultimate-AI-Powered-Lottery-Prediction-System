"""
Memory usage tracking utilities.
"""
import os
import psutil
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def log_memory_usage(func: Callable) -> Callable:
    """
    Decorator to log memory usage before and after function execution.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that logs memory usage
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        process = psutil.Process(os.getpid())
        
        # Log initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Memory usage before {func.__name__}: {initial_memory:.2f} MB")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Log final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = final_memory - initial_memory
            logger.info(f"Memory usage after {func.__name__}: {final_memory:.2f} MB (Δ {memory_diff:+.2f} MB)")
            
            return result
            
        except Exception as e:
            # Log memory on error
            error_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = error_memory - initial_memory
            logger.error(f"Memory usage on error in {func.__name__}: {error_memory:.2f} MB (Δ {memory_diff:+.2f} MB)")
            raise
            
    return wrapper 