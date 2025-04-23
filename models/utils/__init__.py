"""
Utility functions and helpers for model training and management.
"""

from typing import Callable, TypeVar, Any

from .model_storage import (
    save_model_atomic,
    cleanup_old_checkpoints,
    manage_model_cache,
    get_storage_info
)

from .decorators import log_training_errors

# Constants
LOOK_BACK = 200  # Number of previous draws to use for prediction

# Type variable for the decorated function
F = TypeVar('F', bound=Callable[..., Any])

def ensure_valid_prediction(func: F) -> F:
    """Ensure prediction output is valid"""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if result is None or len(result) == 0:
            raise ValueError(f"Invalid prediction from {func.__name__}")
        return result
    return wrapper  # type: ignore

__all__ = [
    'save_model_atomic',
    'cleanup_old_checkpoints',
    'manage_model_cache',
    'get_storage_info',
    'log_training_errors',
    'ensure_valid_prediction',
    'LOOK_BACK'
] 