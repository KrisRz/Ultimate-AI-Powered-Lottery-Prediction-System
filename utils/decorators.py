"""Decorators for model training and monitoring."""

import functools
import logging
from typing import Callable, Any
from .gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

def monitor_gpu(func: Callable) -> Callable:
    """Decorator to monitor GPU usage during function execution.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with GPU monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = GPUMonitor(interval=1.0)
        
        try:
            # Start monitoring
            monitor.start()
            logger.info("Started GPU monitoring")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Get usage summary
            summary = monitor.get_memory_usage_summary()
            logger.info("GPU Memory Usage Summary:")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{key}: {value:.2f}")
                else:
                    logger.info(f"{key}: {value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during monitored execution: {e}")
            raise
        finally:
            # Always stop monitoring
            if monitor:
                monitor.stop()
                logger.info("Stopped GPU monitoring")
    
    return wrapper 