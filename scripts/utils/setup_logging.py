"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: Path = None, level: int = logging.INFO) -> None:
    """Configure logging with file and console handlers."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Always log to lottery.log
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'lottery.log'
    
    # File handler for lottery.log
    file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler) 