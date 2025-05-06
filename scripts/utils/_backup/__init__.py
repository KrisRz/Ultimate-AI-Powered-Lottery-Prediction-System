from pathlib import Path
import logging

# Define constants
LOG_DIR = Path("logs")
CONFIG_DIR = Path("config")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Create directories if they don't exist
for directory in [LOG_DIR, CONFIG_DIR, DATA_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'lottery.log'),
            logging.StreamHandler()
        ]
    )
