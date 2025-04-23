import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import psutil
import gc
import time
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Dict, Tuple, List, Union
import random
from collections import Counter

# Configuration
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Constants
DATA_FILE = "data/lottery_data_1995_2025.csv"
MODEL_DIR = "models"
MODELS_PATH = os.path.join(MODEL_DIR, "trained_models.pkl")
LOOK_BACK = 200
N_PREDICTIONS = 10
EPOCHS = 20

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def log_system_metrics():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_percent = psutil.cpu_percent()
    disk_usage = psutil.disk_usage('/')
    gpu_info = {}
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info[f"GPU {gpu.id}"] = {
                "load": f"{gpu.load*100:.1f}%",
                "memory": f"{gpu.memoryUtil*100:.1f}%",
                "temperature": f"{gpu.temperature}°C"
            }
    except:
        gpu_info = {"status": "No GPU available"}
    
    logging.info(f"System Metrics:")
    logging.info(f"  CPU Usage: {cpu_percent}%")
    logging.info(f"  Memory Usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    logging.info(f"  Disk Usage: {disk_usage.percent}%")
    logging.info(f"  Available Memory: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
    logging.info(f"  GPU Status: {gpu_info}")
    logging.info(f"  Thread Count: {process.num_threads()}")
    logging.info(f"  Open Files: {len(process.open_files())}")

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_percent = psutil.cpu_percent()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB, CPU usage: {cpu_percent:.2f}%")

def ensure_valid_prediction(pred: Union[List, np.ndarray]) -> List[int]:
    """
    Ensure predictions are 6 unique integers between 1 and 59.
    
    Args:
        pred: Raw model prediction
        
    Returns:
        List of 6 unique integers between 1 and 59
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(pred, list):
            pred = np.array(pred)
        
        # Handle NaN, infinity, or other invalid values
        if pred is None or not isinstance(pred, (list, np.ndarray)):
            logging.warning(f"Invalid prediction type: {type(pred)}. Generating random numbers.")
            return sorted(np.random.choice(range(1, 60), size=6, replace=False).tolist())
        
        # Check for NaN or infinite values
        if np.isnan(pred).any() or np.isinf(pred).any():
            logging.warning(f"Prediction contains NaN or Inf values: {pred}. Fixing...")
            pred = np.nan_to_num(pred, nan=30.0, posinf=59.0, neginf=1.0)
        
        # Handle potential fractional values by rounding
        pred = np.round(pred).astype(int)
        
        # Ensure values are within valid range (1-59)
        pred = np.clip(pred, 1, 59)
        
        # Check if we have the right number of predictions
        if len(pred) != 6:
            logging.warning(f"Prediction has {len(pred)} numbers instead of 6. Adjusting...")
            if len(pred) > 6:
                # Take the first 6 values
                pred = pred[:6]
            else:
                # Add random numbers until we have 6
                current = set(pred)
                while len(current) < 6:
                    candidate = np.random.randint(1, 60)
                    if candidate not in current:
                        current.add(candidate)
                pred = np.array(list(current))
        
        # Ensure uniqueness
        unique_values = set(pred)
        if len(unique_values) < 6:
            logging.warning(f"Prediction has duplicate values: {pred}. Fixing...")
            while len(unique_values) < 6:
                candidate = np.random.randint(1, 60)
                if candidate not in unique_values:
                    unique_values.add(candidate)
            pred = np.array(list(unique_values))
        
        # Return sorted list of integers
        return sorted([int(x) for x in pred])
    
    except Exception as e:
        logging.error(f"Error processing prediction: {e}. Generating random numbers.")
        return sorted(np.random.choice(range(1, 60), size=6, replace=False).tolist())

def parse_balls(x: str, df: pd.DataFrame = None) -> Tuple[List[int], int]:
    """Parse Balls string into main numbers and bonus number."""
    try:
        parts = x.split()
        numbers = [int(n) for n in parts if n.isdigit() and 1 <= int(n) <= 59]
        main_numbers = numbers[:6] if len(numbers) >= 6 else numbers
        bonus = numbers[-1] if len(numbers) > 6 else None
        
        if df is None or 'Main_Numbers' not in df.columns or len(main_numbers) < 6:
            while len(main_numbers) < 6:
                candidate = random.randint(1, 59)
                if candidate not in main_numbers:
                    main_numbers.append(candidate)
        else:
            all_nums = [num for draw in df['Main_Numbers'] for num in draw]
            freq = Counter(all_nums)
            top_nums = [num for num, _ in freq.most_common()]
            main_numbers.extend([n for n in top_nums if n not in main_numbers][:6 - len(main_numbers)])
        
        bonus = bonus if bonus else random.randint(1, 59)
        while bonus in main_numbers:
            bonus = random.randint(1, 59)
        return sorted(main_numbers[:6]), bonus
    except Exception as e:
        logging.error(f"Error parsing balls: {str(e)}")
        raise ValueError(f"Invalid Balls format: {x}")

def log_training_errors(func):
    def wrapper(*args, **kwargs):
        try:
            logging.info(f"Starting {func.__name__} training...")
            log_system_metrics()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if isinstance(result, tuple):
                model = result[0]
                if hasattr(model, 'history'):
                    history = model.history.history
                    logging.info(f"Training History for {func.__name__}:")
                    for metric, values in history.items():
                        logging.info(f"  {metric}: {values[-1]:.4f}")
                elif hasattr(model, 'score'):
                    score = model.score(*args)
                    logging.info(f"Model Score: {score:.4f}")
            
            logging.info(f"Completed {func.__name__} training in {duration:.2f} seconds")
            logging.info(f"Training Speed: {len(args[0])/duration:.2f} samples/second")
            log_system_metrics()
            
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.error(f"Error occurred after {time.time() - start_time:.2f} seconds")
            log_system_metrics()
            raise
    return wrapper 