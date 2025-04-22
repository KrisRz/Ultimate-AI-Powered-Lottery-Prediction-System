#!/usr/bin/env python3
"""
Optimized Lottery Model Training Script

This script demonstrates the optimized model training process for the lottery prediction system.
It includes parallel training, model caching, and memory profiling.
"""

import argparse
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimize_training.log')
    ]
)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=200):
    """Create synthetic lottery data for training."""
    dates = pd.date_range(start='2020-01-01', periods=n_samples)
    numbers = []
    
    for _ in range(n_samples):
        draw = sorted(np.random.choice(range(1, 60), size=6, replace=False))
        numbers.append(draw)
    
    df = pd.DataFrame({
        'Draw Date': dates,
        'Main_Numbers': numbers
    })
    
    # Add basic features
    df['Year'] = df['Draw Date'].dt.year
    df['Month'] = df['Draw Date'].dt.month
    df['DayOfWeek'] = df['Draw Date'].dt.dayofweek
    df['Sum'] = df['Main_Numbers'].apply(sum)
    df['Mean'] = df['Main_Numbers'].apply(np.mean)
    df['Std'] = df['Main_Numbers'].apply(np.std)
    df['Min'] = df['Main_Numbers'].apply(min)
    df['Max'] = df['Main_Numbers'].apply(max)
    
    # Add more features required by the models
    df['Odds'] = df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 1))
    df['Evens'] = df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 0))
    df['Range'] = df['Max'] - df['Min']
    df['Unique'] = df['Main_Numbers'].apply(lambda x: len(set(x)))
    df['ZScore_Sum'] = (df['Sum'] - df['Sum'].mean()) / df['Sum'].std()
    df['Freq_10'] = 1
    df['Freq_20'] = 2 
    df['Freq_50'] = 3
    df['Pair_Freq'] = 4
    df['Triplet_Freq'] = 5
    df['Gaps'] = df['Main_Numbers'].apply(lambda x: np.mean([x[i+1] - x[i] for i in range(len(x)-1)]))
    df['Primes'] = df['Main_Numbers'].apply(
        lambda x: sum(1 for n in x if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
    )
    
    return df

def main():
    """Main function to run the optimized training."""
    parser = argparse.ArgumentParser(description="Optimized lottery model training")
    parser.add_argument('--data', type=str, default=None, help='Path to data file (CSV)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for parallel training')
    parser.add_argument('--synthetic', type=int, default=200, help='Number of synthetic samples if no data file')
    args = parser.parse_args()
    
    # Import the optimized training module
    try:
        from models.optimized_training import train_all_models_parallel, load_optimized_models
        logger.info("Successfully imported optimized training module")
    except ImportError as e:
        logger.error(f"Failed to import optimized training module: {e}")
        return
    
    # Load or create training data
    if args.data and Path(args.data).exists():
        logger.info(f"Loading data from {args.data}")
        try:
            from scripts.fetch_data import load_data
            df = load_data(args.data)
        except ImportError:
            logger.warning("Could not import load_data from scripts, using pandas")
            df = pd.read_csv(args.data)
        logger.info(f"Loaded {len(df)} lottery draws")
    else:
        logger.info(f"Creating synthetic data with {args.synthetic} samples")
        df = create_sample_data(args.synthetic)
    
    # Time the training process
    start_time = time.time()
    
    # Train models with optimization
    logger.info(f"Starting optimized training with {args.workers or 'auto'} workers")
    models = train_all_models_parallel(df, max_workers=args.workers)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Report results
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Successfully trained {len(models)} models")
    
    # Verify models can be loaded
    loaded_models = load_optimized_models()
    logger.info(f"Verified loading of {len(loaded_models)} models")

if __name__ == "__main__":
    main() 