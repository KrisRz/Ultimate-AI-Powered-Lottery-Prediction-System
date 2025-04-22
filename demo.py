#!/usr/bin/env python
"""
Lottery Prediction System Demo

This script demonstrates how to use all the predictive models
to generate lottery number predictions.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo.log')
    ]
)
logger = logging.getLogger(__name__)

# Import compatibility functions
try:
    from models.compatibility import (
        predict_next_draw,
        ensemble_prediction,
        monte_carlo_simulation,
        save_predictions,
        ensure_valid_prediction,
        predict_with_model,
        get_model_weights
    )
    logger.info("Successfully imported model functions from models.compatibility")
except ImportError as e:
    logger.error(f"Error importing from models.compatibility: {e}")
    try:
        from scripts.model_bridge import (
            predict_next_draw,
            ensemble_prediction,
            monte_carlo_simulation,
            save_predictions,
            ensure_valid_prediction,
            predict_with_model,
            get_model_weights
        )
        logger.info("Successfully imported model functions from scripts.model_bridge")
    except ImportError as e:
        logger.error(f"Error importing from scripts.model_bridge: {e}")
        raise ImportError("Failed to import required model functions")

# Try to import optimized training functions
try:
    from models.optimized_training import train_all_models_parallel, load_optimized_models
    use_optimized = True
    logger.info("Using optimized training functions")
except ImportError:
    use_optimized = False
    logger.warning("Optimized training not available, falling back to standard")
    # Import standard training function
    try:
        from scripts.train_models import train_all_models, load_trained_models
        logger.info("Successfully imported standard training functions")
    except ImportError as e:
        logger.error(f"Error importing training functions: {e}")
        
        # Define fallback functions
        def load_trained_models():
            return {}
        
        def train_all_models(df, force_retrain=False):
            return {}

# Import data loading function
try:
    from scripts.fetch_data import load_data
    logger.info("Successfully imported data loading function")
except ImportError as e:
    logger.error(f"Error importing data loading function: {e}")
    
    # Define fallback function
    def load_data(path):
        return pd.DataFrame({
            'Draw Date': pd.date_range(start='2020-01-01', periods=100),
            'Main_Numbers': [[np.random.choice(range(1, 60), size=6, replace=False).tolist() 
                              for _ in range(100)]]
        })

def create_sample_data(n_samples=100):
    """Create sample lottery data if no real data is available."""
    logger.info(f"Creating sample data with {n_samples} draws")
    
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
    
    # Add more features for models
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
    """Main function to demonstrate the lottery prediction system."""
    parser = argparse.ArgumentParser(description="Lottery prediction system demo")
    parser.add_argument('--data', type=str, default="data/lottery_data_1995_2025.csv", 
                        help='Path to lottery data file')
    parser.add_argument('--force-train', action='store_true', 
                        help='Force retraining even if models exist')
    parser.add_argument('--predictions', type=int, default=10, 
                        help='Number of predictions to generate')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of worker processes for parallel training')
    parser.add_argument('--use-optimized', type=str, choices=['auto', 'yes', 'no'], default='auto',
                        help='Whether to use optimized training (auto=detect)')
    args = parser.parse_args()
    
    # Determine whether to use optimized training
    if args.use_optimized == 'auto':
        # Already determined in imports
        pass
    elif args.use_optimized == 'yes':
        if not use_optimized:
            logger.warning("Optimized training requested but not available")
    else:  # 'no'
        use_optimized = False
        logger.info("Using standard training per user request")
    
    logger.info("Starting Lottery Prediction System Demo")
    
    # Load data
    data_path = args.data
    try:
        logger.info(f"Attempting to load data from {data_path}")
        df = load_data(data_path)
        logger.info(f"Successfully loaded {len(df)} draws from {data_path}")
    except Exception as e:
        logger.warning(f"Failed to load data from {data_path}: {e}")
        logger.info("Creating sample data instead")
        df = create_sample_data(n_samples=200)
    
    # Check if we have trained models
    if use_optimized:
        models = load_optimized_models()
    else:
        models = load_trained_models()
    
    if not models or args.force_train:
        logger.info(f"{'No' if not models else 'Force retraining requested. No'} trained models found. Training models now...")
        if use_optimized:
            models = train_all_models_parallel(df, max_workers=args.workers)
        else:
            models = train_all_models(df, force_retrain=True)
    
    logger.info(f"Loaded {len(models)} trained models")
    
    if not models:
        logger.warning("No models available for prediction. Using dummy models.")
        # Create dummy models for demonstration
        from unittest.mock import MagicMock
        models = {
            'lstm': MagicMock(),
            'xgboost': MagicMock(),
            'lightgbm': MagicMock(),
            'catboost': MagicMock()
        }
    
    # Generate predictions using each model individually
    logger.info("Generating individual model predictions")
    individual_predictions = {}
    
    for model_name, model in models.items():
        try:
            prediction = predict_with_model(model_name, model, df)
            individual_predictions[model_name] = prediction
            logger.info(f"{model_name} prediction: {prediction}")
        except Exception as e:
            logger.error(f"Error generating prediction for {model_name}: {e}")
    
    # Calculate model weights
    model_weights = get_model_weights(models)
    logger.info(f"Model weights: {model_weights}")
    
    # Generate ensemble predictions
    logger.info("Generating ensemble predictions")
    ensemble_predictions = ensemble_prediction(
        list(individual_predictions.values()),
        df,
        model_weights,
        prediction_count=args.predictions
    )
    
    logger.info(f"Generated {len(ensemble_predictions)} ensemble predictions")
    for i, pred in enumerate(ensemble_predictions, 1):
        logger.info(f"Ensemble prediction {i}: {pred}")
    
    # Generate Monte Carlo predictions
    logger.info("Generating Monte Carlo predictions")
    mc_predictions = monte_carlo_simulation(df, n_simulations=5)
    
    logger.info(f"Generated {len(mc_predictions)} Monte Carlo predictions")
    for i, pred in enumerate(mc_predictions, 1):
        logger.info(f"Monte Carlo prediction {i}: {pred}")
    
    # Save all predictions to file
    all_predictions = {
        'individual': individual_predictions,
        'ensemble': ensemble_predictions,
        'monte_carlo': mc_predictions
    }
    
    # Create output directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)
    
    # Save predictions to JSON
    output_file = Path('results/demo_predictions.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'predictions': all_predictions,
            'model_weights': model_weights,
            'optimized_training_used': use_optimized
        }, f, indent=2)
    
    logger.info(f"Saved predictions to {output_file}")
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(traceback.format_exc()) 