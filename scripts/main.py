#!/usr/bin/env python3
"""
Lottery Prediction System - Main Entry Point

This script serves as the main entry point for the lottery prediction system.
It integrates all functionality including data fetching, model training,
and prediction generation in one convenient command-line interface.

Usage:
    python scripts/main.py --retrain yes|no --force

Options:
    --retrain yes|no    Whether to retrain models or use existing ones
    --force             Force download of fresh lottery data
    --count INT         Number of predictions to generate (default: 10)
    --diversity FLOAT   Diversity level for predictions (0-1, default: 0.5)
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import logging
import json
import pickle
import time
import traceback
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm, trange

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# Configure logging - ensure all logs go to logs/lottery.log
log_file = Path("logs/lottery.log")
log_file.parent.mkdir(exist_ok=True, parents=True)

# Set up root logger to capture all logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure all loggers to use the same file handler
for name in ['tensorflow', 'scripts', 'models', 'utils']:
    module_logger = logging.getLogger(name)
    module_logger.handlers = []
    module_logger.addHandler(logging.FileHandler(log_file))
    module_logger.addHandler(logging.StreamHandler(sys.stdout))
    module_logger.setLevel(logging.INFO)

# Import custom modules
from scripts.train_models import EnsembleTrainer
from scripts.fetch_data import load_data, prepare_sequence_data, DATA_DIR, download_fresh_data
from scripts.new_predict import visualize_predictions

# Define custom metrics for loading models (needed for TensorFlow models)
@tf.keras.utils.register_keras_serializable(package='custom_metrics')
def exact_match_metric(y_true, y_pred):
    """Custom metric: Count exact matches between predictions and actual values."""
    # Round predictions to nearest integer
    y_pred_rounded = tf.round(y_pred * 59)
    y_true_scaled = y_true * 59
    
    # Check if all numbers match
    matches = tf.reduce_all(tf.equal(y_pred_rounded, y_true_scaled), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

@tf.keras.utils.register_keras_serializable(package='custom_metrics')
def partial_match_metric(y_true, y_pred):
    """Custom metric: Count partial matches (at least 3 correct numbers)."""
    # Round predictions to nearest integer
    y_pred_rounded = tf.round(y_pred * 59)
    y_true_scaled = y_true * 59
    
    # Count matching numbers for each prediction
    matches = tf.cast(tf.equal(y_pred_rounded, y_true_scaled), tf.float32)
    match_counts = tf.reduce_sum(matches, axis=1)
    
    # Consider a partial match if at least 3 numbers match
    partial_matches = tf.greater_equal(match_counts, 3)
    return tf.reduce_mean(tf.cast(partial_matches, tf.float32))

# Define paths and directories
OUTPUT_DIR = Path("outputs")
TRAINING_DIR = OUTPUT_DIR / "training"
VALIDATION_DIR = OUTPUT_DIR / "validation"
MONITORING_DIR = OUTPUT_DIR / "monitoring"
INTERPRETATIONS_DIR = OUTPUT_DIR / "interpretations"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = Path("models/checkpoints")
LOGS_DIR = Path("logs")

# Create all directories
for directory in [OUTPUT_DIR, TRAINING_DIR, VALIDATION_DIR, MONITORING_DIR, 
                  INTERPRETATIONS_DIR, VISUALIZATIONS_DIR, PREDICTIONS_DIR, 
                  MODELS_DIR, LOGS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Lottery Prediction System')
    parser.add_argument('--retrain', choices=['yes', 'no'], default='no',
                       help='Retrain models from scratch (yes) or use existing trained models (no)')
    parser.add_argument('--force', action='store_true',
                       help='Force download of fresh lottery data from the web')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of predictions to generate')
    parser.add_argument('--diversity', type=float, default=0.5,
                       help='Diversity level for predictions (0-1)')
    parser.add_argument('--sequence-length', type=int, default=30,
                       help='Sequence length for model training')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    return parser.parse_args()

def generate_predictions(trainer, df, count=10, diversity=0.5):
    """Generate diverse lottery predictions using the ensemble model."""
    # Import prediction functions from new_predict.py
    from scripts.new_predict import (
        prepare_input_data, 
        predict_with_ensemble,
        perturb_prediction,
        generate_monte_carlo_predictions,
        save_predictions,
        visualize_predictions,
        create_ensemble
    )
    
    try:
        # Create ensemble from models
        ensemble = create_ensemble(trainer.models)
        
        # Generate predictions
        logger.info(f"Generating {count} diverse predictions...")
        
        # Prepare the input data
        X = prepare_input_data(df)
        
        # Get the base prediction from the ensemble
        base_prediction = predict_with_ensemble(ensemble, X)
        
        if base_prediction is None:
            logger.warning("Ensemble prediction failed, using individual models")
            # Try individual models
            for model_name, model in ensemble['models'].items():
                from scripts.new_predict import predict_with_model
                base_prediction = predict_with_model(model, X, model_type=model_name)
                if base_prediction is not None:
                    break
        
        if base_prediction is None:
            logger.warning("All model predictions failed, using random numbers")
            # Generate random prediction as fallback
            base_prediction = sorted(random.sample(range(1, 60), 6))
        
        logger.info(f"Base prediction: {base_prediction}")
        
        # Generate diverse predictions by perturbing the base prediction
        all_predictions = [base_prediction]
        
        # Create predictions with varying levels of perturbation
        with tqdm(total=count-1, desc="Generating predictions", unit="pred") as pbar:
            for i in range(1, count):
                # Increase diversity as we go
                current_diversity = diversity * (1 + i / count)
                new_prediction = perturb_prediction(base_prediction, intensity=current_diversity)
                
                # Check if this prediction is already in our list
                if new_prediction not in all_predictions:
                    all_predictions.append(new_prediction)
                else:
                    # Try again with higher diversity
                    attempts = 0
                    while new_prediction in all_predictions and attempts < 5:
                        new_prediction = perturb_prediction(base_prediction, intensity=current_diversity * 1.5)
                        attempts += 1
                    
                    if new_prediction not in all_predictions:
                        all_predictions.append(new_prediction)
                pbar.update(1)
        
        # If we still don't have enough predictions, add some random ones
        if len(all_predictions) < count:
            with tqdm(total=count-len(all_predictions), desc="Adding random predictions", unit="pred") as pbar:
                while len(all_predictions) < count:
                    random_pred = sorted(random.sample(range(1, 60), 6))
                    if random_pred not in all_predictions:
                        all_predictions.append(random_pred)
                        pbar.update(1)
        
        return all_predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        traceback.print_exc()
        # Fall back to Monte Carlo predictions
        return generate_monte_carlo_predictions(count)

def train_models(force_download=False, sequence_length=30):
    """
    Train or retrain all models using the improved training process.
    
    Args:
        force_download: Whether to force download fresh lottery data
        sequence_length: Sequence length for model training
        
    Returns:
        True if training was successful, False otherwise
    """
    print("\n" + "="*50)
    print("LOTTERY MODEL TRAINING")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # 1. Download/update data if needed
        if force_download:
            print("\nDownloading fresh lottery data...")
            # Show progress bar but call the function without custom parameters
            with tqdm(total=1, desc="Downloading data", unit="file") as pbar:
                success = download_fresh_data()
                pbar.update(1)  # Update once download is complete
                
                if not success:
                    logger.error("Failed to download fresh lottery data")
                    return False
            print("Download completed successfully")
        
        # 2. Run the improved training script
        print("\nStarting model training...")
        from scripts.improved_training import main as train_main
        
        # Configure tqdm in tensorflow callbacks
        class TqdmCallback(tf.keras.callbacks.Callback):
            def __init__(self, epochs, metrics=['loss']):
                super().__init__()
                self.epochs = epochs
                self.metrics = metrics
                self.tqdm = None
                
            def on_train_begin(self, logs=None):
                self.tqdm = tqdm(total=self.epochs, desc="Training progress", unit="epoch")
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                metrics_str = " - ".join([f"{m}: {logs.get(m, 0):.4f}" for m in self.metrics if m in logs])
                self.tqdm.set_postfix_str(metrics_str)
                self.tqdm.update(1)
                
            def on_train_end(self, logs=None):
                self.tqdm.close()
                
        # Monkey patch improved_training to use tqdm
        import scripts.improved_training
        original_train = scripts.improved_training.train_lstm_model
        
        def train_with_tqdm(*args, **kwargs):
            if 'callbacks' in kwargs:
                tqdm_callback = TqdmCallback(epochs=kwargs.get('epochs', 100), 
                                           metrics=['loss', 'val_loss'])
                kwargs['callbacks'].append(tqdm_callback)
            return original_train(*args, **kwargs)
        
        scripts.improved_training.train_lstm_model = train_with_tqdm
        
        # Run training
        train_main()
        
        # Restore original function
        scripts.improved_training.train_lstm_model = original_train
        
        # 3. Log completion time
        duration = time.time() - start_time
        print(f"\nTraining completed in {duration:.2f} seconds")
        print("\n" + "="*50)
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        traceback.print_exc()
        print(f"\nError during training: {str(e)}")
        print("\n" + "="*50)
        return False

def load_models():
    """
    Load trained models from disk with proper custom metrics.
    
    Returns:
        EnsembleTrainer object with loaded models or None if loading failed
    """
    try:
        # Create trainer without initial data
        trainer = EnsembleTrainer(None, None)
        
        # Set up custom metrics for model loading
        custom_metrics = {
            'exact_match_metric': exact_match_metric,
            'partial_match_metric': partial_match_metric
        }
        
        # Create a proper loader function with custom metrics
        def load_with_metrics(models_path):
            # Try to load from trained_models.pkl first
            try:
                # Create custom object scope for TensorFlow model loading
                with tf.keras.utils.custom_object_scope(custom_metrics):
                    # Try original loading method
                    if trainer.load_trained_models(models_path):
                        logger.info(f"Successfully loaded models from {models_path}")
                        return True
            except Exception as e:
                logger.error(f"Error loading models from pickle: {str(e)}")
            
            # If that fails, try loading directly from H5 files
            try:
                # Try to load LSTM models directly from checkpoints
                direct_models = {}
                
                # Check training directory for H5 files
                if TRAINING_DIR.exists():
                    h5_files = list(TRAINING_DIR.glob("*_checkpoint_*.h5"))
                    if h5_files:
                        with tqdm(total=len(h5_files), desc="Loading model checkpoints", unit="model") as pbar:
                            for h5_file in h5_files:
                                model_name = "lstm" if "lstm_checkpoint" in h5_file.name and "cnn" not in h5_file.name else "cnn_lstm"
                                logger.info(f"Attempting to load {model_name} from {h5_file}")
                                with tf.keras.utils.custom_object_scope(custom_metrics):
                                    try:
                                        model = tf.keras.models.load_model(h5_file)
                                        direct_models[model_name] = model
                                        logger.info(f"Successfully loaded {model_name} model")
                                    except Exception as model_e:
                                        logger.error(f"Error loading {model_name} from {h5_file}: {str(model_e)}")
                                pbar.update(1)
                
                if direct_models:
                    trainer.models = direct_models
                    from scripts.new_predict import create_ensemble
                    # Create ensemble from the loaded models
                    ensemble = create_ensemble(trainer.models)
                    trainer.ensemble = ensemble['models']
                    trainer.is_trained = True
                    logger.info(f"Successfully loaded {len(direct_models)} models directly from checkpoints")
                    return True
            except Exception as inner_e:
                logger.error(f"Error loading models from H5 files: {str(inner_e)}")
                traceback.print_exc()
            
            return False
        
        # Try to load models
        models_path = str(MODELS_DIR / "trained_models.pkl")
        if load_with_metrics(models_path):
            logger.info(f"Successfully loaded {len(trainer.models)} models")
            print(f"Loaded {len(trainer.models)} models: {', '.join(trainer.models.keys())}")
            return trainer
        else:
            logger.error("Failed to load any models")
            return None
        
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        traceback.print_exc()
        return None

def save_predictions(predictions, method="ensemble"):
    """Save predictions to JSON and text files."""
    # Create timestamps
    timestamp = datetime.now().isoformat()
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # JSON data
    json_data = {
        "timestamp": timestamp,
        "date": date_str,
        "predictions": predictions,
        "metadata": {
            "source": "main.py",
            "generation_method": method,
            "count": len(predictions)
        }
    }
    
    # Save to JSON
    json_path = PREDICTIONS_DIR / f"predictions_{date_str}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save latest copy
    latest_path = PREDICTIONS_DIR / "latest.json"
    with open(latest_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Create formatted text version
    text_path = PREDICTIONS_DIR / f"formatted_predictions_{date_str}.txt"
    with open(text_path, 'w') as f:
        f.write(f"LOTTERY PREDICTIONS - {date_str}\n")
        f.write("="*50 + "\n\n")
        
        for i, pred in enumerate(predictions, 1):
            pred_str = ", ".join(map(str, pred))
            f.write(f"{i}. {pred_str}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write(f"Generated {len(predictions)} predictions\n")
    
    logger.info(f"Saved predictions to {json_path} and {text_path}")
    return json_path, text_path

def display_predictions(predictions):
    """Display predictions in a nicely formatted way."""
    print("\n" + "="*50)
    print("LOTTERY PREDICTIONS")
    print("="*50 + "\n")
    
    for i, pred in enumerate(predictions, 1):
        pred_str = ", ".join(map(str, pred))
        print(f"{i}. {pred_str}")
    
    print("\n" + "="*50)
    print(f"Generated {len(predictions)} predictions")
    print("Predictions saved to outputs/predictions directory")
    print("="*50 + "\n")

def main():
    """Main entry point for the lottery prediction system."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Configure GPU memory growth
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
    
    # Check if we need to (re)train models
    if args.retrain == 'yes':
        print("\nRetrain option selected. Training models from scratch...")
        success = train_models(force_download=args.force, sequence_length=args.sequence_length)
        if not success:
            print("Model training failed. Check logs for details.")
            sys.exit(1)
    
    # Load trained models
    print("\nLoading trained models...")
    trainer = load_models()
    
    if trainer is None or not trainer.models:
        print("\nNo trained models available. Please run with --retrain yes first.")
        print("Alternatively, use --retrain yes to train models from scratch.")
        sys.exit(1)
    
    # Load lottery data
    print("\nLoading lottery data...")
    df = load_data(DATA_DIR / "merged_lottery_data.csv")
    print(f"Loaded {len(df)} lottery records")
    
    # Generate predictions
    print("\nGenerating lottery number predictions...")
    predictions = generate_predictions(
        trainer, 
        df, 
        count=args.count,
        diversity=args.diversity
    )
    
    # Save predictions
    save_predictions(predictions)
    
    # Display predictions
    display_predictions(predictions)
    
    # Create visualization - always on
    print("\nCreating visualizations...")
    try:
        visualize_predictions(predictions)
        print(f"Visualization saved to outputs/visualizations directory")
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        traceback.print_exc()
    
    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    import random  # For Monte Carlo fallback
    main() 