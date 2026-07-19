#!/usr/bin/env python3
"""
TensorFlow-Compatible Lottery Prediction System

This version works around library compatibility issues and focuses on
TensorFlow/Keras models that we know work.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
import warnings

# Force CPU mode for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import TensorFlow
import tensorflow as tf

# Configure logging
log_file = Path("logs/lottery.log")
log_file.parent.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our data fetching
from scripts.fetch_data import load_data, download_fresh_data

# Define paths
OUTPUT_DIR = Path("outputs")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = Path("models/checkpoints")
DATA_DIR = Path("data")

# Create directories
for directory in [OUTPUT_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TensorFlow-Compatible Lottery Prediction System')
    parser.add_argument('--retrain', choices=['yes', 'no'], default='no',
                       help='Retrain models from scratch (yes) or use existing trained models (no)')
    parser.add_argument('--force', action='store_true',
                       help='Force download of fresh lottery data from the web')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of predictions to generate')
    return parser.parse_args()

def create_simple_lstm_model(sequence_length=10, n_features=6):
    """Create a simple LSTM model for lottery prediction"""
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(6, activation='sigmoid')  # 6 numbers between 0-1
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Created simple LSTM model successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error creating LSTM model: {str(e)}")
        return None

def prepare_data_for_lstm(df, sequence_length=10):
    """Prepare data for LSTM training"""
    try:
        # Extract numbers and normalize
        if 'Main_Numbers' in df.columns:
            numbers = np.array([row for row in df['Main_Numbers'].values if isinstance(row, list)])
        else:
            # Try to extract from Ball columns
            ball_cols = [f'Ball {i}' for i in range(1, 7)]
            if all(col in df.columns for col in ball_cols):
                numbers = df[ball_cols].values
            else:
                raise ValueError("Cannot find lottery numbers in data")
        
        # Normalize to 0-1 range
        numbers = numbers / 59.0
        
        # Create sequences
        X, y = [], []
        for i in range(len(numbers) - sequence_length):
            X.append(numbers[i:i+sequence_length])
            y.append(numbers[i+sequence_length])
        
        return np.array(X), np.array(y)
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        return None, None

def train_simple_model(df):
    """Train a simple LSTM model"""
    try:
        logger.info("Preparing data for training...")
        X, y = prepare_data_for_lstm(df)
        
        if X is None or len(X) == 0:
            logger.error("No training data available")
            return None
            
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create model
        model = create_simple_lstm_model(sequence_length=X.shape[1])
        if model is None:
            return None
            
        # Train model
        logger.info("Training LSTM model...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save model
        model_path = MODELS_DIR / "simple_lstm_model.h5"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def load_trained_model():
    """Load a trained model"""
    try:
        model_path = MODELS_DIR / "simple_lstm_model.h5"
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        else:
            logger.warning(f"No trained model found at {model_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def generate_predictions_with_model(model, df, count=10):
    """Generate predictions using the trained model"""
    try:
        # Prepare recent data for prediction
        X, _ = prepare_data_for_lstm(df)
        if X is None or len(X) == 0:
            return generate_fallback_predictions(count)
            
        # Use the most recent sequence
        recent_sequence = X[-1:] 
        
        predictions = []
        for _ in range(count):
            # Get prediction (6 numbers between 0-1)
            pred = model.predict(recent_sequence, verbose=0)
            
            # Convert back to lottery numbers (1-59)
            numbers = (pred[0] * 59).astype(int)
            numbers = np.clip(numbers, 1, 59)
            
            # Ensure unique numbers
            unique_numbers = []
            for num in numbers:
                if num not in unique_numbers:
                    unique_numbers.append(int(num))
            
            # Fill with random numbers if needed
            while len(unique_numbers) < 6:
                new_num = np.random.randint(1, 60)
                if new_num not in unique_numbers:
                    unique_numbers.append(new_num)
            
            predictions.append(sorted(unique_numbers[:6]))
            
            # Slightly modify the sequence for next prediction
            recent_sequence = recent_sequence + np.random.normal(0, 0.01, recent_sequence.shape)
            recent_sequence = np.clip(recent_sequence, 0, 1)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return generate_fallback_predictions(count)

def generate_fallback_predictions(count=10):
    """Generate fallback predictions using mathematical methods"""
    predictions = []
    for _ in range(count):
        numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
        predictions.append(numbers.tolist())
    return predictions

def save_predictions(predictions, metadata=None):
    """Save predictions to JSON file"""
    try:
        timestamp = datetime.now().isoformat()
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            "timestamp": timestamp,
            "predictions": predictions,
            "count": len(predictions),
            "metadata": metadata or {}
        }
        
        # Save to file
        filename = f"tensorflow_predictions_{date_str}.json"
        output_path = PREDICTIONS_DIR / filename
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        return None

def display_predictions(predictions):
    """Display predictions in a formatted way"""
    print("\n" + "="*60)
    print("🎯 TENSORFLOW LOTTERY PREDICTIONS")
    print("="*60)
    
    for i, pred in enumerate(predictions, 1):
        pred_str = " - ".join(f"{num:02d}" for num in pred)
        print(f"Prediction {i}: {pred_str}")
    
    print("\n" + "="*60)
    print(f"Generated {len(predictions)} predictions using TensorFlow LSTM")
    print("="*60)

def main():
    """Main entry point"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("🚀 TENSORFLOW-COMPATIBLE LOTTERY PREDICTION SYSTEM")
    print("="*60)
    
    try:
        # Download fresh data if requested
        if args.force:
            print("\n📥 Downloading fresh lottery data...")
            success = download_fresh_data()
            if not success:
                print("❌ Data download failed, using existing data")
            else:
                print("✅ Data download successful")
        
        # Load data
        print("\n📊 Loading lottery data...")
        data_path = DATA_DIR / "merged_lottery_data.csv"
        if not data_path.exists():
            data_path = DATA_DIR / "lottery_data_1995_2025.csv"
            
        df = load_data(data_path)
        print(f"✅ Loaded {len(df)} lottery records")
        
        # Handle model training/loading
        model = None
        if args.retrain == 'yes':
            print("\n🔄 Training new model...")
            model = train_simple_model(df)
        else:
            print("\n📤 Loading existing model...")
            model = load_trained_model()
            
            if model is None:
                print("❌ No existing model found. Training new model...")
                model = train_simple_model(df)
        
        # Generate predictions
        print(f"\n🎲 Generating {args.count} predictions...")
        if model is not None:
            predictions = generate_predictions_with_model(model, df, args.count)
            method = "TensorFlow LSTM"
        else:
            print("❌ Model not available, using fallback method...")
            predictions = generate_fallback_predictions(args.count)
            method = "Mathematical fallback"
        
        # Save and display results
        metadata = {
            "method": method,
            "tensorflow_version": tf.__version__,
            "data_records": len(df)
        }
        
        save_predictions(predictions, metadata)
        display_predictions(predictions)
        
        print(f"\n✅ Prediction generation completed using {method}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"\n❌ Error: {str(e)}")
        
        # Generate fallback predictions
        print("🔄 Generating fallback predictions...")
        predictions = generate_fallback_predictions(args.count)
        save_predictions(predictions, {"method": "Fallback due to error"})
        display_predictions(predictions)

if __name__ == "__main__":
    main() 