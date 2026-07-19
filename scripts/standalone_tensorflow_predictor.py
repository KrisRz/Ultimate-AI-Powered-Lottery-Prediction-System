#!/usr/bin/env python3
"""
Standalone TensorFlow Lottery Predictor

This script works completely independently and only uses TensorFlow + basic libraries.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime

# Force CPU mode for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def load_lottery_data():
    """Load lottery data from CSV files"""
    data_files = [
        "data/merged_lottery_data.csv",
        "data/lottery_data_1995_2025.csv", 
        "data/lotto-draw-history.csv"
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"📊 Loading data from {file_path}")
            df = pd.read_csv(file_path)
            return df, file_path
    
    raise FileNotFoundError("No lottery data files found")

def extract_numbers_from_data(df):
    """Extract lottery numbers from various data formats"""
    numbers = []
    
    # Try different column formats
    if 'Main_Numbers' in df.columns:
        # Handle string format like "[1, 2, 3, 4, 5, 6]"
        for item in df['Main_Numbers']:
            if pd.isna(item):
                continue
            if isinstance(item, str):
                # Parse string representation of list
                try:
                    nums = eval(item)  # Convert string list to actual list
                    if isinstance(nums, list) and len(nums) == 6:
                        numbers.append(nums)
                except:
                    continue
            elif isinstance(item, list):
                numbers.append(item)
    
    # Try Ball 1-6 columns
    elif all(f'Ball {i}' in df.columns for i in range(1, 7)):
        ball_cols = [f'Ball {i}' for i in range(1, 7)]
        for _, row in df.iterrows():
            try:
                nums = [int(row[col]) for col in ball_cols]
                if all(1 <= num <= 59 for num in nums):
                    numbers.append(nums)
            except:
                continue
    
    # Try Number_1-6 columns
    elif all(f'Number_{i}' in df.columns for i in range(1, 7)):
        num_cols = [f'Number_{i}' for i in range(1, 7)]
        for _, row in df.iterrows():
            try:
                nums = [int(row[col]) for col in num_cols]
                if all(1 <= num <= 59 for num in nums):
                    numbers.append(nums)
            except:
                continue
    
    return np.array(numbers) if numbers else None

def create_lstm_model():
    """Create LSTM model for lottery prediction"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(10, 6)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(numbers, sequence_length=10):
    """Prepare sequences for LSTM training"""
    # Normalize numbers to 0-1 range
    normalized = numbers / 59.0
    
    X, y = [], []
    for i in range(len(normalized) - sequence_length):
        X.append(normalized[i:i+sequence_length])
        y.append(normalized[i+sequence_length])
    
    return np.array(X), np.array(y)

def train_model(numbers):
    """Train the LSTM model"""
    print("🔄 Training LSTM model...")
    
    # Prepare data
    X, y = prepare_sequences(numbers)
    print(f"Training data: {X.shape}, Target: {y.shape}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create and train model
    model = create_lstm_model()
    
    print("Training in progress...")
    history = model.fit(
        X_train, y_train,
        epochs=30,  # Reduced for faster training
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save model
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    model.save("models/checkpoints/standalone_lstm_model.h5")
    print("✅ Model saved to models/checkpoints/standalone_lstm_model.h5")
    
    return model

def load_model():
    """Load existing model"""
    model_path = "models/checkpoints/standalone_lstm_model.h5"
    if Path(model_path).exists():
        print(f"📤 Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    return None

def generate_predictions(model, numbers, count=10):
    """Generate lottery predictions"""
    print(f"🎲 Generating {count} predictions...")
    
    # Use recent sequences for prediction
    recent_sequence = numbers[-10:] / 59.0  # Normalize
    recent_sequence = recent_sequence.reshape(1, 10, 6)
    
    predictions = []
    
    for i in range(count):
        # Get prediction
        pred = model.predict(recent_sequence, verbose=0)
        
        # Convert back to lottery numbers
        lottery_numbers = (pred[0] * 59).astype(int)
        lottery_numbers = np.clip(lottery_numbers, 1, 59)
        
        # Ensure unique numbers
        unique_numbers = []
        for num in lottery_numbers:
            if num not in unique_numbers:
                unique_numbers.append(int(num))
        
        # Fill with random if needed
        while len(unique_numbers) < 6:
            new_num = np.random.randint(1, 60)
            if new_num not in unique_numbers:
                unique_numbers.append(new_num)
        
        predictions.append(sorted(unique_numbers[:6]))
        
        # Slightly modify sequence for next prediction
        recent_sequence = recent_sequence + np.random.normal(0, 0.02, recent_sequence.shape)
        recent_sequence = np.clip(recent_sequence, 0, 1)
    
    return predictions

def save_predictions(predictions):
    """Save predictions to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "count": len(predictions),
        "method": "Standalone TensorFlow LSTM",
        "tensorflow_version": tf.__version__
    }
    
    # Create output directory
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    
    # Save to file
    filename = f"standalone_tensorflow_predictions_{timestamp}.json"
    output_path = f"outputs/predictions/{filename}"
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Saved predictions to {output_path}")
    return output_path

def display_predictions(predictions):
    """Display predictions"""
    print("\n" + "="*60)
    print("🎯 STANDALONE TENSORFLOW LOTTERY PREDICTIONS")
    print("="*60)
    
    for i, pred in enumerate(predictions, 1):
        pred_str = " - ".join(f"{num:02d}" for num in pred)
        print(f"Prediction {i}: {pred_str}")
    
    print("\n" + "="*60)
    print(f"Generated {len(predictions)} predictions using TensorFlow {tf.__version__}")
    print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Standalone TensorFlow Lottery Predictor')
    parser.add_argument('--count', type=int, default=10, help='Number of predictions')
    parser.add_argument('--retrain', action='store_true', help='Force retrain model')
    args = parser.parse_args()
    
    print("🚀 STANDALONE TENSORFLOW LOTTERY PREDICTOR")
    print("="*60)
    
    try:
        # Load data
        df, data_file = load_lottery_data()
        print(f"✅ Loaded {len(df)} records from {data_file}")
        
        # Extract numbers
        numbers = extract_numbers_from_data(df)
        if numbers is None or len(numbers) == 0:
            raise ValueError("Could not extract lottery numbers from data")
        
        print(f"✅ Extracted {len(numbers)} lottery draws")
        print(f"Sample numbers: {numbers[-1]}")  # Show most recent
        
        # Load or train model
        model = None
        if not args.retrain:
            model = load_model()
        
        if model is None:
            if len(numbers) < 20:
                raise ValueError("Not enough data for training (need at least 20 draws)")
            model = train_model(numbers)
        
        # Generate predictions
        predictions = generate_predictions(model, numbers, args.count)
        
        # Save and display
        save_predictions(predictions)
        display_predictions(predictions)
        
        print("\n✅ Prediction generation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        
        # Fallback to random predictions
        print("🔄 Generating fallback predictions...")
        fallback_predictions = []
        for _ in range(args.count):
            numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
            fallback_predictions.append(numbers.tolist())
        
        save_predictions(fallback_predictions)
        display_predictions(fallback_predictions)

if __name__ == "__main__":
    main() 