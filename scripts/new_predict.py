#!/usr/bin/env python3
"""
Simplified prediction script for lottery numbers using improved models.
This script loads models trained by improved_training.py and generates diverse predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime
import random
import argparse
import logging
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_data import load_data, DATA_DIR, prepare_sequence_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/new_predictions.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define output directories
PREDICTIONS_DIR = Path("outputs/predictions")
VISUALIZATION_DIR = Path("outputs/visualizations")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Constants
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 15

# Custom metrics for model loading
def exact_match_metric(y_true, y_pred):
    """Custom metric: Count exact matches between predictions and actual values."""
    # Round predictions to nearest integer
    y_pred_rounded = tf.round(y_pred * 59)
    y_true_scaled = y_true * 59
    
    # Check if all numbers match
    matches = tf.reduce_all(tf.equal(y_pred_rounded, y_true_scaled), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

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

def load_directly():
    """Load the trained models from the directly saved checkpoints."""
    models = {}
    
    # Define custom objects for loading
    custom_objects = {
        'exact_match_metric': exact_match_metric,
        'partial_match_metric': partial_match_metric,
    }
    
    try:
        # Try to load LSTM model
        lstm_path = "outputs/training/lstm_checkpoint_20250505_183215.h5"
        if os.path.exists(lstm_path):
            logger.info(f"Loading LSTM model from {lstm_path}")
            models['lstm'] = load_model(lstm_path, custom_objects=custom_objects)
        
        # Try to load CNN-LSTM model
        cnn_lstm_path = "outputs/training/cnn_lstm_checkpoint_20250505_183342.h5"
        if os.path.exists(cnn_lstm_path):
            logger.info(f"Loading CNN-LSTM model from {cnn_lstm_path}")
            models['cnn_lstm'] = load_model(cnn_lstm_path, custom_objects=custom_objects)
        
        if not models:
            logger.warning("No models found in the outputs/training directory")
        
        return models
    
    except Exception as e:
        logger.error(f"Error loading models directly: {str(e)}")
        traceback.print_exc()
        return {}

def load_models():
    """Load trained models from checkpoints directory."""
    # Try to load directly from saved checkpoints first
    direct_models = load_directly()
    if direct_models and len(direct_models) > 0:
        logger.info(f"Loaded {len(direct_models)} models directly from checkpoints")
        return direct_models
    
    # Fall back to pickle loading if direct loading fails
    models_path = Path('models/checkpoints')
    
    # Load LSTM model
    lstm_path = models_path / "trained_models.pkl"
    if not lstm_path.exists():
        logger.error(f"No models found at {lstm_path}")
        return None
    
    try:
        # Define custom objects for keras model loading
        custom_objects = {
            'exact_match_metric': exact_match_metric,
            'partial_match_metric': partial_match_metric,
        }
        
        # Set custom objects globally
        with tf.keras.utils.custom_object_scope(custom_objects):
            # Load the models from pickle file
            with open(lstm_path, 'rb') as f:
                models_dict = pickle.load(f)
        
        logger.info(f"Successfully loaded {len(models_dict)} models from {lstm_path}")
        
        # Return loaded models
        return models_dict
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        traceback.print_exc()
        return None

def create_ensemble(models):
    """Create a simple ensemble from the loaded models."""
    if not models or len(models) < 1:
        logger.error("No models available for ensemble creation")
        return None
    
    # Default weights (equal weighting)
    weights = {name: 1.0 / len(models) for name in models.keys()}
    
    # If we have an existing config file with optimized weights, load it
    weights_file = Path("outputs/training/ensemble_weights.json")
    if weights_file.exists():
        try:
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
                if 'weights' in weights_data:
                    loaded_weights = weights_data['weights']
                    # Filter to only include models we have
                    weights = {k: v for k, v in loaded_weights.items() if k in models}
                    # Normalize weights
                    total = sum(weights.values())
                    weights = {k: v/total for k, v in weights.items()}
        except Exception as e:
            logger.warning(f"Error loading optimized weights: {str(e)}")
    
    logger.info(f"Using weights: {weights}")
    
    return {
        'models': models,
        'weights': weights
    }

def prepare_input_data(df):
    """Prepare the latest data for prediction."""
    try:
        # Get the most recent sequence of draws
        recent_records = min(SEQUENCE_LENGTH, len(df))
        recent_data = df.tail(recent_records)
        
        # Extract features
        basic_features = []
        for idx, row in recent_data.iterrows():
            numbers = []
            for col in [col for col in row.index if col.startswith('number')]:
                if pd.notna(row[col]):
                    numbers.append(row[col])
            basic_features.append(numbers)
        
        # If we don't have enough data, pad with the first entry
        while len(basic_features) < SEQUENCE_LENGTH:
            basic_features.insert(0, basic_features[0])
        
        # Normalize and create feature matrix
        X = np.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT))
        
        # Fill basic features (lottery numbers)
        for i, nums in enumerate(basic_features[-SEQUENCE_LENGTH:]):
            # Normalize by dividing by 59 (max lottery number)
            normalized = [n/59.0 for n in nums]
            
            # Fill the basic 6 numbers
            for j, num in enumerate(normalized[:6]):
                X[0, i, j] = num
                
            # Add derived features
            if len(nums) >= 6:
                # Sum
                X[0, i, 6] = sum(nums) / 354.0  # max sum would be 59*6
                # Mean
                X[0, i, 7] = sum(nums) / len(nums) / 59.0
                # Min
                X[0, i, 8] = min(nums) / 59.0
                # Max
                X[0, i, 9] = max(nums) / 59.0
                
                # Calculate differences between consecutive numbers
                diffs = [abs(nums[j+1] - nums[j]) for j in range(len(nums)-1)]
                if diffs:
                    # Sum of differences
                    X[0, i, 10] = sum(diffs) / (58 * 5)  # max diff sum
                    # Min diff
                    X[0, i, 11] = min(diffs) / 58.0
                    # Max diff
                    X[0, i, 12] = max(diffs) / 58.0
                    # Range
                    X[0, i, 13] = (max(nums) - min(nums)) / 58.0
                    # Standard deviation
                    X[0, i, 14] = np.std(nums) / 30.0  # approximate max std
        
        logger.info(f"Prepared input with shape {X.shape}")
        return X
    
    except Exception as e:
        logger.error(f"Error preparing input data: {str(e)}")
        traceback.print_exc()
        return np.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT))

def predict_with_model(model, X, model_type='lstm'):
    """Make a prediction with the given model."""
    try:
        # Different handling based on model type
        if model_type.lower() in ['lstm', 'cnn_lstm']:
            # Use 3D input directly
            pred = model.predict(X, verbose=0)
        else:
            # Flatten for tree-based models
            X_flat = X.reshape(X.shape[0], -1)
            pred = model.predict(X_flat)
        
        # Denormalize - multiply by 59 (max lottery number) and round
        if len(pred.shape) > 1 and pred.shape[0] > 0:
            numbers = np.round(pred[0] * 59.0).astype(int)
        else:
            numbers = np.round(pred * 59.0).astype(int)
        
        # Ensure valid lottery numbers (1-59, unique, sorted)
        valid_numbers = ensure_valid_numbers(numbers)
        
        return valid_numbers
    
    except Exception as e:
        logger.error(f"Error predicting with {model_type} model: {str(e)}")
        traceback.print_exc()
        return None

def ensure_valid_numbers(numbers, min_val=1, max_val=59, count=6):
    """Ensure we have valid lottery numbers (unique, within range)."""
    # Convert to python integers and clip to valid range
    numbers = [int(np.clip(n, min_val, max_val)) for n in numbers]
    
    # Get unique numbers
    valid_numbers = []
    for num in numbers:
        if num not in valid_numbers:
            valid_numbers.append(num)
    
    # If we don't have enough, add some random ones
    while len(valid_numbers) < count:
        num = random.randint(min_val, max_val)
        if num not in valid_numbers:
            valid_numbers.append(num)
    
    # If we have too many, take just what we need
    if len(valid_numbers) > count:
        valid_numbers = valid_numbers[:count]
    
    # Sort the numbers
    return sorted(valid_numbers)

def predict_with_ensemble(ensemble, X):
    """Make a prediction using the ensemble of models."""
    if not ensemble or 'models' not in ensemble or not ensemble['models']:
        logger.warning("No ensemble models available")
        return None
    
    # Get predictions from each model
    all_predictions = []
    total_weight = 0
    
    for model_name, model in ensemble['models'].items():
        try:
            prediction = predict_with_model(model, X, model_type=model_name)
            if prediction is not None:
                weight = ensemble['weights'].get(model_name, 1.0)
                all_predictions.append((prediction, weight))
                total_weight += weight
        except Exception as e:
            logger.error(f"Error getting prediction from {model_name}: {str(e)}")
    
    if not all_predictions:
        logger.warning("No valid predictions from any model")
        return None
    
    # Different ways to combine predictions
    combination_method = "frequency"  # Options: weighted_mean, frequency, consensus
    
    if combination_method == "weighted_mean":
        # Weighted mean of predictions (normalize by dividing by max number)
        weighted_sum = np.zeros(6)
        for pred, weight in all_predictions:
            weighted_sum += np.array(pred) * weight
        
        # Normalize and round
        if total_weight > 0:
            final_pred = np.round(weighted_sum / total_weight).astype(int)
        else:
            final_pred = np.round(weighted_sum / len(all_predictions)).astype(int)
        
        return ensure_valid_numbers(final_pred)
    
    elif combination_method == "frequency":
        # Count frequency of each number across all predictions
        frequency = {}
        for pred, weight in all_predictions:
            for num in pred:
                frequency[num] = frequency.get(num, 0) + weight
        
        # Sort by frequency and take top 6
        top_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:6]
        return sorted([num for num, _ in top_numbers])
    
    elif combination_method == "consensus":
        # Take numbers that appear in majority of predictions
        counts = {}
        for pred, _ in all_predictions:
            for num in pred:
                counts[num] = counts.get(num, 0) + 1
        
        # Sort by count
        consensus = [num for num, count in counts.items() 
                    if count >= len(all_predictions) / 2]
        
        # If we don't have enough, add highest counts
        if len(consensus) < 6:
            additional = sorted([(num, count) for num, count in counts.items() 
                               if num not in consensus], 
                              key=lambda x: x[1], reverse=True)
            
            for num, _ in additional:
                if len(consensus) < 6:
                    consensus.append(num)
                else:
                    break
        
        return sorted(consensus[:6])
    
    else:
        # Default: Just return the prediction from the highest-weighted model
        best_prediction = max(all_predictions, key=lambda x: x[1])[0]
        return best_prediction

def perturb_prediction(prediction, intensity=0.3):
    """Create a perturbed version of the prediction with random changes."""
    # Copy the prediction
    perturbed = prediction.copy()
    
    # Number of elements to change (1-3 based on intensity)
    num_changes = max(1, min(3, int(round(intensity * 4))))
    positions = random.sample(range(len(perturbed)), num_changes)
    
    # Change selected positions
    for pos in positions:
        while True:
            # Generate a new number that's not already in the perturbed list
            new_num = random.randint(1, 59)
            if new_num not in perturbed:
                perturbed[pos] = new_num
                break
    
    # Sort the resulting numbers
    return sorted(perturbed)

def generate_predictions(ensemble, df, count=20, diversity=0.3):
    """Generate diverse lottery predictions using the ensemble and perturbation."""
    logger.info(f"Generating {count} diverse predictions...")
    
    # Prepare the input data
    X = prepare_input_data(df)
    
    # Get the base prediction from the ensemble
    base_prediction = predict_with_ensemble(ensemble, X)
    
    if base_prediction is None:
        logger.warning("Ensemble prediction failed, using individual models")
        # Try individual models
        for model_name, model in ensemble['models'].items():
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
    
    # If we still don't have enough predictions, add some random ones
    while len(all_predictions) < count:
        random_pred = sorted(random.sample(range(1, 60), 6))
        if random_pred not in all_predictions:
            all_predictions.append(random_pred)
    
    return all_predictions

def save_predictions(predictions, method="ensemble"):
    """Save predictions to both JSON and formatted text files."""
    # Create timestamps
    timestamp = datetime.now().isoformat()
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # JSON data
    json_data = {
        "timestamp": timestamp,
        "date": date_str,
        "predictions": predictions,
        "metadata": {
            "source": "new_predict.py",
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

def visualize_predictions(predictions):
    """Create visualizations of the predictions."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        plt.figure(figsize=(15, 10))
        
        # 1. Frequency of each number
        plt.subplot(2, 1, 1)
        counts = {}
        for i in range(1, 60):
            counts[i] = 0
            
        for pred in predictions:
            for num in pred:
                counts[num] = counts.get(num, 0) + 1
        
        # Plot frequencies
        x = list(counts.keys())
        y = list(counts.values())
        plt.bar(x, y, color='royalblue')
        plt.title('Frequency of Numbers in Predictions', fontsize=14)
        plt.xlabel('Lottery Number', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(range(0, 60, 5))
        plt.grid(axis='y', alpha=0.3)
        
        # Highlight top 10 most frequent numbers
        top_numbers = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for num, count in top_numbers:
            plt.text(num, count + 0.1, str(num), ha='center', fontweight='bold')
        
        # 2. Heatmap of predictions
        plt.subplot(2, 1, 2)
        data = np.zeros((min(10, len(predictions)), 6))
        for i, pred in enumerate(predictions[:10]):
            for j, num in enumerate(pred):
                data[i, j] = num
        
        plt.imshow(data, cmap='YlOrRd')
        plt.colorbar(label='Number Value')
        plt.title('Top 10 Predictions', fontsize=14)
        plt.yticks(range(min(10, len(predictions))), [f'Pred {i+1}' for i in range(min(10, len(predictions)))])
        plt.xticks(range(6), ['Num 1', 'Num 2', 'Num 3', 'Num 4', 'Num 5', 'Num 6'])
        
        # Add number text on each cell
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, int(data[i, j]), ha='center', va='center', 
                        color='black' if data[i, j] < 30 else 'white')
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = VISUALIZATION_DIR / f"prediction_viz_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {viz_path}")
        return viz_path
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        traceback.print_exc()
        return None

def generate_monte_carlo_predictions(count=20):
    """Generate fallback predictions using Monte Carlo simulation."""
    logger.info(f"Generating {count} Monte Carlo predictions as fallback")
    
    predictions = []
    for _ in range(count):
        # Simple randomized selection
        prediction = sorted(random.sample(range(1, 60), 6))
        predictions.append(prediction)
    
    return predictions

def main():
    """Main function to generate lottery predictions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate lottery predictions')
    parser.add_argument('--count', type=int, default=20, help='Number of predictions to generate')
    parser.add_argument('--diversity', type=float, default=0.3, help='Diversity level (0-1)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--fallback', action='store_true', help='Use fallback Monte Carlo predictions')
    args = parser.parse_args()
    
    try:
        # Check if fallback mode is enabled
        if args.fallback:
            logger.info("Using fallback mode with Monte Carlo predictions")
            predictions = generate_monte_carlo_predictions(count=args.count)
            method = "monte_carlo"
        else:
            # Load models
            logger.info("Loading trained models...")
            models = load_models()
            
            if models is None or len(models) == 0:
                logger.warning("No models available. Using Monte Carlo fallback.")
                predictions = generate_monte_carlo_predictions(count=args.count)
                method = "monte_carlo"
            else:
                logger.info(f"Loaded {len(models)} models: {', '.join(models.keys())}")
                
                # Create ensemble
                ensemble = create_ensemble(models)
                
                # Load lottery data
                logger.info("Loading lottery data...")
                df = load_data(DATA_DIR / "merged_lottery_data.csv")
                logger.info(f"Loaded {len(df)} lottery records")
                
                # Generate predictions
                predictions = generate_predictions(
                    ensemble, 
                    df, 
                    count=args.count,
                    diversity=args.diversity
                )
                method = "ensemble"
        
        # Save predictions
        json_path, text_path = save_predictions(predictions, method=method)
        
        # Create visualization
        if not args.no_viz:
            viz_path = visualize_predictions(predictions)
            if viz_path:
                print(f"Visualization saved to {viz_path}")
        
        # Display predictions
        print("\n" + "="*50)
        print("LOTTERY PREDICTIONS")
        print("="*50 + "\n")
        
        for i, pred in enumerate(predictions, 1):
            pred_str = ", ".join(map(str, pred))
            print(f"{i}. {pred_str}")
        
        print("\n" + "="*50)
        print(f"Generated {len(predictions)} predictions using {method} method")
        print(f"Predictions saved to {text_path}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        traceback.print_exc()
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 