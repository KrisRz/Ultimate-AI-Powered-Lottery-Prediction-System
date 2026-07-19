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
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    tf = None
    load_model = None
    TF_AVAILABLE = False

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_data import load_data, DATA_DIR, prepare_sequence_data
MODEL_CONFIG_PATH = Path("models/checkpoints/model_config.json")

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
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available for metric evaluation")
    y_pred_rounded = tf.round(y_pred * 59)
    y_true_scaled = y_true * 59
    matches = tf.reduce_all(tf.equal(y_pred_rounded, y_true_scaled), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

def partial_match_metric(y_true, y_pred):
    """Custom metric: Count partial matches (at least 3 correct numbers)."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available for metric evaluation")
    y_pred_rounded = tf.round(y_pred * 59)
    y_true_scaled = y_true * 59
    matches = tf.cast(tf.equal(y_pred_rounded, y_true_scaled), tf.float32)
    match_counts = tf.reduce_sum(matches, axis=1)
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
        if not TF_AVAILABLE:
            return {}
        # Try to load LSTM model from outputs/training (dated checkpoints)
        lstm_path = "outputs/training/lstm_checkpoint_20250505_183215.h5"
        if os.path.exists(lstm_path):
            logger.info(f"Loading LSTM model from {lstm_path}")
            models['lstm'] = load_model(lstm_path, custom_objects=custom_objects)
        
        # Try to load CNN-LSTM model from outputs/training
        cnn_lstm_path = "outputs/training/cnn_lstm_checkpoint_20250505_183342.h5"
        if os.path.exists(cnn_lstm_path):
            logger.info(f"Loading CNN-LSTM model from {cnn_lstm_path}")
            models['cnn_lstm'] = load_model(cnn_lstm_path, custom_objects=custom_objects)

        # Also scan models/checkpoints for any SavedModels/H5 (e.g., standalone_lstm_model.h5)
        ckpt_dir = Path('models/checkpoints')
        if ckpt_dir.exists():
            for h5 in ckpt_dir.glob('*.h5'):
                try:
                    model_name = 'lstm' if 'lstm' in h5.stem.lower() else h5.stem
                    logger.info(f"Loading model '{model_name}' from {h5}")
                    models[model_name] = load_model(h5, custom_objects=custom_objects)
                except Exception as e:
                    logger.warning(f"Failed to load model from {h5}: {e}")
        
        if not models:
            logger.warning("No models found in known checkpoint locations")
        
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
        logger.error(f"Error loading models from pickle: {str(e)}")
        traceback.print_exc()
        # Return empty to allow caller to handle
        return {}

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

def _load_model_config() -> dict:
    try:
        if MODEL_CONFIG_PATH.exists():
            with open(MODEL_CONFIG_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read model_config.json: {e}")
    return {}


def prepare_input_data(df, model_key: str = 'lstm'):
    """Prepare the latest data for prediction (shape-safe)."""
    try:
        cfg = _load_model_config()
        mcfg = cfg.get('models', {}).get(model_key, {})
        exp_t = mcfg.get('expected_timesteps', SEQUENCE_LENGTH)
        exp_f = mcfg.get('expected_features', 6)
        norm_by = mcfg.get('normalize_by', 59)

        # Get the most recent sequence of draws
        recent_records = min(exp_t, len(df))
        recent_data = df.tail(recent_records)
        
        # Extract features
        basic_features = []
        for idx, row in recent_data.iterrows():
            numbers = []
            # Preferred explicit numbered columns if present
            numbered_cols = [f"Number_{i}" for i in range(1, 7)]
            if all(col in row.index for col in numbered_cols):
                numbers = [int(row[col]) for col in numbered_cols if pd.notna(row[col])]
            elif 'Main_Numbers' in row.index and isinstance(row['Main_Numbers'], (list, tuple)):
                numbers = [int(n) for n in row['Main_Numbers']]
            else:
                # Fallback: case-insensitive scan for number_* columns
                lower_map = {c.lower(): c for c in row.index}
                candidates = [lower_map.get(f'number_{i}'.lower()) for i in range(1, 7)]
                candidates = [c for c in candidates if c is not None]
                for col in candidates:
                    if pd.notna(row[col]):
                        numbers.append(int(row[col]))
            basic_features.append(numbers)
        
        # If we don't have enough data, pad with the first entry
        while len(basic_features) < exp_t:
            basic_features.insert(0, basic_features[0])
        
        # Normalize and create feature matrix
        X = np.zeros((1, exp_t, max(exp_f, 6)))
        
        # Fill basic features (lottery numbers)
        for i, nums in enumerate(basic_features[-exp_t:]):
            # Normalize by dividing by configured max
            normalized = [n/float(norm_by) for n in nums]
            
            # Fill the basic 6 numbers
            for j, num in enumerate(normalized[:6]):
                X[0, i, j] = num
                
            # Add derived features only if model expects >6 features
            if exp_f > 6 and len(nums) >= 6:
                # Sum, mean, min, max
                X[0, i, 6] = sum(nums) / float(norm_by * 6)
                X[0, i, 7] = sum(nums) / len(nums) / float(norm_by)
                X[0, i, 8] = min(nums) / float(norm_by)
                X[0, i, 9] = max(nums) / float(norm_by)

                # Differences
                diffs = [abs(nums[j+1] - nums[j]) for j in range(len(nums)-1)]
                if diffs and exp_f > 10:
                    X[0, i, 10] = sum(diffs) / (58.0 * 5)
                    if exp_f > 11:
                        X[0, i, 11] = min(diffs) / 58.0
                    if exp_f > 12:
                        X[0, i, 12] = max(diffs) / 58.0
                    if exp_f > 13:
                        X[0, i, 13] = (max(nums) - min(nums)) / 58.0
                    if exp_f > 14:
                        X[0, i, 14] = np.std(nums) / 30.0
        
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
            # Ensure 3D input and match expected (timesteps, features)
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                # model.input_shape: (None, timesteps, features)
                _, exp_t, exp_f = model.input_shape
                cur_t = X.shape[1]
                cur_f = X.shape[2] if len(X.shape) > 2 else 1
                # Adjust timesteps: use last exp_t
                if exp_t and cur_t != exp_t:
                    if cur_t > exp_t:
                        X = X[:, -exp_t:, :]
                    else:
                        # pad at the front by repeating first frame
                        pad = np.repeat(X[:, :1, :], exp_t - cur_t, axis=1)
                        X = np.concatenate([pad, X], axis=1)
                # Adjust features: keep first exp_f (first 6 slots are basic numbers)
                if exp_f and cur_f != exp_f:
                    if cur_f > exp_f:
                        X = X[:, :, :exp_f]
                    else:
                        pad_f = np.zeros((X.shape[0], X.shape[1], exp_f - cur_f))
                        X = np.concatenate([X, pad_f], axis=2)
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

def predict_with_ensemble(ensemble, X, combination_method: str = "frequency"):
    """Make a prediction using the ensemble of models.

    Args:
        ensemble: dict with 'models' and 'weights'
        X: input array (3D for LSTM)
        combination_method: 'frequency' | 'weighted' | 'consensus'
    """
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
    if combination_method == "weighted":
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
    
    elif combination_method == "probmap":
        # Build a probability map over 1..59 by accumulating weighted occurrences
        scores = {n: 0.0 for n in range(1, 60)}
        for pred, weight in all_predictions:
            for n in pred:
                scores[n] += float(weight)
        # Pick top-6 with a simple spacing heuristic
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen = []
        for n, _ in ranked:
            if all(abs(n - m) >= 1 for m in chosen):  # minimal spacing 1 (can be tuned)
                chosen.append(n)
            if len(chosen) == 6:
                break
        return sorted(chosen)
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
    
    # Change selected positions with basic constraints on sum and gaps
    for pos in positions:
        while True:
            # Generate a new number that's not already in the perturbed list
            new_num = random.randint(1, 59)
            if new_num not in perturbed:
                trial = perturbed.copy()
                trial[pos] = new_num
                # Enforce simple bands around sum and gap distribution
                s = sum(trial)
                if 80 <= s <= 300:  # permissive band
                    gaps = [abs(trial[j+1] - trial[j]) for j in range(len(trial)-1)]
                    if gaps:
                        if 0 <= min(gaps) <= 20 and 0 <= max(gaps) <= 58:
                            perturbed[pos] = new_num
                            break
    
    # Sort the resulting numbers
    return sorted(perturbed)

def generate_predictions(ensemble, df, count=20, diversity=0.3, combination_method: str = "frequency"):
    """Generate diverse lottery predictions using the ensemble and perturbation."""
    logger.info(f"Generating {count} diverse predictions...")
    
    # Prepare the input data
    X = prepare_input_data(df)
    
    # Get the base prediction from the ensemble
    base_prediction = predict_with_ensemble(ensemble, X, combination_method=combination_method)
    
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


def optimize_portfolio(
    predictions: list[list[int]],
    target_size: int = 10,
    cap_number_usage: int = 0,
    require_high_per_line: bool = False,
    decade_balance: bool = False,
    wildcards: int = 0,
) -> list[list[int]]:
    """Greedy portfolio optimizer to reduce overlap and increase coverage.

    - Penalize repeated pairs across lines
    - Encourage decade coverage and include at least one high number (>=50)
    - Cap dominance of any single number
    """
    if not predictions:
        return predictions

    # Parameters
    max_occ_per_number = cap_number_usage or max(2, target_size // 3)
    required_high_numbers = 3  # across portfolio

    chosen: list[list[int]] = []
    pair_counts: dict[tuple[int, int], int] = {}
    num_counts: dict[int, int] = {}
    high_count = 0

    # Score function
    def score(line: list[int]) -> float:
        line_sorted = sorted(line)
        # Penalize repeated pairs
        pairs = [(line_sorted[i], line_sorted[j]) for i in range(6) for j in range(i+1, 6)]
        pair_penalty = sum(pair_counts.get(p, 0) for p in pairs)
        # Encourage decade coverage
        decades = {n // 10 for n in line_sorted}
        decade_bonus = (len(decades) / 6.0) if decade_balance else 0.0
        # Encourage high numbers inclusion (>=50)
        high_bonus = 0.3 if require_high_per_line and any(n >= 50 for n in line_sorted) else 0.0
        # Penalize number overuse
        overuse_penalty = sum(max(0, num_counts.get(n, 0) - 1) for n in line_sorted)
        return decade_bonus + high_bonus - 0.2 * pair_penalty - 0.1 * overuse_penalty

    pool = [sorted(p) for p in predictions]
    # Greedy selection
    # Wildcard selection: pick a few lines that are farthest from current stats
    wildcard_selected = 0
    while pool and len(chosen) < target_size:
        if wildcards and wildcard_selected < wildcards:
            # select by maximum novelty: fewest overlaps with current counts
            def novelty(line: list[int]) -> float:
                return -sum(num_counts.get(n, 0) for n in line)  # fewer overlaps is better
            best = max(pool, key=novelty)
            wildcard_selected += 1
        else:
            best = max(pool, key=score)
        chosen.append(best)
        # Update stats
        for i in range(6):
            num_counts[best[i]] = num_counts.get(best[i], 0) + 1
            if best[i] >= 50:
                high_count += 1
            for j in range(i+1, 6):
                pair = (best[i], best[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pool.remove(best)

    # Post-adjustment: reduce dominance and ensure high-number presence
    # If dominance of any number is too high, try replacing last lines with alternates
    dominant = {n for n, c in num_counts.items() if c > max_occ_per_number}
    if dominant or high_count < required_high_numbers:
        # Try to replace last half with lines that relieve dominance and add highs
        replace_pool = [p for p in predictions if p not in chosen]
        for idx in range(len(chosen)//2, len(chosen)):
            cand_best = chosen[idx]
            if any(n in dominant for n in cand_best) or (high_count < required_high_numbers and not any(n >= 50 for n in cand_best)):
                # find a better candidate
                def repl_score(line: list[int]) -> float:
                    s = score(line)
                    if any(n in dominant for n in line):
                        s -= 0.5
                    if require_high_per_line and any(n >= 50 for n in line):
                        s += 0.2
                    return s
                if replace_pool:
                    alt = max(replace_pool, key=repl_score)
                    if repl_score(alt) > repl_score(cand_best):
                        chosen[idx] = alt
                        replace_pool.remove(alt)
    return [sorted(p) for p in chosen[:target_size]]

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


def analyze_portfolio(predictions: list[list[int]]) -> dict:
    # Frequency of numbers
    freq: dict[int, int] = {}
    for line in predictions:
        for n in line:
            freq[n] = freq.get(n, 0) + 1

    # Top repeats
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    # Range
    flat = [n for line in predictions for n in line]
    spread = (min(flat), max(flat)) if flat else (None, None)

    # Pair counts
    pair_counts: dict[tuple[int, int], int] = {}
    for line in predictions:
        s = sorted(line)
        for i in range(6):
            for j in range(i+1, 6):
                pair = (s[i], s[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    top_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # Decade distribution
    decades = {}
    for n in flat:
        d = (n // 10) * 10
        decades[d] = decades.get(d, 0) + 1

    # High-number coverage
    high_count = sum(1 for n in flat if n >= 50)

    return {
        'top_numbers': top[:10],
        'spread': spread,
        'top_pairs': [(list(p), c) for p, c in top_pairs],
        'decades': decades,
        'high_numbers_count': high_count,
    }

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
    parser.add_argument('--ensemble', choices=['frequency', 'weighted', 'consensus', 'probmap'], default='frequency',
                        help='Ensemble combination method')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--fallback', action='store_true', help='Use fallback Monte Carlo predictions (explicit only)')
    parser.add_argument('--optimize-coverage', action='store_true', help='Apply portfolio diversity/coverage optimizer')
    parser.add_argument('--no-analyze', action='store_true', help='Skip portfolio analytics summary')
    parser.add_argument('--cap-number-usage', type=int, default=0, help='Cap usage of any single number across the portfolio (0=off)')
    parser.add_argument('--require-high-per-line', action='store_true', help='Ensure each line has at least one number >= 50')
    parser.add_argument('--decade-balance', action='store_true', help='Favor balancing decades across the portfolio')
    parser.add_argument('--wildcards', type=int, default=0, help='Number of wildcard lines to force away from core clusters (0=off)')
    args = parser.parse_args()
    
    try:
        # Always attempt to update data before prediction
        try:
            from scripts.fetch_data import download_fresh_data
            _ = download_fresh_data()
        except Exception as _upd_err:
            logger.warning(f"Could not update data before predictions: {_upd_err}")

        # Apply sensible defaults for coverage optimization if enabled but unspecified
        if args.optimize_coverage and not any([
            args.cap_number_usage > 0,
            args.require_high_per_line,
            args.decade_balance,
            args.wildcards > 0,
        ]):
            # Defaults tuned for 10 lines
            args.cap_number_usage = max(4, int(round(args.count * 0.6)))
            args.require_high_per_line = True
            args.decade_balance = True
            args.wildcards = max(2, args.count // 5)

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
                raise RuntimeError(
                    "No trained models found. Run `python scripts/main.py --retrain yes --force` to train before predicting."
                )

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
                diversity=args.diversity,
                combination_method=args.ensemble,
            )
            if args.optimize_coverage:
                predictions = optimize_portfolio(
                    predictions,
                    target_size=args.count,
                    cap_number_usage=args.cap_number_usage,
                    require_high_per_line=args.require_high_per_line,
                    decade_balance=args.decade_balance,
                    wildcards=args.wildcards,
                )
            method = "ensemble"
        
        # Save predictions
        json_path, text_path = save_predictions(predictions, method=method)

        # Portfolio analytics summary
        if not args.no_analyze:
            summary = analyze_portfolio(predictions)
            print("\nPORTFOLIO ANALYSIS")
            print(f"Top numbers: {summary['top_numbers']}")
            print(f"Range: {summary['spread'][0]} → {summary['spread'][1]}")
            print(f"Top pairs: {summary['top_pairs']}")
            print(f"Decades: {summary['decades']}")
            print(f"High numbers (>=50) count across portfolio: {summary['high_numbers_count']}")
        
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