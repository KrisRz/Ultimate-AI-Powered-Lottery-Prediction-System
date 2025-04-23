import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time
import traceback
import ast
import matplotlib.pyplot as plt
from collections import Counter

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load data directly to bypass complex loading
def load_simple_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert string representations of lists to actual lists
    if 'Main_Numbers' in df.columns:
        try:
            df['Main_Numbers'] = df['Main_Numbers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            logger.info(f"Successfully processed Main_Numbers column")
        except Exception as e:
            logger.warning(f"Could not convert Main_Numbers column: {e}")
    
    # Drop rows with invalid Main_Numbers
    if 'Main_Numbers' in df.columns:
        valid_rows = df['Main_Numbers'].apply(lambda x: isinstance(x, list) and len(x) == 6)
        df = df[valid_rows].reset_index(drop=True)
        logger.info(f"Kept {len(df)} rows with valid main numbers")
    
    return df

# Simple XGBoost-only training function
def train_xgboost_simple(df):
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    
    logger.info("Preparing features for XGBoost")
    
    # Extract features from raw data
    features = []
    targets = []
    
    # We'll use simple features - just the numbers from previous draws
    window_size = 10  # Use last 10 draws to predict next
    
    for i in range(window_size, len(df)):
        # Create feature vector from previous draws
        row_features = []
        for j in range(i - window_size, i):
            numbers = df['Main_Numbers'].iloc[j]
            row_features.extend(numbers)
        
        # Target is the next draw
        target = df['Main_Numbers'].iloc[i]
        
        features.append(row_features)
        targets.append(target)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(targets)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training data shape: {X_train.shape}")
    
    # Train model
    logger.info("Training XGBoost model")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluate model
    score = model.score(X_test, y_test)
    logger.info(f"Model R² score: {score:.4f}")
    
    return model

# Simple prediction function
def predict_next_draw(model, df, n_predictions=5):
    logger.info("Generating predictions")
    
    # Create feature vector from last draws
    window_size = 10
    last_features = []
    
    for j in range(len(df) - window_size, len(df)):
        numbers = df['Main_Numbers'].iloc[j]
        last_features.extend(numbers)
    
    # Reshape for model
    X_pred = np.array([last_features])
    
    # Generate predictions
    predictions = []
    for i in range(n_predictions):
        # Get raw prediction
        raw_pred = model.predict(X_pred)[0]
        
        # Add some randomness to diversify predictions
        noise = np.random.normal(0, 3, size=raw_pred.shape)  # Normal noise with std deviation 3
        raw_pred = raw_pred + noise
        
        # Round and clip values
        pred = np.round(raw_pred).astype(int)
        pred = np.clip(pred, 1, 59)
        
        # Ensure unique values
        unique_pred = set(pred)
        
        # If the model gives too few unique numbers, add random ones
        while len(unique_pred) < 6:
            # Choose from higher probability numbers (1-49 range is common in lotteries)
            new_num = np.random.randint(1, 50)
            if new_num not in unique_pred:
                unique_pred.add(new_num)
        
        # If we have more than 6 numbers, take the 6 with highest predicted values
        if len(unique_pred) > 6:
            # Create dictionary of number -> predicted value
            num_vals = {num: raw_pred[i] for i, num in enumerate(pred) if num in unique_pred}
            # Sort by predicted value (descending) and take top 6
            unique_pred = [k for k, v in sorted(num_vals.items(), key=lambda item: -item[1])[:6]]
        
        # Sort and save
        sorted_pred = sorted(list(unique_pred))[:6]
        predictions.append(sorted_pred)
    
    return predictions

# Analyze historical data frequencies
def analyze_number_frequencies(df):
    logger.info("Analyzing number frequencies")
    
    # Flatten all numbers
    all_numbers = []
    for nums in df['Main_Numbers']:
        all_numbers.extend(nums)
    
    # Count frequencies
    counter = Counter(all_numbers)
    
    # Convert to percentages
    total_draws = len(df)
    percentages = {num: (count / total_draws) * 100 for num, count in counter.items()}
    
    # Sort by number
    sorted_percentages = {k: percentages[k] for k in sorted(percentages.keys())}
    
    return sorted_percentages

# Visualize frequencies and predictions
def visualize_results(frequencies, predictions, output_path='results/number_frequencies.png'):
    logger.info("Creating visualization")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Number frequencies
    numbers = list(frequencies.keys())
    freqs = list(frequencies.values())
    
    bars = ax1.bar(numbers, freqs, color='skyblue')
    ax1.set_title('Lottery Number Frequencies (%)')
    ax1.set_xlabel('Number')
    ax1.set_ylabel('Frequency (%)')
    ax1.set_xticks(range(1, 60, 5))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Get average frequency
    avg_freq = sum(freqs) / len(freqs)
    ax1.axhline(y=avg_freq, color='r', linestyle='-', label=f'Average ({avg_freq:.2f}%)')
    ax1.legend()
    
    # Highlight predicted numbers in the frequency chart
    # Combine all predictions into a single set of unique numbers
    all_predicted = set()
    for pred in predictions:
        all_predicted.update(pred)
    
    # Highlight the bars corresponding to predicted numbers
    for i, num in enumerate(numbers):
        if num in all_predicted:
            bars[i].set_color('orange')
    
    # Plot 2: Prediction heatmap
    prediction_matrix = np.zeros((len(predictions), 60))
    for i, pred in enumerate(predictions):
        for num in pred:
            prediction_matrix[i, num] = 1
    
    # Create heatmap
    ax2.imshow(prediction_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_title('Predictions')
    ax2.set_xlabel('Number')
    ax2.set_ylabel('Prediction #')
    ax2.set_xticks(range(0, 60, 5))
    ax2.set_xticklabels(range(0, 60, 5))
    ax2.set_yticks(range(len(predictions)))
    ax2.set_yticklabels([f'Prediction {i+1}' for i in range(len(predictions))])
    
    # Add annotations
    for i in range(len(predictions)):
        for j in range(1, 60):
            if prediction_matrix[i, j] == 1:
                ax2.text(j, i, str(j), ha='center', va='center', color='black', fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")
    
    return fig

def main():
    try:
        logger.info("Starting simplified lottery prediction")
        
        # Load data from the fixed data file
        data_path = "data/fixed_lottery_data.csv"
        df = load_simple_data(data_path)
        logger.info(f"Loaded {len(df)} lottery draws")
        
        # Analyze historical frequencies
        frequencies = analyze_number_frequencies(df)
        
        # Train model
        model = train_xgboost_simple(df)
        
        # Generate predictions
        predictions = predict_next_draw(model, df, n_predictions=5)
        
        # Create visualization
        visualize_results(frequencies, predictions)
        
        # Display predictions
        print("\n" + "="*50)
        print("LOTTERY PREDICTIONS")
        print("="*50)
        for i, pred in enumerate(predictions, 1):
            print(f"Prediction {i}: {' - '.join(f'{n:02d}' for n in pred)}")
        print("="*50 + "\n")
        print(f"Visualization saved to results/number_frequencies.png")
        print("="*50 + "\n")
        
        logger.info("Prediction process completed")
        return True
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 