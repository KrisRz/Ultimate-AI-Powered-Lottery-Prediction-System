#!/usr/bin/env python3
"""
Simple UK Lottery Prediction Runner using predictions.py
Generates 10 unique predictions for UK Lotto (6 numbers from 1-59)
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add scripts directory to path to import predictions module
sys.path.append('scripts')

# Import prediction functions
from predictions import (
    ensure_valid_prediction, 
    validate_predictions,
    format_predictions_for_display,
    save_predictions,
    plot_predictions
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_fallback_predictions(n_predictions: int = 10) -> list:
    """
    Generate predictions using various fallback algorithms when models aren't available.
    
    Args:
        n_predictions: Number of predictions to generate
        
    Returns:
        List of prediction lists
    """
    predictions = []
    
    algorithms = [
        "frequency_analysis",
        "fibonacci_sequence", 
        "prime_numbers",
        "sum_targeting",
        "gap_analysis",
        "pattern_based",
        "random_constrained",
        "balanced_odd_even",
        "mathematical_sequence",
        "pure_random"
    ]
    
    for i in range(n_predictions):
        algorithm = algorithms[i % len(algorithms)]
        
        if algorithm == "frequency_analysis":
            # Simulate hot numbers (frequently drawn)
            hot_numbers = [3, 7, 14, 23, 31, 38, 42, 47, 52, 59]
            prediction = np.random.choice(hot_numbers, 6, replace=False)
            
        elif algorithm == "fibonacci_sequence":
            # Use Fibonacci-like sequence
            fib_base = [1, 2, 3, 5, 8, 13, 21, 34, 55]
            prediction = np.random.choice(fib_base, 6, replace=False)
            
        elif algorithm == "prime_numbers":
            # Focus on prime numbers
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
            prediction = np.random.choice(primes, 6, replace=False)
            
        elif algorithm == "sum_targeting":
            # Target specific sum ranges (UK lottery avg sum ~150)
            target_sum = np.random.randint(120, 180)
            prediction = generate_sum_targeted_numbers(target_sum)
            
        elif algorithm == "gap_analysis":
            # Use gap pattern analysis
            prediction = generate_gap_based_numbers()
            
        elif algorithm == "pattern_based":
            # Use sequential patterns
            start = np.random.randint(1, 30)
            prediction = [start + i*7 for i in range(6)]
            prediction = [min(num, 59) for num in prediction]
            
        elif algorithm == "random_constrained":
            # Random with constraints
            prediction = generate_constrained_random()
            
        elif algorithm == "balanced_odd_even":
            # Balance odd and even numbers
            prediction = generate_balanced_odd_even()
            
        elif algorithm == "mathematical_sequence":
            # Use mathematical sequences
            prediction = generate_mathematical_sequence()
            
        else:  # pure_random
            prediction = np.random.choice(range(1, 60), 6, replace=False)
        
        # Ensure prediction is valid
        valid_prediction = ensure_valid_prediction(list(prediction))
        predictions.append(valid_prediction)
    
    return predictions

def generate_sum_targeted_numbers(target_sum: int) -> list:
    """Generate numbers targeting a specific sum."""
    attempts = 0
    while attempts < 100:
        numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
        if abs(sum(numbers) - target_sum) <= 10:
            return numbers
        attempts += 1
    
    # Fallback: adjust random numbers to get closer to target
    numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
    return numbers

def generate_gap_based_numbers() -> list:
    """Generate numbers based on gap analysis."""
    gaps = [2, 3, 5, 7, 11, 13]  # Common gaps
    start = np.random.randint(1, 20)
    numbers = [start]
    
    for gap in gaps[:5]:
        next_num = numbers[-1] + gap
        if next_num <= 59:
            numbers.append(next_num)
        else:
            numbers.append(np.random.randint(1, 59))
    
    return sorted(list(set(numbers))[:6])

def generate_constrained_random() -> list:
    """Generate random numbers with constraints."""
    # Ensure at least one number from each decade
    decades = [[1, 10], [11, 20], [21, 30], [31, 40], [41, 50], [51, 59]]
    numbers = []
    
    for decade in decades[:6]:
        numbers.append(np.random.randint(decade[0], decade[1] + 1))
    
    # Remove duplicates and fill if needed
    numbers = list(set(numbers))
    while len(numbers) < 6:
        numbers.append(np.random.randint(1, 60))
    
    return sorted(numbers[:6])

def generate_balanced_odd_even() -> list:
    """Generate balanced odd/even numbers."""
    odd_nums = list(range(1, 60, 2))
    even_nums = list(range(2, 60, 2))
    
    # 3 odd, 3 even
    odds = np.random.choice(odd_nums, 3, replace=False)
    evens = np.random.choice(even_nums, 3, replace=False)
    
    return sorted(list(odds) + list(evens))

def generate_mathematical_sequence() -> list:
    """Generate numbers using mathematical sequences."""
    # Triangular numbers: n(n+1)/2
    triangular = [n*(n+1)//2 for n in range(1, 12) if n*(n+1)//2 <= 59]
    
    # Square numbers
    squares = [n*n for n in range(1, 8) if n*n <= 59]
    
    # Combine and select
    candidates = list(set(triangular + squares + [6, 28, 496]))  # Perfect numbers
    
    if len(candidates) >= 6:
        return sorted(np.random.choice(candidates, 6, replace=False))
    else:
        # Fill with random if not enough candidates
        additional = np.random.choice(range(1, 60), 6 - len(candidates), replace=False)
        return sorted(candidates + list(additional))

def create_sample_historical_data() -> pd.DataFrame:
    """Create sample historical data for metrics calculation."""
    # Generate sample historical lottery data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='W')
    historical_data = []
    
    for date in dates:
        # Generate realistic lottery numbers
        numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
        historical_data.append({
            'Draw_Date': date,
            'Main_Numbers': numbers
        })
    
    return pd.DataFrame(historical_data)

def main():
    """Main function to generate and display predictions."""
    try:
        print("🎯 UK Lottery Prediction System")
        print("=" * 50)
        print("Generating 10 predictions for UK Lotto (6 numbers from 1-59)...")
        print()
        
        # Generate predictions using fallback algorithms
        predictions = generate_fallback_predictions(10)
        
        # Validate predictions
        validated_predictions, invalid_indices = validate_predictions(predictions)
        
        if invalid_indices:
            logger.warning(f"Fixed {len(invalid_indices)} invalid predictions")
        
        # Create sample historical data for metrics
        historical_data = create_sample_historical_data()
        
        # Format for display
        display_text = format_predictions_for_display(
            validated_predictions,
            title="UK Lottery Predictions - Next Draw"
        )
        
        print(display_text)
        
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/predictions/uk_lottery_predictions_{timestamp}.json"
        
        metadata = {
            "lottery_type": "UK Lotto",
            "number_range": "1-59",
            "numbers_per_draw": 6,
            "generation_method": "Fallback algorithms",
            "algorithms_used": [
                "frequency_analysis", "fibonacci_sequence", "prime_numbers",
                "sum_targeting", "gap_analysis", "pattern_based",
                "random_constrained", "balanced_odd_even", 
                "mathematical_sequence", "pure_random"
            ]
        }
        
        success = save_predictions(
            validated_predictions,
            metadata=metadata,
            output_path=output_path
        )
        
        if success:
            print(f"✅ Predictions saved to: {output_path}")
        
        # Create visualization
        plot_success = plot_predictions(
            validated_predictions,
            f"outputs/visualizations/uk_lottery_predictions_{timestamp}.png"
        )
        
        if plot_success:
            print(f"📊 Visualization saved to: outputs/visualizations/uk_lottery_predictions_{timestamp}.png")
        
        # Display summary statistics
        print("\n📈 PREDICTION SUMMARY:")
        print(f"Total predictions: {len(validated_predictions)}")
        
        all_numbers = [num for pred in validated_predictions for num in pred]
        print(f"Most frequent numbers: {sorted(set(all_numbers), key=all_numbers.count, reverse=True)[:10]}")
        
        sums = [sum(pred) for pred in validated_predictions]
        print(f"Sum range: {min(sums)} - {max(sums)} (avg: {sum(sums)/len(sums):.1f})")
        
        print("\n🎲 Good luck with your lottery predictions!")
        print("Remember: Lottery numbers are random - these are for entertainment only!")
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 