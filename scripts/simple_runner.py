#!/usr/bin/env python3
"""
Simple runner for predictions.py functions
Generates UK lottery predictions without requiring TensorFlow or complex models
"""

import sys
import os
import numpy as np
import pandas as pd
import random
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

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

def generate_mathematical_predictions(n_predictions=10):
    """Generate predictions using mathematical algorithms."""
    predictions = []
    
    # Different mathematical approaches
    algorithms = [
        "frequency_simulation",
        "fibonacci_based", 
        "prime_numbers",
        "sum_targeting",
        "gap_analysis",
        "pattern_sequential",
        "constrained_random",
        "balanced_distribution",
        "mathematical_series",
        "random_weighted"
    ]
    
    for i in range(n_predictions):
        algorithm = algorithms[i % len(algorithms)]
        
        if algorithm == "frequency_simulation":
            # Simulate frequently drawn numbers
            hot_numbers = [3, 7, 14, 23, 31, 38, 42, 47, 52, 59]
            prediction = np.random.choice(hot_numbers, 6, replace=False)
            
        elif algorithm == "fibonacci_based":
            # Use Fibonacci-like progression
            fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55]
            prediction = np.random.choice(fib_numbers, 6, replace=False)
            
        elif algorithm == "prime_numbers":
            # Focus on prime numbers
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
            prediction = np.random.choice(primes, 6, replace=False)
            
        elif algorithm == "sum_targeting":
            # Target specific sum ranges
            prediction = generate_sum_targeted(random.randint(120, 200))
            
        elif algorithm == "gap_analysis":
            # Use gap patterns
            prediction = generate_gap_pattern()
            
        elif algorithm == "pattern_sequential":
            # Sequential patterns
            start = random.randint(1, 25)
            step = random.choice([3, 5, 7, 9])
            prediction = [start + i*step for i in range(6)]
            prediction = [min(num, 59) for num in prediction]
            
        elif algorithm == "constrained_random":
            # Random with decade constraints
            prediction = generate_decade_spread()
            
        elif algorithm == "balanced_distribution":
            # Balance odd/even, high/low
            prediction = generate_balanced_numbers()
            
        elif algorithm == "mathematical_series":
            # Mathematical sequences
            prediction = generate_math_sequence()
            
        else:  # random_weighted
            # Weighted random selection
            weights = [1.5 if i <= 30 else 0.8 for i in range(1, 60)]
            prediction = np.random.choice(range(1, 60), 6, replace=False, p=np.array(weights)/sum(weights))
        
        # Ensure prediction is valid
        valid_prediction = ensure_valid_prediction(list(prediction))
        predictions.append(valid_prediction)
    
    return predictions

def generate_sum_targeted(target_sum):
    """Generate numbers targeting a specific sum."""
    attempts = 0
    while attempts < 50:
        numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
        if abs(sum(numbers) - target_sum) <= 15:
            return numbers
        attempts += 1
    
    # Fallback
    return sorted(np.random.choice(range(1, 60), 6, replace=False))

def generate_gap_pattern():
    """Generate numbers based on gap patterns."""
    gaps = [2, 3, 5, 7, 11, 13]  # Prime gaps
    start = random.randint(1, 15)
    numbers = [start]
    
    for gap in gaps[:5]:
        next_num = numbers[-1] + gap
        if next_num <= 59:
            numbers.append(next_num)
        else:
            numbers.append(random.randint(1, 59))
    
    # Remove duplicates and ensure we have 6 unique numbers
    unique_numbers = list(set(numbers))
    while len(unique_numbers) < 6:
        unique_numbers.append(random.randint(1, 59))
    
    return sorted(unique_numbers[:6])

def generate_decade_spread():
    """Generate numbers spread across decades."""
    decades = [
        range(1, 11),
        range(11, 21), 
        range(21, 31),
        range(31, 41),
        range(41, 51),
        range(51, 60)
    ]
    
    numbers = []
    for decade in decades:
        if len(numbers) < 6:
            numbers.append(random.choice(decade))
    
    # Remove duplicates and fill if needed
    numbers = list(set(numbers))
    while len(numbers) < 6:
        numbers.append(random.randint(1, 59))
    
    return sorted(numbers[:6])

def generate_balanced_numbers():
    """Generate balanced odd/even and high/low numbers."""
    # 3 odd, 3 even
    odd_range = list(range(1, 60, 2))
    even_range = list(range(2, 60, 2))
    
    odds = random.sample(odd_range, 3)
    evens = random.sample(even_range, 3)
    
    return sorted(odds + evens)

def generate_math_sequence():
    """Generate numbers using mathematical sequences."""
    # Triangular numbers
    triangular = [n*(n+1)//2 for n in range(1, 15) if n*(n+1)//2 <= 59]
    # Square numbers
    squares = [n*n for n in range(1, 8) if n*n <= 59]
    # Cube numbers
    cubes = [n*n*n for n in range(1, 5) if n*n*n <= 59]
    
    candidates = list(set(triangular + squares + cubes + [6, 28]))  # Perfect numbers
    
    if len(candidates) >= 6:
        return sorted(random.sample(candidates, 6))
    else:
        # Fill with random numbers
        additional = random.sample(range(1, 60), 6 - len(candidates))
        return sorted(candidates + additional)

def create_sample_data():
    """Create minimal sample data for metrics."""
    return pd.DataFrame({
        'Main_Numbers': [
            [1, 12, 23, 34, 45, 56],
            [7, 14, 21, 28, 35, 42],
            [3, 9, 15, 27, 33, 51]
        ]
    })

def main():
    """Main execution function."""
    try:
        print("🎯 UK LOTTERY PREDICTION SYSTEM")
        print("=" * 50)
        print("Generating 10 predictions using mathematical algorithms...")
        print()
        
        # Generate predictions
        predictions = generate_mathematical_predictions(10)
        
        # Validate predictions
        validated_predictions, invalid_indices = validate_predictions(predictions)
        
        if invalid_indices:
            logger.warning(f"Fixed {len(invalid_indices)} invalid predictions")
        
        # Display predictions
        display_text = format_predictions_for_display(
            validated_predictions,
            title="UK LOTTERY PREDICTIONS"
        )
        print(display_text)
        
        # Save predictions with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/predictions/mathematical_predictions_{timestamp}.json"
        
        metadata = {
            "lottery_type": "UK Lotto",
            "number_range": "1-59", 
            "numbers_per_draw": 6,
            "generation_method": "Mathematical algorithms",
            "algorithms_used": [
                "frequency_simulation", "fibonacci_based", "prime_numbers",
                "sum_targeting", "gap_analysis", "pattern_sequential", 
                "constrained_random", "balanced_distribution",
                "mathematical_series", "random_weighted"
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
        try:
            plot_success = plot_predictions(
                validated_predictions,
                f"outputs/visualizations/mathematical_predictions_{timestamp}.png"
            )
            
            if plot_success:
                print(f"📊 Visualization saved to: outputs/visualizations/mathematical_predictions_{timestamp}.png")
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")
        
        # Summary statistics
        print("\n📈 PREDICTION SUMMARY:")
        print(f"Total predictions: {len(validated_predictions)}")
        
        all_numbers = [num for pred in validated_predictions for num in pred]
        freq_count = {}
        for num in all_numbers:
            freq_count[num] = freq_count.get(num, 0) + 1
        
        most_frequent = sorted(freq_count.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Most frequent numbers: {[num for num, count in most_frequent]}")
        
        sums = [sum(pred) for pred in validated_predictions]
        print(f"Sum range: {min(sums)} - {max(sums)} (avg: {sum(sums)/len(sums):.1f})")
        
        print("\n🎲 Good luck with your lottery predictions!")
        print("Remember: These are mathematical predictions for entertainment only!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 