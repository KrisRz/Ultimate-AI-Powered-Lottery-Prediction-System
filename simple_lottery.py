#!/usr/bin/env python3
"""
Simple UK Lottery Number Prediction System
Generates 10 unique predictions for UK Lotto (6 numbers from 1-59)
"""

import random
import numpy as np
import json
from datetime import datetime
from pathlib import Path

def create_directories():
    """Create output directories"""
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

def frequency_based_prediction():
    """Generate prediction based on simulated frequency analysis"""
    # Simulate hot numbers (frequently drawn)
    hot_numbers = [3, 7, 14, 23, 31, 38, 42, 47, 52, 59]
    # Simulate cold numbers (less frequently drawn)
    cold_numbers = [1, 8, 16, 24, 29, 35, 41, 46, 53, 58]
    # Mix of all numbers
    all_numbers = list(range(1, 60))
    
    # Weighted selection favoring hot numbers
    weights = []
    for num in all_numbers:
        if num in hot_numbers:
            weights.append(3)  # Hot numbers 3x more likely
        elif num in cold_numbers:
            weights.append(0.5)  # Cold numbers less likely
        else:
            weights.append(1)  # Normal weight
    
    prediction = []
    available = all_numbers.copy()
    available_weights = weights.copy()
    
    for _ in range(6):
        # Weighted random selection
        selected_idx = np.random.choice(len(available), p=np.array(available_weights)/sum(available_weights))
        selected_num = available[selected_idx]
        prediction.append(selected_num)
        available.pop(selected_idx)
        available_weights.pop(selected_idx)
    
    return sorted(prediction)

def fibonacci_based_prediction():
    """Generate prediction using Fibonacci sequence patterns"""
    fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    fib_related = []
    
    # Include Fibonacci numbers under 60
    for fib in fib_numbers:
        if fib < 60:
            fib_related.append(fib)
    
    # Add numbers that are sums or differences of Fibonacci numbers
    for i in range(len(fib_numbers)-1):
        for j in range(i+1, len(fib_numbers)):
            sum_val = fib_numbers[i] + fib_numbers[j]
            diff_val = abs(fib_numbers[j] - fib_numbers[i])
            if 1 <= sum_val <= 59:
                fib_related.append(sum_val)
            if 1 <= diff_val <= 59 and diff_val not in fib_related:
                fib_related.append(diff_val)
    
    # Remove duplicates and select 6
    fib_related = list(set(fib_related))
    if len(fib_related) >= 6:
        prediction = random.sample(fib_related, 6)
    else:
        # Fill remaining with random numbers
        prediction = fib_related.copy()
        remaining = [n for n in range(1, 60) if n not in prediction]
        prediction.extend(random.sample(remaining, 6 - len(prediction)))
    
    return sorted(prediction)

def prime_based_prediction():
    """Generate prediction favoring prime numbers"""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    non_primes = [n for n in range(1, 60) if n not in primes]
    
    # Select 3-4 primes and 2-3 non-primes
    num_primes = random.randint(3, 4)
    selected_primes = random.sample(primes, num_primes)
    selected_non_primes = random.sample(non_primes, 6 - num_primes)
    
    prediction = selected_primes + selected_non_primes
    return sorted(prediction)

def sum_range_prediction(target_sum_range=(120, 180)):
    """Generate prediction targeting a specific sum range"""
    while True:
        prediction = random.sample(range(1, 60), 6)
        total = sum(prediction)
        if target_sum_range[0] <= total <= target_sum_range[1]:
            return sorted(prediction)

def gap_analysis_prediction():
    """Generate prediction based on gap analysis between numbers"""
    prediction = []
    current = random.randint(1, 10)  # Start in first decade
    prediction.append(current)
    
    for _ in range(5):
        # Gaps typically range from 1 to 15
        gap = random.choices([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                           weights=[1,2,3,4,5,6,5,4,3,2,2,1,1,1,1])[0]
        current += gap
        if current <= 59:
            prediction.append(current)
        else:
            # Wrap around or pick random
            available = [n for n in range(1, 60) if n not in prediction]
            if available:
                prediction.append(random.choice(available))
    
    # Ensure we have exactly 6 unique numbers
    prediction = list(set(prediction))
    while len(prediction) < 6:
        available = [n for n in range(1, 60) if n not in prediction]
        prediction.append(random.choice(available))
    
    return sorted(prediction[:6])

def pattern_based_prediction():
    """Generate prediction based on number patterns"""
    patterns = [
        # Arithmetic sequence
        lambda: [5, 12, 19, 26, 33, 40],
        # Multiples of specific numbers
        lambda: sorted(random.sample([3*i for i in range(1, 20) if 3*i <= 59], 3) + 
                      random.sample([7*i for i in range(1, 9) if 7*i <= 59], 2) +
                      [random.randint(1, 59)]),
        # Alternating odd/even
        lambda: sorted(random.sample([i for i in range(1, 60, 2)], 3) + 
                      random.sample([i for i in range(2, 60, 2)], 3)),
    ]
    
    pattern_func = random.choice(patterns)
    prediction = pattern_func()
    
    # Ensure unique and exactly 6 numbers
    prediction = list(set(prediction))
    while len(prediction) < 6:
        available = [n for n in range(1, 60) if n not in prediction]
        prediction.append(random.choice(available))
    
    return sorted(prediction[:6])

def random_with_constraints():
    """Generate random prediction with certain constraints"""
    # Ensure good distribution across decades
    decades = [
        list(range(1, 11)),    # 1-10
        list(range(11, 21)),   # 11-20
        list(range(21, 31)),   # 21-30
        list(range(31, 41)),   # 31-40
        list(range(41, 51)),   # 41-50
        list(range(51, 60))    # 51-59
    ]
    
    prediction = []
    # Pick at least one from first 3 decades
    for i in range(3):
        if random.random() < 0.6:  # 60% chance
            prediction.append(random.choice(decades[i]))
    
    # Fill remaining slots
    while len(prediction) < 6:
        available = [n for n in range(1, 60) if n not in prediction]
        prediction.append(random.choice(available))
    
    return sorted(prediction)

def balanced_odd_even():
    """Generate prediction with balanced odd/even numbers"""
    odds = [i for i in range(1, 60, 2)]
    evens = [i for i in range(2, 60, 2)]
    
    # 3 odd, 3 even or 4 odd, 2 even
    if random.random() < 0.7:
        num_odds = 3
    else:
        num_odds = 4
    
    selected_odds = random.sample(odds, num_odds)
    selected_evens = random.sample(evens, 6 - num_odds)
    
    return sorted(selected_odds + selected_evens)

def mathematical_sequence():
    """Generate prediction based on mathematical sequences"""
    sequences = [
        # Powers of 2 (modified)
        [2, 4, 8, 16, 32],
        # Triangular numbers
        [1, 3, 6, 10, 15, 21, 28, 36, 45, 55],
        # Perfect squares
        [1, 4, 9, 16, 25, 36, 49],
    ]
    
    seq = random.choice(sequences)
    # Filter numbers <= 59
    valid_seq = [n for n in seq if n <= 59]
    
    # Take some from sequence and fill with randoms
    num_from_seq = min(random.randint(2, 4), len(valid_seq))
    prediction = random.sample(valid_seq, num_from_seq)
    
    # Fill remaining
    while len(prediction) < 6:
        available = [n for n in range(1, 60) if n not in prediction]
        prediction.append(random.choice(available))
    
    return sorted(prediction)

def generate_all_predictions():
    """Generate 10 different predictions using various algorithms"""
    algorithms = [
        ("Frequency Analysis", frequency_based_prediction),
        ("Fibonacci Sequence", fibonacci_based_prediction),
        ("Prime Numbers", prime_based_prediction),
        ("Sum Range (150±30)", lambda: sum_range_prediction((120, 180))),
        ("Gap Analysis", gap_analysis_prediction),
        ("Pattern Based", pattern_based_prediction),
        ("Random Constrained", random_with_constraints),
        ("Balanced Odd/Even", balanced_odd_even),
        ("Mathematical Sequence", mathematical_sequence),
        ("Pure Random", lambda: sorted(random.sample(range(1, 60), 6)))
    ]
    
    predictions = []
    used_combinations = set()
    
    for name, func in algorithms:
        attempts = 0
        while attempts < 10:  # Max 10 attempts to avoid infinite loop
            try:
                prediction = func()
                prediction_tuple = tuple(prediction)
                
                # Ensure unique combination
                if prediction_tuple not in used_combinations:
                    used_combinations.add(prediction_tuple)
                    predictions.append({
                        "algorithm": name,
                        "numbers": prediction,
                        "sum": sum(prediction),
                        "odd_count": sum(1 for n in prediction if n % 2 == 1),
                        "even_count": sum(1 for n in prediction if n % 2 == 0)
                    })
                    break
            except Exception as e:
                print(f"Error with {name}: {e}")
            attempts += 1
        
        if attempts == 10:
            # Fallback to pure random
            prediction = sorted(random.sample(range(1, 60), 6))
            predictions.append({
                "algorithm": f"{name} (fallback)",
                "numbers": prediction,
                "sum": sum(prediction),
                "odd_count": sum(1 for n in prediction if n % 2 == 1),
                "even_count": sum(1 for n in prediction if n % 2 == 0)
            })
    
    return predictions

def save_predictions(predictions):
    """Save predictions to JSON and text files"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save as JSON
    json_file = f"outputs/predictions/predictions_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "lottery_type": "UK Lotto",
            "number_range": "1-59",
            "numbers_per_draw": 6,
            "predictions": predictions
        }, f, indent=2)
    
    # Save as readable text
    text_file = f"outputs/predictions/predictions_{timestamp}.txt"
    with open(text_file, 'w') as f:
        f.write("UK LOTTERY PREDICTIONS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Numbers: 6 from 1-59\n\n")
        
        for i, pred in enumerate(predictions, 1):
            f.write(f"Prediction {i:2d}: {' '.join(f'{n:2d}' for n in pred['numbers'])} ")
            f.write(f"(Sum: {pred['sum']:3d}, {pred['algorithm']})\n")
    
    return json_file, text_file

def display_predictions(predictions):
    """Display predictions in a nice format"""
    print("\n" + "="*60)
    print("         UK LOTTERY PREDICTIONS")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Format: 6 numbers from 1-59")
    print("-"*60)
    
    for i, pred in enumerate(predictions, 1):
        numbers_str = " ".join(f"{n:2d}" for n in pred["numbers"])
        print(f"Prediction {i:2d}: [{numbers_str}]  Sum: {pred['sum']:3d}  ({pred['algorithm']})")
    
    print("-"*60)
    print("Statistics:")
    total_sum = sum(pred['sum'] for pred in predictions)
    avg_sum = total_sum / len(predictions)
    print(f"Average sum: {avg_sum:.1f}")
    
    all_numbers = [num for pred in predictions for num in pred['numbers']]
    unique_numbers = len(set(all_numbers))
    print(f"Unique numbers used: {unique_numbers}/59")
    print("="*60)

def main():
    """Main function"""
    print("Initializing UK Lottery Prediction System...")
    
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    # np.random.seed(42)
    
    create_directories()
    
    print("Generating 10 unique predictions using various algorithms...")
    predictions = generate_all_predictions()
    
    # Display results
    display_predictions(predictions)
    
    # Save to files
    json_file, text_file = save_predictions(predictions)
    print(f"\nPredictions saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Text: {text_file}")
    
    print(f"\n📋 QUICK REFERENCE - Your 10 Predictions:")
    for i, pred in enumerate(predictions, 1):
        numbers_str = " ".join(f"{n:2d}" for n in pred["numbers"])
        print(f"{i:2d}. [{numbers_str}]")

if __name__ == "__main__":
    main() 