#!/usr/bin/env python3
"""
Prediction Analysis Tool
Analyzes multiple prediction runs to identify patterns and assess accuracy potential
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import glob

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Setup for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_all_predictions(predictions_dir="outputs/predictions"):
    """Load all prediction files from the directory."""
    prediction_files = glob.glob(f"{predictions_dir}/mathematical_predictions_*.json")
    all_predictions = []
    all_metadata = []
    
    print(f"Found {len(prediction_files)} prediction files")
    
    for file_path in sorted(prediction_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
                metadata = data.get('metadata', {})
                timestamp = data.get('timestamp', '')
                
                all_predictions.extend(predictions)
                all_metadata.append({
                    'file': Path(file_path).name,
                    'timestamp': timestamp,
                    'count': len(predictions),
                    'metadata': metadata
                })
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_predictions, all_metadata

def analyze_number_frequencies(predictions):
    """Analyze frequency of individual numbers across all predictions."""
    all_numbers = [num for pred in predictions for num in pred]
    frequency_counter = Counter(all_numbers)
    
    # Create frequency analysis
    numbers = list(range(1, 60))
    frequencies = [frequency_counter.get(num, 0) for num in numbers]
    
    frequency_df = pd.DataFrame({
        'Number': numbers,
        'Frequency': frequencies,
        'Percentage': [f/len(predictions)*100 for f in frequencies]
    })
    
    return frequency_df.sort_values('Frequency', ascending=False)

def analyze_sum_patterns(predictions):
    """Analyze sum patterns in predictions."""
    sums = [sum(pred) for pred in predictions]
    
    return {
        'sums': sums,
        'mean': np.mean(sums),
        'std': np.std(sums),
        'min': min(sums),
        'max': max(sums),
        'median': np.median(sums),
        'q25': np.percentile(sums, 25),
        'q75': np.percentile(sums, 75)
    }

def analyze_distribution_patterns(predictions):
    """Analyze distribution patterns (odd/even, high/low, etc.)."""
    odd_even_ratios = []
    high_low_ratios = []
    consecutive_counts = []
    
    for pred in predictions:
        # Odd/Even analysis
        odds = sum(1 for num in pred if num % 2 == 1)
        odd_even_ratios.append(odds / 6)
        
        # High/Low analysis (high = 30-59, low = 1-29)
        highs = sum(1 for num in pred if num >= 30)
        high_low_ratios.append(highs / 6)
        
        # Consecutive numbers
        sorted_pred = sorted(pred)
        consecutive = 0
        for i in range(len(sorted_pred) - 1):
            if sorted_pred[i+1] - sorted_pred[i] == 1:
                consecutive += 1
        consecutive_counts.append(consecutive)
    
    return {
        'odd_ratios': odd_even_ratios,
        'high_ratios': high_low_ratios,
        'consecutive_counts': consecutive_counts,
        'avg_odd_ratio': np.mean(odd_even_ratios),
        'avg_high_ratio': np.mean(high_low_ratios),
        'avg_consecutive': np.mean(consecutive_counts)
    }

def analyze_uniqueness(predictions):
    """Analyze uniqueness of predictions."""
    unique_predictions = []
    duplicate_count = 0
    
    for pred in predictions:
        sorted_pred = tuple(sorted(pred))
        if sorted_pred not in unique_predictions:
            unique_predictions.append(sorted_pred)
        else:
            duplicate_count += 1
    
    return {
        'total_predictions': len(predictions),
        'unique_predictions': len(unique_predictions),
        'duplicate_count': duplicate_count,
        'uniqueness_ratio': len(unique_predictions) / len(predictions)
    }

def simulate_accuracy_test(predictions, num_simulations=1000):
    """Simulate accuracy by testing against random 'actual' draws."""
    match_results = defaultdict(int)
    
    for _ in range(num_simulations):
        # Generate a random "actual" draw
        actual_draw = sorted(np.random.choice(range(1, 60), 6, replace=False))
        
        # Test each prediction against this actual draw
        for pred in predictions:
            matches = len(set(pred) & set(actual_draw))
            match_results[matches] += 1
    
    # Calculate probabilities
    total_tests = len(predictions) * num_simulations
    match_probabilities = {
        matches: count / total_tests 
        for matches, count in match_results.items()
    }
    
    return match_probabilities

def create_visualizations(frequency_df, sum_stats, distribution_stats, output_dir="outputs/visualizations"):
    """Create comprehensive visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Number Frequency Plot
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    top_20 = frequency_df.head(20)
    plt.bar(top_20['Number'], top_20['Frequency'], color='skyblue')
    plt.title('Top 20 Most Frequent Numbers')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # 2. Sum Distribution
    plt.subplot(2, 3, 2)
    plt.hist(sum_stats['sums'], bins=20, alpha=0.7, color='lightgreen')
    plt.axvline(sum_stats['mean'], color='red', linestyle='--', label=f'Mean: {sum_stats["mean"]:.1f}')
    plt.title('Sum Distribution')
    plt.xlabel('Sum of Numbers')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. Odd/Even Ratio Distribution
    plt.subplot(2, 3, 3)
    plt.hist(distribution_stats['odd_ratios'], bins=10, alpha=0.7, color='orange')
    plt.axvline(distribution_stats['avg_odd_ratio'], color='red', linestyle='--', 
                label=f'Avg: {distribution_stats["avg_odd_ratio"]:.2f}')
    plt.title('Odd Numbers Ratio Distribution')
    plt.xlabel('Ratio of Odd Numbers')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 4. High/Low Ratio Distribution
    plt.subplot(2, 3, 4)
    plt.hist(distribution_stats['high_ratios'], bins=10, alpha=0.7, color='purple')
    plt.axvline(distribution_stats['avg_high_ratio'], color='red', linestyle='--',
                label=f'Avg: {distribution_stats["avg_high_ratio"]:.2f}')
    plt.title('High Numbers (30-59) Ratio Distribution')
    plt.xlabel('Ratio of High Numbers')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 5. Consecutive Numbers
    plt.subplot(2, 3, 5)
    consecutive_counts = Counter(distribution_stats['consecutive_counts'])
    plt.bar(list(consecutive_counts.keys()), list(consecutive_counts.values()), color='coral')
    plt.title('Consecutive Number Pairs')
    plt.xlabel('Number of Consecutive Pairs')
    plt.ylabel('Frequency')
    
    # 6. Number Range Distribution
    plt.subplot(2, 3, 6)
    ranges = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-59']
    range_counts = [
        len(frequency_df[(frequency_df['Number'] >= 1) & (frequency_df['Number'] <= 10)]),
        len(frequency_df[(frequency_df['Number'] >= 11) & (frequency_df['Number'] <= 20)]),
        len(frequency_df[(frequency_df['Number'] >= 21) & (frequency_df['Number'] <= 30)]),
        len(frequency_df[(frequency_df['Number'] >= 31) & (frequency_df['Number'] <= 40)]),
        len(frequency_df[(frequency_df['Number'] >= 41) & (frequency_df['Number'] <= 50)]),
        len(frequency_df[(frequency_df['Number'] >= 51) & (frequency_df['Number'] <= 59)])
    ]
    plt.bar(ranges, range_counts, color='lightblue')
    plt.title('Number Distribution by Range')
    plt.xlabel('Number Range')
    plt.ylabel('Count of Different Numbers')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/prediction_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive analysis saved to: {output_dir}/prediction_analysis_{timestamp}.png")
    plt.show()

def generate_report(predictions, metadata):
    """Generate a comprehensive analysis report."""
    print("🔍 COMPREHENSIVE PREDICTION ANALYSIS REPORT")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Predictions Analyzed: {len(predictions)}")
    print(f"Total Runs Analyzed: {len(metadata)}")
    print()
    
    # Basic Statistics
    frequency_df = analyze_number_frequencies(predictions)
    sum_stats = analyze_sum_patterns(predictions)
    distribution_stats = analyze_distribution_patterns(predictions)
    uniqueness_stats = analyze_uniqueness(predictions)
    accuracy_simulation = simulate_accuracy_test(predictions[:50])  # Test subset for speed
    
    # Report sections
    print("📊 FREQUENCY ANALYSIS:")
    print("-" * 30)
    print("Top 10 Most Frequent Numbers:")
    for i, row in frequency_df.head(10).iterrows():
        print(f"  {int(row['Number']):2d}: {int(row['Frequency']):3d} times ({row['Percentage']:5.1f}%)")
    
    print(f"\nLeast Frequent Numbers:")
    least_frequent = frequency_df.tail(5)
    for i, row in least_frequent.iterrows():
        print(f"  {int(row['Number']):2d}: {int(row['Frequency']):3d} times ({row['Percentage']:5.1f}%)")
    
    print(f"\n📈 SUM ANALYSIS:")
    print("-" * 30)
    print(f"  Mean Sum: {sum_stats['mean']:.1f}")
    print(f"  Std Dev: {sum_stats['std']:.1f}")
    print(f"  Range: {sum_stats['min']} - {sum_stats['max']}")
    print(f"  Median: {sum_stats['median']:.1f}")
    print(f"  Q1-Q3: {sum_stats['q25']:.1f} - {sum_stats['q75']:.1f}")
    
    print(f"\n🎯 DISTRIBUTION PATTERNS:")
    print("-" * 30)
    print(f"  Average Odd Numbers per Draw: {distribution_stats['avg_odd_ratio']*6:.1f}/6")
    print(f"  Average High Numbers (30-59): {distribution_stats['avg_high_ratio']*6:.1f}/6")
    print(f"  Average Consecutive Pairs: {distribution_stats['avg_consecutive']:.1f}")
    
    print(f"\n🔄 UNIQUENESS ANALYSIS:")
    print("-" * 30)
    print(f"  Total Predictions: {uniqueness_stats['total_predictions']}")
    print(f"  Unique Predictions: {uniqueness_stats['unique_predictions']}")
    print(f"  Duplicates: {uniqueness_stats['duplicate_count']}")
    print(f"  Uniqueness Ratio: {uniqueness_stats['uniqueness_ratio']:.1%}")
    
    print(f"\n🎲 SIMULATED ACCURACY TEST:")
    print("-" * 30)
    print("Expected match probabilities vs random draws:")
    for matches in sorted(accuracy_simulation.keys()):
        prob = accuracy_simulation[matches]
        print(f"  {matches} matches: {prob:.1%}")
    
    # Create visualizations
    create_visualizations(frequency_df, sum_stats, distribution_stats)
    
    print(f"\n💡 INSIGHTS:")
    print("-" * 30)
    most_frequent = frequency_df.iloc[0]
    least_frequent = frequency_df.iloc[-1]
    print(f"  • Number {most_frequent['Number']} appears most frequently ({most_frequent['Frequency']} times)")
    print(f"  • Number {least_frequent['Number']} appears least frequently ({least_frequent['Frequency']} times)")
    print(f"  • {uniqueness_stats['uniqueness_ratio']:.1%} of predictions are unique")
    print(f"  • Average sum is {sum_stats['mean']:.1f} (UK lottery typical range: 100-200)")
    
    if distribution_stats['avg_odd_ratio'] > 0.6:
        print(f"  • Predictions tend to favor odd numbers")
    elif distribution_stats['avg_odd_ratio'] < 0.4:
        print(f"  • Predictions tend to favor even numbers")
    else:
        print(f"  • Good balance between odd and even numbers")
    
    print(f"\n🎯 CALIBRATION RECOMMENDATIONS:")
    print("-" * 30)
    if uniqueness_stats['uniqueness_ratio'] < 0.9:
        print("  • Increase diversity algorithms to reduce duplicates")
    if sum_stats['std'] < 30:
        print("  • Increase sum range variation for better coverage")
    if distribution_stats['avg_consecutive'] > 2:
        print("  • Reduce consecutive number generation")
    
    print("\n✅ Analysis Complete!")
    return {
        'frequency_df': frequency_df,
        'sum_stats': sum_stats,
        'distribution_stats': distribution_stats,
        'uniqueness_stats': uniqueness_stats,
        'accuracy_simulation': accuracy_simulation
    }

def main():
    """Main analysis function."""
    try:
        # Load all predictions
        print("Loading prediction data...")
        predictions, metadata = load_all_predictions()
        
        if not predictions:
            print("❌ No prediction files found!")
            return 1
        
        # Generate comprehensive report
        analysis_results = generate_report(predictions, metadata)
        
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"outputs/analysis/prediction_analysis_{timestamp}.json"
        Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to regular Python types for JSON serialization
        serializable_results = {
            'timestamp': timestamp,
            'total_predictions': len(predictions),
            'frequency_analysis': analysis_results['frequency_df'].to_dict('records'),
                         'sum_statistics': {k: float(v) if hasattr(v, 'dtype') and 'float' in str(v.dtype) else v 
                              for k, v in analysis_results['sum_stats'].items()},
            'uniqueness_stats': analysis_results['uniqueness_stats'],
            'accuracy_simulation': analysis_results['accuracy_simulation']
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📁 Analysis results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 