#!/usr/bin/env python3
"""
Compare Multiple Prediction Runs
Creates a detailed comparison table of all prediction runs
"""

import json
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_and_compare_runs():
    """Load all prediction files and create comparison table."""
    prediction_files = glob.glob("outputs/predictions/mathematical_predictions_*.json")
    
    if not prediction_files:
        print("❌ No prediction files found!")
        return
    
    runs_data = []
    all_predictions_by_run = []
    
    print("🔍 LOADING PREDICTION RUNS...")
    print("=" * 60)
    
    for i, file_path in enumerate(sorted(prediction_files), 1):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
                timestamp = data.get('timestamp', '')
                
                # Calculate statistics for this run
                sums = [sum(pred) for pred in predictions]
                all_numbers = [num for pred in predictions for num in pred]
                
                # Count odd/even
                odds = sum(1 for num in all_numbers if num % 2 == 1)
                
                run_stats = {
                    'Run': i,
                    'File': Path(file_path).name,
                    'Timestamp': timestamp[:19] if timestamp else '',
                    'Predictions': len(predictions),
                    'Min_Sum': min(sums) if sums else 0,
                    'Max_Sum': max(sums) if sums else 0,
                    'Avg_Sum': sum(sums)/len(sums) if sums else 0,
                    'Odd_Numbers': odds,
                    'Even_Numbers': len(all_numbers) - odds,
                    'Odd_Ratio': odds / len(all_numbers) if all_numbers else 0,
                    'Most_Frequent': None,
                    'Least_Frequent': None
                }
                
                # Find most/least frequent numbers in this run
                from collections import Counter
                freq_counter = Counter(all_numbers)
                if freq_counter:
                    most_common = freq_counter.most_common(3)
                    least_common = freq_counter.most_common()[-3:]
                    run_stats['Most_Frequent'] = [num for num, count in most_common]
                    run_stats['Least_Frequent'] = [num for num, count in least_common]
                
                runs_data.append(run_stats)
                all_predictions_by_run.append(predictions)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Create DataFrame for easy comparison
    df = pd.DataFrame(runs_data)
    
    # Display comparison table
    print(f"📊 COMPARISON OF {len(runs_data)} PREDICTION RUNS")
    print("=" * 60)
    print()
    
    # Basic stats table
    print("📈 BASIC STATISTICS:")
    print("-" * 50)
    stats_table = df[['Run', 'Predictions', 'Min_Sum', 'Max_Sum', 'Avg_Sum', 'Odd_Ratio']].copy()
    stats_table['Avg_Sum'] = stats_table['Avg_Sum'].round(1)
    stats_table['Odd_Ratio'] = (stats_table['Odd_Ratio'] * 100).round(1)
    stats_table.columns = ['Run', 'Preds', 'Min Sum', 'Max Sum', 'Avg Sum', 'Odd %']
    
    print(stats_table.to_string(index=False))
    
    print(f"\n🎯 DETAILED PREDICTIONS BY RUN:")
    print("-" * 50)
    
    for i, (run_data, predictions) in enumerate(zip(runs_data, all_predictions_by_run), 1):
        print(f"\n📍 RUN {i} ({run_data['Timestamp']}):")
        for j, pred in enumerate(predictions, 1):
            pred_str = ' - '.join(f'{num:02d}' for num in sorted(pred))
            sum_val = sum(pred)
            print(f"  {j:2d}. [{pred_str}] (Sum: {sum_val:3d})")
    
    # Cross-run frequency analysis
    print(f"\n🔢 CROSS-RUN FREQUENCY ANALYSIS:")
    print("-" * 50)
    
    all_numbers_total = [num for predictions in all_predictions_by_run for pred in predictions for num in pred]
    from collections import Counter
    total_freq = Counter(all_numbers_total)
    
    print("Top 15 Most Frequent Numbers (across all runs):")
    most_frequent_overall = total_freq.most_common(15)
    for i, (num, count) in enumerate(most_frequent_overall, 1):
        percentage = (count / len(all_numbers_total)) * 100
        print(f"  {i:2d}. Number {num:2d}: {count:2d} times ({percentage:4.1f}%)")
    
    print(f"\nNumbers Never Appeared:")
    all_appeared = set(all_numbers_total)
    never_appeared = [num for num in range(1, 60) if num not in all_appeared]
    if never_appeared:
        print(f"  {never_appeared}")
    else:
        print("  All numbers 1-59 appeared at least once!")
    
    # Unique vs duplicate analysis across runs
    print(f"\n🔄 UNIQUENESS ACROSS ALL RUNS:")
    print("-" * 50)
    
    all_predictions_tuples = [tuple(sorted(pred)) for predictions in all_predictions_by_run for pred in predictions]
    unique_predictions = list(set(all_predictions_tuples))
    duplicates = len(all_predictions_tuples) - len(unique_predictions)
    
    print(f"Total predictions across all runs: {len(all_predictions_tuples)}")
    print(f"Unique predictions: {len(unique_predictions)}")
    print(f"Duplicate predictions: {duplicates}")
    print(f"Uniqueness ratio: {len(unique_predictions)/len(all_predictions_tuples)*100:.1f}%")
    
    # Find if any predictions appear in multiple runs
    if duplicates > 0:
        print(f"\n🔍 DUPLICATE PREDICTIONS FOUND:")
        from collections import Counter
        pred_counter = Counter(all_predictions_tuples)
        duplicated_preds = [(pred, count) for pred, count in pred_counter.items() if count > 1]
        
        for pred, count in duplicated_preds:
            pred_str = ' - '.join(f'{num:02d}' for num in pred)
            print(f"  [{pred_str}] appeared {count} times")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/analysis/runs_comparison_{timestamp}.json"
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        'timestamp': timestamp,
        'total_runs': len(runs_data),
        'runs_data': runs_data,
        'frequency_analysis': dict(total_freq.most_common()),
        'uniqueness_stats': {
            'total_predictions': len(all_predictions_tuples),
            'unique_predictions': len(unique_predictions),
            'duplicates': duplicates,
            'uniqueness_ratio': len(unique_predictions)/len(all_predictions_tuples)
        },
        'never_appeared_numbers': never_appeared
    }
    
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n📁 Comparison results saved to: {output_file}")
    
    return comparison_results

def main():
    """Main function."""
    try:
        print("🎯 PREDICTION RUNS COMPARISON ANALYSIS")
        print("=" * 60)
        print()
        
        results = load_and_compare_runs()
        
        if results:
            print(f"\n✅ Analysis complete!")
            print(f"   • Analyzed {results['total_runs']} prediction runs")
            print(f"   • Total predictions: {results['uniqueness_stats']['total_predictions']}")
            print(f"   • Uniqueness: {results['uniqueness_stats']['uniqueness_ratio']*100:.1f}%")
            print(f"   • Numbers never appeared: {len(results['never_appeared_numbers'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 