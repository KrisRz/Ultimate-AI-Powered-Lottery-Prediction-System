"""
Benchmark tests for lottery prediction system.
This script tests performance and accuracy of prediction models and utilities.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import logging
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

# Import all necessary modules
from scripts.predictions import (
    generate_next_draw_predictions,
    predict_with_model,
    calculate_prediction_metrics,
    validate_predictions,
    score_combinations
)
from scripts.new_predict import (
    prepare_input_data,
    predict_with_ensemble,
    generate_predictions
)
from scripts.train_models import (
    train_lstm_model,
    train_cnn_lstm_model,
    evaluate_model
)
from scripts.model_bridge import load_models, create_ensemble
from scripts.performance_tracking import track_prediction_performance
from scripts.analyze_data import analyze_prediction_patterns

# Constants
MIN_NUMBER = 1
MAX_NUMBER = 59
N_NUMBERS = 6
BENCHMARK_ITERATIONS = 100
TEST_DATA_SIZE = 1000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkMetrics:
    """Class to track benchmark metrics."""
    def __init__(self):
        self.prediction_times = []
        self.accuracy_scores = []
        self.memory_usage = []
        self.model_performances = {}
    
    def add_prediction_time(self, time_ms: float):
        self.prediction_times.append(time_ms)
    
    def add_accuracy_score(self, score: float):
        self.accuracy_scores.append(score)
    
    def add_memory_usage(self, usage_mb: float):
        self.memory_usage.append(usage_mb)
    
    def add_model_performance(self, model_name: str, metrics: Dict):
        if model_name not in self.model_performances:
            self.model_performances[model_name] = []
        self.model_performances[model_name].append(metrics)
    
    def get_summary(self) -> Dict:
        return {
            "prediction_time": {
                "mean": np.mean(self.prediction_times),
                "std": np.std(self.prediction_times),
                "min": min(self.prediction_times),
                "max": max(self.prediction_times)
            },
            "accuracy": {
                "mean": np.mean(self.accuracy_scores),
                "std": np.std(self.accuracy_scores)
            },
            "memory_usage": {
                "mean": np.mean(self.memory_usage),
                "peak": max(self.memory_usage)
            },
            "model_performances": {
                model: {
                    "mean_accuracy": np.mean([m["accuracy"] for m in metrics]),
                    "mean_loss": np.mean([m["loss"] for m in metrics])
                }
                for model, metrics in self.model_performances.items()
            }
        }

@pytest.fixture
def benchmark_metrics():
    """Fixture to provide benchmark metrics instance."""
    return BenchmarkMetrics()

@pytest.fixture
def sample_data():
    """Generate sample lottery data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=TEST_DATA_SIZE)
    numbers = [
        sorted(np.random.choice(range(MIN_NUMBER, MAX_NUMBER + 1), N_NUMBERS, replace=False))
        for _ in range(TEST_DATA_SIZE)
    ]
    
    data = {
        'Draw Date': dates,
        'Main_Numbers': numbers,
        'Day_of_Week': [d.dayofweek for d in dates]
    }
    
    # Add derived features
    df = pd.DataFrame(data)
    df['Sum'] = df['Main_Numbers'].apply(sum)
    df['Mean'] = df['Main_Numbers'].apply(np.mean)
    df['Std'] = df['Main_Numbers'].apply(np.std)
    
    return df

@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    lstm_mock = Mock()
    lstm_mock.predict.return_value = np.random.rand(1, N_NUMBERS)
    
    cnn_lstm_mock = Mock()
    cnn_lstm_mock.predict.return_value = np.random.rand(1, N_NUMBERS)
    
    return {
        'lstm': lstm_mock,
        'cnn_lstm': cnn_lstm_mock
    }

def test_prediction_speed(mock_models, sample_data, benchmark_metrics):
    """Benchmark prediction generation speed."""
    logger.info("Testing prediction speed...")
    
    for _ in range(BENCHMARK_ITERATIONS):
        start_time = time.time()
        
        predictions = generate_next_draw_predictions(
            mock_models,
            sample_data.tail(50),
            n_predictions=10
        )
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        benchmark_metrics.add_prediction_time(duration_ms)
        
        # Validate predictions
        assert len(predictions) == 10
        for pred in predictions:
            assert len(pred) == N_NUMBERS
            assert all(MIN_NUMBER <= x <= MAX_NUMBER for x in pred)

def test_model_accuracy(mock_models, sample_data, benchmark_metrics):
    """Benchmark model prediction accuracy."""
    logger.info("Testing model accuracy...")
    
    # Split data into train/test
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    test_data = sample_data[train_size:]
    
    for model_name, model in mock_models.items():
        predictions = []
        actuals = test_data['Main_Numbers'].tolist()
        
        for _ in range(BENCHMARK_ITERATIONS):
            pred = predict_with_model(model_name, model, test_data)
            predictions.append(pred)
            
            # Calculate accuracy
            accuracy = calculate_prediction_accuracy(pred, actuals[-1])
            benchmark_metrics.add_accuracy_score(accuracy)
            
            metrics = {
                "accuracy": accuracy,
                "loss": np.random.random()  # Mock loss for demonstration
            }
            benchmark_metrics.add_model_performance(model_name, metrics)

def test_ensemble_performance(mock_models, sample_data, benchmark_metrics):
    """Benchmark ensemble model performance."""
    logger.info("Testing ensemble performance...")
    
    weights = {'lstm': 0.6, 'cnn_lstm': 0.4}
    
    for _ in range(BENCHMARK_ITERATIONS):
        start_time = time.time()
        
        predictions = generate_predictions(
            {'models': mock_models, 'weights': weights},
            sample_data,
            count=10
        )
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        benchmark_metrics.add_prediction_time(duration_ms)
        
        # Score predictions
        scores = score_combinations(predictions, sample_data)
        avg_score = np.mean([score for _, score in scores])
        benchmark_metrics.add_accuracy_score(avg_score)

def test_memory_usage(mock_models, sample_data, benchmark_metrics):
    """Benchmark memory usage during predictions."""
    logger.info("Testing memory usage...")
    
    import psutil
    process = psutil.Process()
    
    for _ in range(BENCHMARK_ITERATIONS):
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate predictions
        _ = generate_next_draw_predictions(
            mock_models,
            sample_data,
            n_predictions=20
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        benchmark_metrics.add_memory_usage(memory_used)

def test_data_processing_speed(sample_data, benchmark_metrics):
    """Benchmark data processing and feature extraction speed."""
    logger.info("Testing data processing speed...")
    
    for _ in range(BENCHMARK_ITERATIONS):
        start_time = time.time()
        
        # Prepare input data
        processed_data = prepare_input_data(sample_data)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        benchmark_metrics.add_prediction_time(duration_ms)
        
        assert isinstance(processed_data, np.ndarray)

def generate_benchmark_report(benchmark_metrics: BenchmarkMetrics) -> str:
    """Generate a detailed benchmark report."""
    summary = benchmark_metrics.get_summary()
    
    report = [
        "Lottery Prediction System Benchmark Report",
        "========================================"
        "",
        "Prediction Performance:",
        f"- Average prediction time: {summary['prediction_time']['mean']:.2f}ms",
        f"- Prediction time std: {summary['prediction_time']['std']:.2f}ms",
        f"- Min prediction time: {summary['prediction_time']['min']:.2f}ms",
        f"- Max prediction time: {summary['prediction_time']['max']:.2f}ms",
        "",
        "Accuracy Metrics:",
        f"- Mean accuracy: {summary['accuracy']['mean']:.4f}",
        f"- Accuracy std: {summary['accuracy']['std']:.4f}",
        "",
        "Memory Usage:",
        f"- Average memory usage: {summary['memory_usage']['mean']:.2f}MB",
        f"- Peak memory usage: {summary['memory_usage']['peak']:.2f}MB",
        "",
        "Model Performance:"
    ]
    
    for model, perf in summary['model_performances'].items():
        report.extend([
            f"  {model}:",
            f"  - Mean accuracy: {perf['mean_accuracy']:.4f}",
            f"  - Mean loss: {perf['mean_loss']:.4f}"
        ])
    
    return "\n".join(report)

def plot_benchmark_results(benchmark_metrics: BenchmarkMetrics, output_dir: str = "benchmark_results"):
    """Generate plots of benchmark results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot prediction times
    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_metrics.prediction_times)
    plt.title("Prediction Times Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.savefig(f"{output_dir}/prediction_times.png")
    plt.close()
    
    # Plot accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(benchmark_metrics.accuracy_scores, bins=20)
    plt.title("Accuracy Distribution")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/accuracy_distribution.png")
    plt.close()
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_metrics.memory_usage)
    plt.title("Memory Usage Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Memory (MB)")
    plt.savefig(f"{output_dir}/memory_usage.png")
    plt.close()

def main():
    """Run all benchmark tests and generate report."""
    pytest.main([__file__, "-v"])
    
    # Note: In practice, you would need to collect metrics during the pytest run
    # and pass them to these functions. This is just to show the structure.
    metrics = BenchmarkMetrics()  # This would be populated during tests
    
    # Generate report
    report = generate_benchmark_report(metrics)
    with open("benchmark_report.txt", "w") as f:
        f.write(report)
    
    # Generate plots
    plot_benchmark_results(metrics)

if __name__ == "__main__":
    main()
