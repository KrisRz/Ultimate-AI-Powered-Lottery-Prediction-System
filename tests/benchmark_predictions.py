"""Benchmark tests for prediction models."""
import pytest
import pandas as pd
import numpy as np
import psutil
import time
from scripts.predict_numbers import monte_carlo_simulation, predict_with_model, score_combinations
from scripts.analyze_data import get_prediction_weights
from models.prediction import (
    predict_with_lstm,
    predict_with_cnn_lstm,
    predict_with_autoencoder,
    validate_prediction_format
)

# Sample data fixture
@pytest.fixture
def sample_data():
    # Create sample lottery data with realistic variation
    data = {
        'Draw Date': pd.date_range(start='2020-01-01', periods=100),
        'Main_Numbers': [sorted(np.random.choice(range(1, 60), 6, replace=False)) 
                        for _ in range(100)],
        'DayOfWeek': [i % 7 for i in range(100)]
    }
    df = pd.DataFrame(data)
    
    # Add derived features
    df['Sum'] = df['Main_Numbers'].apply(sum)
    df['Mean'] = df['Main_Numbers'].apply(np.mean)
    df['Std'] = df['Main_Numbers'].apply(np.std)
    df['Unique'] = df['Main_Numbers'].apply(lambda x: len(set(x)))
    df['Primes'] = df['Main_Numbers'].apply(lambda x: sum(1 for n in x if is_prime(n)))
    df['Odds'] = df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2))
    
    return df 