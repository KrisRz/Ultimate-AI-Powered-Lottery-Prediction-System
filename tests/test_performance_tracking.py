import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import os
import sys
from unittest.mock import patch, MagicMock, mock_open

# Add project root to path to resolve imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

try:
    from scripts.performance_tracking import (
        calculate_metrics,
        calculate_top_k_accuracy,
        track_model_performance,
        log_performance_trends,
        get_model_weights,
        normalize_weights,
        get_default_weights,
        get_model_performance_summary,
        calculate_ensemble_weights,
        calculate_diversity_bonus
    )
except ImportError:
    try:
        # Try direct imports without 'scripts.' prefix
        from performance_tracking import (
            calculate_metrics,
            calculate_top_k_accuracy,
            track_model_performance,
            log_performance_trends,
            get_model_weights,
            normalize_weights,
            get_default_weights,
            get_model_performance_summary,
            calculate_ensemble_weights,
            calculate_diversity_bonus
        )
    except ImportError:
        pytest.skip("Could not import performance_tracking module. Skipping tests.", allow_module_level=True)

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_predictions():
    """Fixture to create sample predictions."""
    return np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]
    ])

@pytest.fixture
def sample_actual():
    """Fixture to create sample actual draws."""
    return np.array([
        [1, 2, 3, 4, 5, 7],      # 5 matches with first prediction
        [7, 8, 9, 10, 11, 13],   # 5 matches with second prediction
        [13, 14, 15, 16, 18, 20] # 4 matches with third prediction
    ])

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        'Draw Date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        'Main_Numbers': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
        'Bonus': [7, 13, 19]
    }
    return pd.DataFrame(data)

@pytest.fixture
def performance_log_data():
    """Fixture to create sample performance log data."""
    return [
        {
            'model': 'lstm',
            'timestamp': '2023-01-01T00:00:00',
            'prediction_count': 10,
            'metrics': {'accuracy': 0.5, 'match_rate': 3.0, 'rmse': 2.0, 'perfect_match_rate': 0.0}
        },
        {
            'model': 'xgboost',
            'timestamp': '2023-01-02T00:00:00',
            'prediction_count': 10,
            'metrics': {'accuracy': 0.6, 'match_rate': 3.5, 'rmse': 1.8, 'perfect_match_rate': 0.0}
        },
        {
            'model': 'lstm',
            'timestamp': '2023-01-03T00:00:00',
            'prediction_count': 10,
            'metrics': {'accuracy': 0.45, 'match_rate': 2.8, 'rmse': 2.2, 'perfect_match_rate': 0.0}
        },
        {
            'model': 'catboost',
            'timestamp': '2023-01-04T00:00:00',
            'prediction_count': 10,
            'metrics': {'accuracy': 0.55, 'match_rate': 3.2, 'rmse': 1.9, 'perfect_match_rate': 0.0}
        }
    ]

@pytest.fixture
def performance_log_file(tmp_path, performance_log_data):
    """Fixture to create a temporary performance log file."""
    results_dir = tmp_path / 'results'
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / 'performance_log.json'
    
    with open(log_path, 'w') as f:
        json.dump(performance_log_data, f)
    
    return log_path

def test_calculate_metrics(sample_predictions, sample_actual):
    """Test calculating performance metrics."""
    metrics = calculate_metrics(sample_predictions, sample_actual)
    
    # Check keys
    expected_keys = ['accuracy', 'match_rate', 'perfect_match_rate', 'rmse', 'mae', 'top_3_accuracy']
    for key in expected_keys:
        assert key in metrics
    
    # Check values
    assert metrics['accuracy'] == 14/18  # 5+5+4 matches out of 18 total numbers
    assert metrics['match_rate'] == (5+5+4)/3  # Average matches per prediction
    assert metrics['perfect_match_rate'] == 0.0  # No exact matches
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    assert 0 <= metrics['top_3_accuracy'] <= 1

def test_calculate_metrics_invalid_shapes():
    """Test calculate_metrics with invalid input shapes."""
    # Test with inconsistent shapes
    predictions = np.array([[1, 2, 3], [4, 5, 6]])
    actual = np.array([[1, 2], [3, 4]])
    
    metrics = calculate_metrics(predictions, actual)
    
    # Should return default metrics
    assert metrics['accuracy'] == 0.0
    assert metrics['match_rate'] == 0.0
    assert metrics['perfect_match_rate'] == 0.0
    assert metrics['rmse'] == float('inf')
    assert metrics['mae'] == float('inf')

def test_calculate_metrics_empty_input():
    """Test calculate_metrics with empty input."""
    predictions = np.array([])
    actual = np.array([])
    
    metrics = calculate_metrics(predictions, actual)
    
    # Should return default metrics
    assert metrics['accuracy'] == 0.0
    assert metrics['match_rate'] == 0.0
    assert metrics['perfect_match_rate'] == 0.0
    assert metrics['rmse'] == float('inf')
    assert metrics['mae'] == float('inf')

def test_calculate_top_k_accuracy(sample_predictions, sample_actual):
    """Test calculating top-k accuracy."""
    # Test with k=3
    accuracy_k3 = calculate_top_k_accuracy(sample_predictions, sample_actual, k=3)
    assert 0 <= accuracy_k3 <= 1
    
    # Test with k=1
    accuracy_k1 = calculate_top_k_accuracy(sample_predictions, sample_actual, k=1)
    assert 0 <= accuracy_k1 <= 1
    
    # k=3 should give higher or equal accuracy than k=1
    assert accuracy_k3 >= accuracy_k1

def test_calculate_top_k_accuracy_invalid_input():
    """Test calculate_top_k_accuracy with invalid input."""
    # Test with inconsistent shapes
    predictions = np.array([[1, 2, 3]])
    actual = np.array([[1, 2]])
    
    accuracy = calculate_top_k_accuracy(predictions, actual, k=1)
    assert accuracy == 0.0

@patch('scripts.performance_tracking.Path.exists')
@patch('scripts.performance_tracking.calculate_metrics')
def test_track_model_performance(mock_calculate, mock_exists, sample_predictions, sample_actual, tmp_path):
    """Test tracking model performance."""
    # Setup
    mock_calculate.return_value = {
        'accuracy': 0.7,
        'match_rate': 4.0,
        'perfect_match_rate': 0.1,
        'rmse': 1.5,
        'mae': 1.0,
        'top_3_accuracy': 0.8
    }
    
    results_dir = tmp_path / 'results'
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / 'performance_log.json'
    
    mock_exists.return_value = False  # No existing log file
    
    # Test with new log file
    with patch('builtins.open', mock_open()) as m:
        result = track_model_performance(
            model_name='lstm',
            predictions=sample_predictions,
            actual=sample_actual,
            draw_date='2023-01-05',
            metadata={'test': True},
            performance_log_path=str(log_path)
        )
    
    # Check result
    assert result['model'] == 'lstm'
    assert result['timestamp'] == '2023-01-05'
    assert result['prediction_count'] == len(sample_predictions)
    assert result['metrics']['accuracy'] == 0.7
    assert result['metrics']['match_rate'] == 4.0
    assert result['metadata'] == {'test': True}
    
    # Test with existing log file
    mock_exists.return_value = True
    mock_log_data = [{'model': 'xgboost', 'timestamp': '2023-01-01T00:00:00'}]
    
    with patch('builtins.open', mock_open(read_data=json.dumps(mock_log_data))) as m:
        result = track_model_performance(
            model_name='lstm',
            predictions=sample_predictions,
            actual=sample_actual,
            draw_date='2023-01-05',
            performance_log_path=str(log_path)
        )
    
    # Verify that calculate_metrics was called
    mock_calculate.assert_called()

def test_log_performance_trends(performance_log_data):
    """Test logging performance trends."""
    # Test with declining performance
    with patch('scripts.performance_tracking.logger') as mock_logger:
        log_performance_trends('lstm', performance_log_data)
        mock_logger.warning.assert_called()  # Should warn about declining accuracy
    
    # Test with improving performance
    improved_data = performance_log_data.copy()
    improved_data.append({
        'model': 'lstm',
        'timestamp': '2023-01-05T00:00:00',
        'prediction_count': 10,
        'metrics': {'accuracy': 0.65, 'match_rate': 3.9, 'rmse': 1.5, 'perfect_match_rate': 0.1}
    })
    
    with patch('scripts.performance_tracking.logger') as mock_logger:
        log_performance_trends('lstm', improved_data)
        mock_logger.info.assert_called()  # Should info about improving accuracy

def test_get_model_weights(performance_log_file):
    """Test getting model weights based on performance history."""
    with patch('scripts.performance_tracking.PERFORMANCE_LOG_PATH', str(performance_log_file)):
        models = ['lstm', 'xgboost', 'catboost', 'unknown_model']
        weights = get_model_weights(models, history_length=10)
        
        # Check that all requested models have weights
        for model in models:
            assert model in weights
        
        # Check that weights are non-negative and sum to 1
        assert all(w >= 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # Check that better-performing models have higher weights
        assert weights['xgboost'] > weights['lstm']  # xgboost has better metrics
        assert weights['unknown_model'] == min(weights.values())  # Unknown model gets minimum weight

def test_get_model_weights_no_log_file(tmp_path):
    """Test get_model_weights when no log file exists."""
    non_existent_path = tmp_path / 'non_existent.json'
    
    with patch('scripts.performance_tracking.PERFORMANCE_LOG_PATH', str(non_existent_path)):
        with patch('scripts.performance_tracking.Path.exists', return_value=False):
            models = ['lstm', 'xgboost']
            weights = get_model_weights(models)
            
            # Should return equal weights
            assert weights['lstm'] == weights['xgboost']
            assert abs(sum(weights.values()) - 1.0) < 1e-6

def test_normalize_weights():
    """Test normalizing weights."""
    # Test with positive weights
    weights = {'lstm': 2.0, 'xgboost': 3.0, 'catboost': 5.0}
    normalized = normalize_weights(weights)
    
    assert abs(sum(normalized.values()) - 1.0) < 1e-6
    assert normalized['catboost'] > normalized['xgboost'] > normalized['lstm']
    
    # Test with minimum weight constraint
    weights = {'lstm': 0.1, 'xgboost': 0.8, 'catboost': 0.1}
    normalized = normalize_weights(weights, min_weight=0.2)
    
    assert abs(sum(normalized.values()) - 1.0) < 1e-6
    assert all(w >= 0.2 for w in normalized.values())  # All weights should meet minimum
    
    # Test with negative values
    weights = {'lstm': -1.0, 'xgboost': 2.0, 'catboost': -0.5}
    normalized = normalize_weights(weights)
    
    assert abs(sum(normalized.values()) - 1.0) < 1e-6
    assert normalized['lstm'] == normalized['catboost']  # Equal floor weights
    assert normalized['xgboost'] > normalized['lstm']

def test_get_default_weights():
    """Test getting default model weights."""
    weights = get_default_weights()
    
    # Check that all expected models have weights
    expected_models = [
        'lstm', 'holtwinters', 'linear', 'xgboost', 'lightgbm', 
        'knn', 'gradient_boosting', 'catboost', 'cnn_lstm', 
        'autoencoder', 'meta'
    ]
    for model in expected_models:
        assert model in weights
    
    # Weights should sum to 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    # Meta should have zero weight by default
    assert weights['meta'] == 0.0

def test_get_model_performance_summary(performance_log_file):
    """Test getting model performance summary."""
    with patch('scripts.performance_tracking.PERFORMANCE_LOG_PATH', str(performance_log_file)):
        summary = get_model_performance_summary(history_length=10)
        
        # Check that all models in the log have summaries
        assert 'lstm' in summary
        assert 'xgboost' in summary
        assert 'catboost' in summary
        
        # Check summary structure
        for model in ['lstm', 'xgboost', 'catboost']:
            assert 'entries' in summary[model]
            assert 'avg_metrics' in summary[model]
            assert 'accuracy' in summary[model]['avg_metrics']
            assert 'match_rate' in summary[model]['avg_metrics']
            assert 'rmse' in summary[model]['avg_metrics']
        
        # Check specific values
        assert summary['lstm']['entries'] == 2  # Two lstm entries in log
        assert summary['xgboost']['entries'] == 1
        assert summary['catboost']['entries'] == 1
        
        # Check that averages are correct
        assert summary['lstm']['avg_metrics']['accuracy'] == (0.5 + 0.45) / 2

def test_get_model_performance_summary_no_file(tmp_path):
    """Test get_model_performance_summary when no log file exists."""
    non_existent_path = tmp_path / 'non_existent.json'
    
    with patch('scripts.performance_tracking.PERFORMANCE_LOG_PATH', str(non_existent_path)):
        with patch('scripts.performance_tracking.Path.exists', return_value=False):
            summary = get_model_performance_summary()
            
            assert 'error' in summary
            assert summary['error'] == 'No performance log found'

@patch('scripts.performance_tracking.get_model_weights')
@patch('scripts.performance_tracking.calculate_diversity_bonus')
def test_calculate_ensemble_weights(mock_diversity, mock_weights, sample_data):
    """Test calculating ensemble weights with diversity bonus."""
    mock_weights.return_value = {'lstm': 0.4, 'xgboost': 0.3, 'catboost': 0.3}
    mock_diversity.return_value = {'lstm': 0.2, 'xgboost': 0.1, 'catboost': 0.1}
    
    models = ['lstm', 'xgboost', 'catboost']
    weights = calculate_ensemble_weights(models, sample_data)
    
    # Check that all models have weights
    for model in models:
        assert model in weights
    
    # Weights should sum to 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    # LSTM should have higher weight due to diversity bonus
    assert weights['lstm'] > weights['xgboost']
    assert weights['lstm'] > weights['catboost']

def test_calculate_diversity_bonus():
    """Test calculating diversity bonus for different model types."""
    models = ['lstm', 'holtwinters', 'xgboost', 'catboost', 'linear']
    bonuses = calculate_diversity_bonus(models)
    
    # Check that all models have bonuses
    for model in models:
        assert model in bonuses
    
    # Statistical models should have higher bonus than ML models
    assert bonuses['holtwinters'] > bonuses['xgboost']
    assert bonuses['linear'] > bonuses['catboost']
    
    # Deep learning should have higher bonus than ML models
    assert bonuses['lstm'] > bonuses['xgboost']
    
    # Check with a smaller set containing only similar models
    similar_models = ['xgboost', 'catboost', 'lightgbm']
    similar_bonuses = calculate_diversity_bonus(similar_models)
    
    # Bonuses should be smaller for similar models
    assert all(b < 0.1 for b in similar_bonuses.values())

@patch('scripts.performance_tracking.get_model_performance_summary')
def test_get_model_weights_with_no_performance(mock_summary):
    """Test get_model_weights when no performance data is available."""
    mock_summary.return_value = {'error': 'No performance log found'}
    
    models = ['lstm', 'xgboost', 'catboost']
    weights = get_model_weights(models)
    
    # Should return equal weights
    assert weights['lstm'] == weights['xgboost'] == weights['catboost']
    assert abs(sum(weights.values()) - 1.0) < 1e-6

if __name__ == "__main__":
    pytest.main([__file__]) 