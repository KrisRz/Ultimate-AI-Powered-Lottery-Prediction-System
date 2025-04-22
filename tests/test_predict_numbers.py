import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path to resolve imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

try:
    from scripts.predict_numbers import (
        ensure_valid_prediction,
        import_prediction_function,
        predict_with_model,
        score_combinations,
        ensemble_prediction,
        get_model_weights,
        predict_next_draw,
        save_predictions,
        calculate_prediction_metrics,
        monte_carlo_simulation,
        backtest
    )
except ImportError:
    try:
        # Try direct imports without 'scripts.' prefix
        from predict_numbers import (
            ensure_valid_prediction,
            import_prediction_function,
            predict_with_model,
            score_combinations,
            ensemble_prediction,
            get_model_weights,
            predict_next_draw,
            save_predictions,
            calculate_prediction_metrics,
            monte_carlo_simulation,
            backtest
        )
    except ImportError:
        pytest.skip("Could not import predict_numbers module. Skipping tests.", allow_module_level=True)

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        'Draw Date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'Main_Numbers': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        'Bonus': [7, 13]
    }
    return pd.DataFrame(data)

@pytest.fixture
def enhanced_sample_data(sample_data):
    """Fixture to create enhanced sample data with additional features."""
    df = sample_data.copy()
    df['Year'] = df['Draw Date'].dt.year
    df['Month'] = df['Draw Date'].dt.month
    df['DayOfWeek'] = df['Draw Date'].dt.dayofweek
    
    # Add some numerical features for model training
    df['sum'] = df['Main_Numbers'].apply(sum)
    df['mean'] = df['Main_Numbers'].apply(np.mean)
    df['std'] = df['Main_Numbers'].apply(np.std)
    df['min'] = df['Main_Numbers'].apply(min)
    df['max'] = df['Main_Numbers'].apply(max)
    
    return df

@pytest.fixture
def mock_models():
    """Fixture to create mock models."""
    return {
        'lstm': MagicMock(),
        'xgboost': MagicMock(),
        'catboost': MagicMock()
    }

@pytest.fixture
def sample_weights():
    """Fixture to provide sample prediction weights."""
    return {
        'number_weights': {i: 1/59 for i in range(1, 60)},
        'pair_weights': {(1, 2): 0.1, (3, 4): 0.05},
        'pattern_weights': {
            'consecutive_pairs': 0.1,
            'even_odd_ratio': {i: 1/7 for i in range(7)},
            'sum_ranges': {'mean': 150, 'std': 20},
            'hot_numbers': {1: 2.0, 2: 2.0},
            'cold_numbers': {58: 0.5, 59: 0.5}
        }
    }

def test_ensure_valid_prediction():
    """Test ensuring valid predictions."""
    # Valid prediction
    valid_pred = [1, 2, 3, 4, 5, 6]
    assert ensure_valid_prediction(valid_pred) == [1, 2, 3, 4, 5, 6]
    
    # Invalid predictions
    invalid_preds = [
        [1, 2, 3, 4, 5],  # Too short
        [1, 2, 3, 4, 5, 60],  # Out of range
        [1, 1, 2, 3, 4, 5],  # Duplicates
        [],  # Empty list
        None,  # None
        [1, 2, 3, 4, 5, "six"]  # Non-integer
    ]
    
    for pred in invalid_preds:
        result = ensure_valid_prediction(pred)
        assert len(result) == 6
        assert all(1 <= n <= 59 for n in result)
        assert len(set(result)) == 6
        assert result == sorted(result)  # Should be sorted

@patch('scripts.predict_numbers.importlib.import_module')
def test_import_prediction_function(mock_import):
    """Test importing prediction functions."""
    # Setup mock module
    mock_module = MagicMock()
    mock_module.predict_lstm_model = MagicMock(return_value=[1, 2, 3, 4, 5, 6])
    mock_import.return_value = mock_module
    
    # Test valid model
    func = import_prediction_function('lstm')
    assert func is not None
    assert func == mock_module.predict_lstm_model
    
    # Test unknown model
    mock_import.side_effect = ImportError("Module not found")
    assert import_prediction_function('unknown') is None

def test_predict_with_model():
    """Test prediction with a model."""
    # Create a mock prediction function
    mock_predict_func = MagicMock(return_value=[1, 2, 3, 4, 5, 6])
    
    # Test with valid prediction
    with patch('scripts.predict_numbers.import_prediction_function', return_value=mock_predict_func):
        with patch('scripts.predict_numbers.validate_prediction', return_value=(True, "")):
            model = MagicMock()
            data = pd.DataFrame({'Main_Numbers': [[1, 2, 3, 4, 5, 6]]})
            prediction = predict_with_model('lstm', model, data)
            
            assert prediction == [1, 2, 3, 4, 5, 6]
            assert mock_predict_func.called
    
    # Test with invalid prediction
    with patch('scripts.predict_numbers.import_prediction_function', return_value=mock_predict_func):
        with patch('scripts.predict_numbers.validate_prediction', return_value=(False, "Invalid prediction")):
            with patch('scripts.predict_numbers.ensure_valid_prediction', return_value=[10, 20, 30, 40, 50, 59]):
                model = MagicMock()
                data = pd.DataFrame({'Main_Numbers': [[1, 2, 3, 4, 5, 6]]})
                prediction = predict_with_model('lstm', model, data)
                
                assert prediction == [10, 20, 30, 40, 50, 59]
                assert mock_predict_func.called

    # Test with import error
    with patch('scripts.predict_numbers.import_prediction_function', return_value=None):
        with patch('scripts.predict_numbers.validate_prediction', return_value=(True, "")):
            with patch('scripts.predict_numbers.ensure_valid_prediction', return_value=[10, 20, 30, 40, 50, 59]):
                model = MagicMock()
                data = pd.DataFrame({'Main_Numbers': [[1, 2, 3, 4, 5, 6]]})
                prediction = predict_with_model('unknown', model, data)
                
                assert prediction == [10, 20, 30, 40, 50, 59]

def test_score_combinations(sample_data, sample_weights):
    """Test scoring combinations."""
    # Create combinations to score
    combinations = [
        [1, 2, 3, 4, 5, 6],  # Has high weights due to sample_weights
        [7, 8, 9, 10, 11, 12],  # Standard weights
        [54, 55, 56, 57, 58, 59]  # Contains cold numbers
    ]
    
    # Setup prediction weights mock
    with patch('scripts.predict_numbers.get_prediction_weights', return_value=sample_weights):
        scored = score_combinations(combinations, sample_data)
        
        # Verify results
        assert len(scored) == 3
        assert isinstance(scored[0], tuple)
        assert len(scored[0][0]) == 6  # Prediction
        assert isinstance(scored[0][1], float)  # Score
        
        # Check ordering by score (descending)
        assert scored[0][1] >= scored[1][1]
        assert scored[1][1] >= scored[2][1]
        
        # First combination should score highest due to hot numbers
        assert scored[0][0] == [1, 2, 3, 4, 5, 6]

@patch('scripts.predict_numbers.get_prediction_weights')
def test_ensemble_prediction(mock_get_weights, sample_data, sample_weights):
    """Test ensemble prediction."""
    mock_get_weights.return_value = sample_weights
    
    # Create individual predictions
    individual_predictions = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]
    ]
    
    # Set model weights
    model_weights = {'lstm': 0.5, 'xgboost': 0.3, 'catboost': 0.2}
    
    # Test ensemble predictions
    predictions = ensemble_prediction(individual_predictions, sample_data, model_weights, prediction_count=2)
    
    # Verify results
    assert len(predictions) == 2
    assert all(len(pred) == 6 for pred in predictions)
    assert all(len(set(pred)) == 6 for pred in predictions)
    assert all(all(1 <= n <= 59 for n in pred) for pred in predictions)
    
    # Test with empty inputs
    with pytest.raises(ValueError):
        ensemble_prediction([], sample_data, model_weights)
    
    # Test with no model weights
    predictions = ensemble_prediction(individual_predictions, sample_data, {}, prediction_count=1)
    assert len(predictions) == 1

@patch('scripts.predict_numbers.Path.exists')
@patch('builtins.open')
def test_get_model_weights(mock_open, mock_exists):
    """Test getting model weights."""
    # Setup mock files
    mock_exists.return_value = True
    mock_performance = [
        {
            'model': 'lstm',
            'metrics': {'accuracy': 0.1, 'rmse': 2.0}
        },
        {
            'model': 'xgboost',
            'metrics': {'accuracy': 0.2, 'rmse': 1.0}
        }
    ]
    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_performance)
    
    # Test getting weights
    weights = get_model_weights({'lstm': None, 'xgboost': None, 'catboost': None})
    
    # Verify weights
    assert 'lstm' in weights
    assert 'xgboost' in weights
    assert 'catboost' in weights  # Should get default weight
    assert abs(sum(weights.values()) - 1.0) < 1e-6  # Weights should sum to 1
    assert weights['xgboost'] > weights['lstm']  # xgboost has better metrics
    
    # Test with no performance log
    mock_exists.return_value = False
    weights = get_model_weights({'lstm': None})
    assert 'lstm' in weights
    assert weights['lstm'] == 1.0

@patch('scripts.predict_numbers.predict_with_model')
@patch('scripts.predict_numbers.get_model_weights')
@patch('scripts.predict_numbers.ensemble_prediction')
@patch('scripts.predict_numbers.save_predictions')
def test_predict_next_draw(mock_save, mock_ensemble, mock_weights, mock_predict, sample_data, mock_models):
    """Test predicting next draw."""
    # Setup mocks
    mock_predict.side_effect = [
        [1, 2, 3, 4, 5, 6],  # lstm
        [7, 8, 9, 10, 11, 12],  # xgboost
        [13, 14, 15, 16, 17, 18]  # catboost
    ]
    mock_weights.return_value = {'lstm': 0.5, 'xgboost': 0.3, 'catboost': 0.2}
    mock_ensemble.return_value = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    mock_save.return_value = True
    
    # Test prediction
    predictions = predict_next_draw(mock_models, sample_data, n_predictions=2)
    
    # Verify results
    assert len(predictions) == 2
    assert all(len(pred) == 6 for pred in predictions)
    assert mock_predict.call_count == 3
    assert mock_weights.called
    assert mock_ensemble.called
    assert mock_save.called

    # Test with save=False
    mock_save.reset_mock()
    predictions = predict_next_draw(mock_models, sample_data, n_predictions=1, save=False)
    assert len(predictions) == 1
    assert not mock_save.called

def test_save_predictions(tmp_path):
    """Test saving predictions to file."""
    # Create test data
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    metrics = {'accuracy': 0.5, 'rmse': 2.0}
    output_path = tmp_path / 'predictions.json'
    
    # Test saving
    success = save_predictions(predictions, metrics, str(output_path))
    
    # Verify results
    assert success
    assert output_path.exists()
    
    # Check file content
    with open(output_path, 'r') as f:
        saved = json.load(f)
    
    assert 'timestamp' in saved
    assert saved['count'] == 2
    assert saved['predictions'] == predictions
    assert saved['metrics'] == metrics
    
    # Test with invalid path
    invalid_path = "/nonexistent/directory/predictions.json"
    success = save_predictions(predictions, metrics, invalid_path)
    assert not success

@patch('scripts.predict_numbers.calculate_metrics')
def test_calculate_prediction_metrics(mock_metrics, sample_data):
    """Test calculating prediction metrics."""
    # Setup mock for calculate_metrics
    mock_metrics.return_value = {
        'accuracy': 0.5,
        'match_rate': 3.0,
        'perfect_match_rate': 0.0,
        'rmse': 2.0
    }
    
    # Create predictions
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    
    # Test metrics calculation
    metrics = calculate_prediction_metrics(predictions, sample_data)
    
    # Verify results
    assert 'accuracy' in metrics
    assert 'match_rate' in metrics
    assert 'perfect_match_rate' in metrics
    assert 'rmse' in metrics
    assert mock_metrics.called
    
    # Test with calculation error
    mock_metrics.side_effect = Exception("Calculation error")
    metrics = calculate_prediction_metrics(predictions, sample_data)
    assert 'error' in metrics

@patch('scripts.predict_numbers.get_prediction_weights')
@patch('scripts.predict_numbers.score_combinations')
def test_monte_carlo_simulation(mock_score, mock_weights, sample_data, sample_weights):
    """Test Monte Carlo simulation."""
    # Setup mocks
    mock_weights.return_value = sample_weights
    
    # Mock score_combinations to return sorted combinations
    def mock_score_func(combs, data):
        # Sort by first number for predictable testing
        return [(comb, sum(comb)) for comb in sorted(combs, key=lambda x: x[0])]
    
    mock_score.side_effect = mock_score_func
    
    # Test simulation
    predictions = monte_carlo_simulation(sample_data, n_simulations=3, n_combinations=10)
    
    # Verify results
    assert len(predictions) == 3
    assert all(len(pred) == 6 for pred in predictions)
    assert all(len(set(pred)) == 6 for pred in predictions)
    assert all(all(1 <= n <= 59 for n in pred) for pred in predictions)
    
    # Verify calls
    assert mock_weights.called
    assert mock_score.called
    assert mock_score.call_count == 3  # Once per simulation

@patch('scripts.predict_numbers.predict_next_draw')
def test_backtest(mock_predict, sample_data, mock_models):
    """Test backtesting."""
    # Setup mock
    mock_predict.return_value = [[1, 2, 3, 4, 5, 6]]
    
    # Test backtest with minimal data
    metrics = backtest(mock_models, sample_data, test_size=1)
    
    # Verify results
    assert 'total_predictions' in metrics
    assert 'accuracy' in metrics
    assert 'match_rate' in metrics
    assert 'perfect_match' in metrics
    assert metrics['total_predictions'] == 1
    assert mock_predict.called
    
    # Test with too small dataset
    small_data = pd.DataFrame({'Draw Date': [datetime(2023, 1, 1)], 'Main_Numbers': [[1, 2, 3, 4, 5, 6]]})
    with pytest.raises(ValueError):
        backtest(mock_models, small_data, test_size=2)
    
    # Test with calculation error
    mock_predict.side_effect = Exception("Prediction error")
    metrics = backtest(mock_models, sample_data, test_size=1)
    assert 'error' in metrics

if __name__ == "__main__":
    pytest.main([__file__]) 