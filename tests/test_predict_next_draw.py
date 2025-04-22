import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
from unittest.mock import patch, MagicMock
from scripts.predict_next_draw import (
    generate_next_draw_predictions,
    format_predictions,
    ensure_valid_prediction,
    validate_predictions,
    calculate_prediction_metrics,
    save_predictions,
    load_predictions,
    format_predictions_for_display,
    validate_and_save_predictions
)

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
def mock_models():
    """Fixture to create mock models."""
    return {
        'lstm': MagicMock(),
        'xgboost': MagicMock(),
        'catboost': MagicMock()
    }

@pytest.fixture
def predictions_file(tmp_path):
    """Fixture to create a temporary predictions file."""
    predictions_path = tmp_path / 'predictions.json'
    predictions_data = {
        'timestamp': '2023-01-01T00:00:00',
        'predictions': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        'count': 2,
        'metrics': {'accuracy': 0.5}
    }
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f)
    return predictions_path

@patch('scripts.predict_next_draw.get_model_weights')
@patch('scripts.predict_next_draw.predict_lstm')
@patch('scripts.predict_next_draw.predict_xgboost')
@patch('scripts.predict_next_draw.predict_catboost')
def test_generate_next_draw_predictions(mock_catboost, mock_xgboost, mock_lstm, mock_get_weights, sample_data, mock_models):
    """Test generating next draw predictions."""
    mock_get_weights.return_value = {'lstm': 0.4, 'xgboost': 0.4, 'catboost': 0.2}
    mock_lstm.return_value = np.array([1, 2, 3, 4, 5, 6])
    mock_xgboost.return_value = np.array([2, 3, 4, 5, 6, 7])
    mock_catboost.return_value = np.array([3, 4, 5, 6, 7, 8])
    
    predictions = generate_next_draw_predictions(mock_models, sample_data.iloc[-1:], n_predictions=2)
    assert len(predictions) == 2
    assert all(len(pred) == 6 for pred in predictions)
    assert all(len(set(pred)) == 6 for pred in predictions)
    assert all(all(1 <= n <= 59 for n in pred) for pred in predictions)
    mock_lstm.assert_called()
    mock_xgboost.assert_called()
    mock_catboost.assert_called()

def test_format_predictions():
    """Test formatting predictions for display."""
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    formatted = format_predictions(predictions)
    assert "Next Draw Predictions" in formatted
    assert "Prediction 1: 1, 2, 3, 4, 5, 6" in formatted
    assert "Prediction 2: 7, 8, 9, 10, 11, 12" in formatted

def test_ensure_valid_prediction():
    """Test ensuring valid predictions."""
    # Valid prediction
    valid_pred = [1, 2, 3, 4, 5, 6]
    assert ensure_valid_prediction(valid_pred) == [1, 2, 3, 4, 5, 6]
    
    # Invalid predictions
    invalid_pred = [1, 2, 3, 4, 5, 60]  # Out of range
    result = ensure_valid_prediction(invalid_pred)
    assert len(result) == 6
    assert len(set(result)) == 6
    assert all(1 <= n <= 59 for n in result)

@patch('scripts.predict_next_draw.validate_prediction')
def test_validate_predictions(mock_validate):
    """Test validating a list of predictions."""
    predictions = [
        [1, 2, 3, 4, 5, 6],  # Valid
        [1, 2, 3, 4, 5, 60]  # Invalid
    ]
    mock_validate.side_effect = [(True, ""), (False, "Out of range")]
    
    validated, invalid_indices = validate_predictions(predictions)
    assert len(validated) == 2
    assert validated[0] == [1, 2, 3, 4, 5, 6]
    assert len(invalid_indices) == 1
    assert invalid_indices == [1]
    assert len(set(validated[1])) == 6

@patch('scripts.predict_next_draw.calculate_metrics')
def test_calculate_prediction_metrics(mock_calculate_metrics, sample_data):
    """Test calculating prediction metrics."""
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    mock_calculate_metrics.return_value = {'accuracy': 0.5, 'match_rate': 3.0}
    
    metrics = calculate_prediction_metrics(predictions, sample_data)
    assert metrics['accuracy'] == 0.5
    assert metrics['match_rate'] == 3.0
    mock_calculate_metrics.assert_called()

def test_calculate_prediction_metrics_missing_column(sample_data):
    """Test calculating metrics with missing Main_Numbers."""
    df = pd.DataFrame({'Draw Date': sample_data['Draw Date']})
    predictions = [[1, 2, 3, 4, 5, 6]]
    metrics = calculate_prediction_metrics(predictions, df)
    assert 'error' in metrics
    assert metrics['error'] == "No 'Main_Numbers' column in data"

def test_save_predictions(predictions_file):
    """Test saving predictions to file."""
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    metrics = {'accuracy': 0.5}
    success = save_predictions(predictions, metrics, str(predictions_file))
    assert success
    with open(predictions_file, 'r') as f:
        saved = json.load(f)
    assert saved['count'] == 2
    assert saved['predictions'] == predictions
    assert saved['metrics'] == metrics

def test_load_predictions(predictions_file):
    """Test loading predictions from file."""
    data = load_predictions(str(predictions_file))
    assert data['count'] == 2
    assert data['predictions'] == [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    assert data['metrics'] == {'accuracy': 0.5}

def test_load_predictions_no_file(tmp_path):
    """Test loading predictions when file doesn't exist."""
    data = load_predictions(str(tmp_path / 'non_existent.json'))
    assert 'error' in data
    assert data['predictions'] == []

def test_format_predictions_for_display():
    """Test formatting predictions for display."""
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    metrics = {'accuracy': 0.5, 'match_rate': 3.0}
    formatted = format_predictions_for_display(predictions, metrics, title="Test Predictions")
    assert "TEST PREDICTIONS" in formatted
    assert "Prediction 1: 01 - 02 - 03 - 04 - 05 - 06" in formatted
    assert "Prediction 2: 07 - 08 - 09 - 10 - 11 - 12" in formatted
    assert "Accuracy: 0.5000" in formatted
    assert "Match Rate: 3.0000" in formatted

@patch('scripts.predict_next_draw.validate_predictions')
@patch('scripts.predict_next_draw.calculate_prediction_metrics')
@patch('scripts.predict_next_draw.save_predictions')
@patch('scripts.predict_next_draw.format_predictions_for_display')
def test_validate_and_save_predictions(mock_format, mock_save, mock_metrics, mock_validate, sample_data):
    """Test validating and saving predictions."""
    predictions = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    mock_validate.return_value = (predictions, [])
    mock_metrics.return_value = {'accuracy': 0.5}
    mock_save.return_value = True
    mock_format.return_value = "Formatted predictions"
    
    result = validate_and_save_predictions(predictions, sample_data, output_path='results/predictions.json')
    assert result['predictions'] == predictions
    assert result['metrics'] == {'accuracy': 0.5}
    assert result['invalid_count'] == 0
    assert result['success'] is True
    assert result['display'] == "Formatted predictions"
    mock_validate.assert_called()
    mock_metrics.assert_called()
    mock_save.assert_called()

if __name__ == "__main__":
    pytest.main([__file__])