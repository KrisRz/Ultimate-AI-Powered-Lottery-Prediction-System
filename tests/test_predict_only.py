import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import pickle
import os
import sys
import json
from unittest.mock import patch, MagicMock, mock_open

# Add project root to path to resolve imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

try:
    from scripts.predict_only import (
        load_trained_models,
        predict_next_draw,
        generate_next_draw_predictions,
        format_predictions
    )
except ImportError:
    try:
        # Try direct imports without 'scripts.' prefix
        from predict_only import (
            load_trained_models,
            predict_next_draw,
            generate_next_draw_predictions,
            format_predictions
        )
    except ImportError:
        pytest.skip("Could not import predict_only module. Skipping tests.", allow_module_level=True)

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
def models_path(tmp_path):
    """Fixture to create a temporary models path with mock models."""
    models_dir = tmp_path / 'models'
    models_dir.mkdir(exist_ok=True)
    models_path = models_dir / 'trained_models.pkl'
    
    mock_models = {
        'lstm': MagicMock(),
        'xgboost': MagicMock(),
        'catboost': MagicMock()
    }
    
    mock_data = {
        'models': mock_models,
        'timestamp': '2023-01-01T00:00:00',
        'data_size': 100
    }
    
    with open(models_path, 'wb') as f:
        pickle.dump(mock_data, f)
    
    return models_path

@pytest.fixture
def corrupted_models_path(tmp_path):
    """Fixture to create a corrupted models file."""
    models_dir = tmp_path / 'models'
    models_dir.mkdir(exist_ok=True)
    models_path = models_dir / 'trained_models.pkl'
    
    with open(models_path, 'w') as f:
        f.write("This is not a valid pickle file")
    
    return models_path

def test_load_trained_models(models_path):
    """Test loading trained models from file."""
    with patch('scripts.predict_only.MODELS_PATH', str(models_path)):
        models = load_trained_models()
        
        assert isinstance(models, dict)
        assert set(models.keys()) == {'lstm', 'xgboost', 'catboost'}
        assert len(models) == 3

def test_load_trained_models_no_file():
    """Test loading models when file doesn't exist."""
    with patch('scripts.predict_only.MODELS_PATH', '/nonexistent/path/models.pkl'):
        with patch('scripts.predict_only.logger') as mock_logger:
            models = load_trained_models()
            
            assert models is None
            mock_logger.error.assert_called()

def test_load_trained_models_corrupted_file(corrupted_models_path):
    """Test loading models from corrupted file."""
    with patch('scripts.predict_only.MODELS_PATH', str(corrupted_models_path)):
        with patch('scripts.predict_only.logger') as mock_logger:
            models = load_trained_models()
            
            assert models is None
            mock_logger.error.assert_called()

@patch('scripts.predict_only.load_data')
@patch('scripts.predict_only.enhance_features')
@patch('scripts.predict_only.DataValidator')
@patch('scripts.predict_only.load_trained_models')
@patch('scripts.predict_only.generate_next_draw_predictions')
@patch('scripts.predict_only.format_predictions')
def test_predict_next_draw(mock_format, mock_generate, mock_load_models, 
                          mock_validator_class, mock_enhance, mock_load_data, 
                          sample_data, mock_models):
    """Test predicting the next draw with all successful steps."""
    # Mock dependencies
    mock_load_data.return_value = sample_data
    
    enhanced_data = sample_data.copy()
    enhanced_data['feature1'] = [1, 2]
    mock_enhance.return_value = enhanced_data
    
    mock_validator = MagicMock()
    mock_validator.validate_data.return_value = {'errors': [], 'warnings': []}
    mock_validator_class.return_value = mock_validator
    
    mock_load_models.return_value = mock_models
    mock_generate.return_value = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    mock_format.return_value = "Predictions formatted"
    
    # Call function
    result = predict_next_draw(data_path='data.csv', n_predictions=2)
    
    # Verify results
    assert 'next_draw_predictions' in result
    assert 'validation_results' in result
    assert len(result['next_draw_predictions']) == 2
    assert result['next_draw_predictions'] == [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    assert result['formatted_predictions'] == "Predictions formatted"
    
    # Verify calls
    mock_load_data.assert_called_with('data.csv')
    mock_enhance.assert_called_with(sample_data)
    mock_validator.validate_data.assert_called()
    mock_load_models.assert_called()
    mock_generate.assert_called_with(mock_models, enhanced_data, 2)
    mock_format.assert_called_with([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

@patch('scripts.predict_only.load_data')
@patch('scripts.predict_only.enhance_features')
@patch('scripts.predict_only.DataValidator')
@patch('scripts.predict_only.load_trained_models')
def test_predict_next_draw_no_models(mock_load_models, mock_validator_class, 
                                   mock_enhance, mock_load_data, sample_data):
    """Test predict_next_draw when no models are found."""
    # Mock dependencies
    mock_load_data.return_value = sample_data
    mock_enhance.return_value = sample_data
    
    mock_validator = MagicMock()
    mock_validator.validate_data.return_value = {'errors': [], 'warnings': []}
    mock_validator_class.return_value = mock_validator
    
    mock_load_models.return_value = None
    
    # Call function
    result = predict_next_draw(data_path='data.csv')
    
    # Verify results
    assert 'error' in result
    assert result['error'] == 'No pre-trained models found'
    
    # Verify calls
    mock_load_data.assert_called_with('data.csv')
    mock_validator.validate_data.assert_called()
    mock_load_models.assert_called()

@patch('scripts.predict_only.load_data')
@patch('scripts.predict_only.enhance_features')
@patch('scripts.predict_only.DataValidator')
def test_predict_next_draw_validation_failure(mock_validator_class, mock_enhance, 
                                            mock_load_data, sample_data):
    """Test predict_next_draw when data validation fails."""
    # Mock dependencies
    mock_load_data.return_value = sample_data
    mock_enhance.return_value = sample_data
    
    mock_validator = MagicMock()
    mock_validator.validate_data.return_value = {'errors': ['Invalid data format'], 'warnings': []}
    mock_validator_class.return_value = mock_validator
    
    # Call function
    result = predict_next_draw(data_path='data.csv')
    
    # Verify results
    assert 'error' in result
    assert result['error'] == 'Data validation failed'
    assert 'validation_results' in result
    assert result['validation_results']['errors'] == ['Invalid data format']
    
    # Verify calls
    mock_load_data.assert_called_with('data.csv')
    mock_validator.validate_data.assert_called()

@patch('scripts.predict_only.load_data')
def test_predict_next_draw_load_data_failure(mock_load_data):
    """Test predict_next_draw when loading data fails."""
    # Mock dependencies
    mock_load_data.side_effect = Exception("Could not load data")
    
    # Call function
    result = predict_next_draw(data_path='data.csv')
    
    # Verify results
    assert 'error' in result
    assert 'Could not load data' in result['error']
    
    # Verify calls
    mock_load_data.assert_called_with('data.csv')

@patch('scripts.predict_only.predict_with_model')
def test_generate_next_draw_predictions(mock_predict, sample_data, mock_models):
    """Test generating predictions with different models."""
    # Mock dependencies
    mock_predict.side_effect = [
        [1, 2, 3, 4, 5, 6],  # lstm
        [7, 8, 9, 10, 11, 12],  # xgboost
        [13, 14, 15, 16, 17, 18]  # catboost
    ]
    
    # Call function
    predictions = generate_next_draw_predictions(mock_models, sample_data, n_predictions=3)
    
    # Verify results
    assert len(predictions) == 3
    assert [1, 2, 3, 4, 5, 6] in predictions
    assert [7, 8, 9, 10, 11, 12] in predictions
    assert [13, 14, 15, 16, 17, 18] in predictions
    
    # Verify calls
    assert mock_predict.call_count == 3
    mock_predict.assert_any_call('lstm', mock_models['lstm'], sample_data)
    mock_predict.assert_any_call('xgboost', mock_models['xgboost'], sample_data)
    mock_predict.assert_any_call('catboost', mock_models['catboost'], sample_data)

def test_format_predictions():
    """Test formatting predictions for display."""
    # Test data
    predictions = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18]
    ]
    
    # Call function
    formatted = format_predictions(predictions)
    
    # Verify results
    assert isinstance(formatted, str)
    assert "Prediction 1: 1-2-3-4-5-6" in formatted
    assert "Prediction 2: 7-8-9-10-11-12" in formatted
    assert "Prediction 3: 13-14-15-16-17-18" in formatted

if __name__ == "__main__":
    pytest.main([__file__]) 