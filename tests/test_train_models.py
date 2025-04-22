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
    from scripts.train_models import (
        train_model,
        validate_model_predictions,
        train_all_models,
        load_trained_models,
        save_trained_models
    )
except ImportError:
    try:
        # Try direct imports without 'scripts.' prefix
        from train_models import (
            train_model,
            validate_model_predictions,
            train_all_models,
            load_trained_models,
            save_trained_models
        )
    except ImportError:
        pytest.skip("Could not import train_models module. Skipping tests.", allow_module_level=True)

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
def mock_validator():
    """Fixture to create a mock DataValidator."""
    validator = MagicMock()
    validator.validate_prediction.return_value = (True, "")
    return validator

@pytest.fixture
def models_path(tmp_path):
    """Fixture to create a temporary models path."""
    models_dir = tmp_path / 'models'
    models_dir.mkdir(exist_ok=True)
    return models_dir / 'trained_models.pkl'

def test_train_model():
    """Test training a single model."""
    # Setup mock training function and data
    mock_train_func = MagicMock(return_value=MagicMock())
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    
    # Test with valid parameters
    with patch('scripts.train_models.logger'):
        model = train_model('test_model', mock_train_func, X, y, param1='value')
        
        assert model is not None
        mock_train_func.assert_called_once_with(X, y, param1='value')

def test_train_model_failure():
    """Test training a model that fails."""
    # Setup mock training function that raises an exception
    error_message = "Training error"
    mock_train_func = MagicMock(side_effect=Exception(error_message))
    X = np.array([[1, 2]])
    y = np.array([[1, 2, 3, 4, 5, 6]])
    
    # Test with failing function
    with patch('scripts.train_models.logger') as mock_logger:
        model = train_model('test_model', mock_train_func, X, y)
        
        assert model is None
        mock_train_func.assert_called_once()
        mock_logger.error.assert_called()
        assert error_message in str(mock_logger.error.call_args)

@patch('scripts.train_models.prepare_training_data')
@patch('scripts.train_models.prepare_feature_data')
@patch('scripts.train_models.prepare_sequence_data')
def test_validate_model_predictions(mock_sequence_data, mock_feature_data, mock_training_data, sample_data, mock_validator):
    """Test validating model predictions."""
    # Mock data preparation
    mock_training_data.return_value = (np.array([[[1], [2], [3], [4], [5], [6]]]), np.array([[7, 8, 9, 10, 11, 12]]))
    mock_feature_data.return_value = (np.array([[1, 2]]), np.array([[1, 2, 3, 4, 5, 6]]))
    mock_sequence_data.return_value = (np.array([[[1, 2]]]), np.array([[1, 2, 3, 4, 5, 6]]))
    
    # Test LSTM model
    with patch('scripts.train_models.DataValidator', return_value=mock_validator):
        mock_lstm = MagicMock()
        mock_lstm.predict.return_value = np.array([[1, 2, 3, 4, 5, 6]])
        assert validate_model_predictions('lstm', mock_lstm, sample_data)
        mock_training_data.assert_called()
    
    # Test HoltWinters model
    with patch('scripts.train_models.DataValidator', return_value=mock_validator):
        mock_holtwinters = MagicMock()
        mock_holtwinters.forecast.return_value = np.array([1, 2, 3, 4, 5, 6])
        assert validate_model_predictions('holtwinters', mock_holtwinters, sample_data)
    
    # Test feature-based model (XGBoost)
    with patch('scripts.train_models.DataValidator', return_value=mock_validator):
        mock_xgboost = MagicMock()
        mock_xgboost.predict.return_value = np.array([[1, 2, 3, 4, 5, 6]])
        assert validate_model_predictions('xgboost', mock_xgboost, sample_data)
        mock_feature_data.assert_called()
    
    # Test sequence-based model (CNN_LSTM)
    with patch('scripts.train_models.DataValidator', return_value=mock_validator):
        mock_cnn_lstm = MagicMock()
        mock_cnn_lstm.predict.return_value = np.array([[1, 2, 3, 4, 5, 6]])
        assert validate_model_predictions('cnn_lstm', mock_cnn_lstm, sample_data)
        mock_sequence_data.assert_called()

def test_validate_model_predictions_failure(sample_data):
    """Test validating model predictions with failures."""
    # Test with invalid model type
    with patch('scripts.train_models.logger') as mock_logger:
        mock_model = MagicMock()
        result = validate_model_predictions('unknown_model_type', mock_model, sample_data)
        assert result is False
        mock_logger.warning.assert_called()
    
    # Test with prediction error
    with patch('scripts.train_models.prepare_feature_data') as mock_prepare:
        mock_prepare.return_value = (np.array([[1, 2]]), np.array([[1, 2, 3, 4, 5, 6]]))
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction error")
        
        with patch('scripts.train_models.logger') as mock_logger:
            with patch('scripts.train_models.DataValidator') as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator_class.return_value = mock_validator
                
                result = validate_model_predictions('xgboost', mock_model, sample_data)
                assert result is False
                mock_logger.error.assert_called()

@patch('scripts.train_models.DataValidator')
@patch('scripts.train_models.train_model')
@patch('scripts.train_models.prepare_training_data')
@patch('scripts.train_models.prepare_feature_data')
@patch('scripts.train_models.prepare_sequence_data')
@patch('scripts.train_models.validate_model_predictions')
@patch('scripts.train_models.save_trained_models')
def test_train_all_models(mock_save, mock_validate, mock_sequence_data, mock_feature_data, mock_training_data, 
                          mock_train_model, mock_validator_class, sample_data, models_path):
    """Test training all models."""
    # Mock data preparation
    mock_training_data.return_value = (np.array([[[1]], [[2]]]), np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))
    mock_feature_data.return_value = (np.array([[1], [2]]), np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))
    mock_sequence_data.return_value = (np.array([[[1]], [[2]]]), np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]))
    
    # Mock model training and validation
    mock_train_model.return_value = MagicMock()
    mock_validate.return_value = True
    mock_save.return_value = True
    
    # Set up validator
    mock_validator = MagicMock()
    mock_validator.validate_prediction.return_value = (True, "")
    mock_validator_class.return_value = mock_validator
    
    # Test with all models enabled
    with patch('scripts.train_models.config', {'models': {'lstm': True, 'holtwinters': True, 'xgboost': True, 'catboost': True}}):
        models = train_all_models(sample_data, models_dir=str(models_path.parent), force_retrain=True)
        
        # Verify that models were trained and saved
        assert isinstance(models, dict)
        assert len(models) > 0
        assert mock_train_model.call_count >= 4  # At least LSTM, HoltWinters, XGBoost, CatBoost
        assert mock_save.called
    
    # Test with selective models enabled
    mock_train_model.reset_mock()
    with patch('scripts.train_models.config', {'models': {'lstm': True, 'holtwinters': False, 'xgboost': False, 'catboost': False}}):
        models = train_all_models(sample_data, models_dir=str(models_path.parent), force_retrain=True)
        
        # Verify only LSTM was trained
        assert len(models) == 1
        assert mock_train_model.call_count == 1

@patch('scripts.train_models.Path.exists')
def test_load_trained_models_success(mock_exists, models_path):
    """Test loading trained models."""
    mock_exists.return_value = True
    mock_data = {
        'models': {'lstm': MagicMock(), 'xgboost': MagicMock()},
        'timestamp': '2023-01-01T00:00:00',
        'data_size': 100
    }
    
    m = mock_open(read_data=pickle.dumps(mock_data))
    with patch('builtins.open', m):
        models = load_trained_models(models_dir=str(models_path.parent))
        
        assert isinstance(models, dict)
        assert len(models) == 2
        assert set(models.keys()) == {'lstm', 'xgboost'}

def test_load_trained_models_no_file():
    """Test loading models when file doesn't exist."""
    with patch('scripts.train_models.Path.exists', return_value=False):
        with patch('scripts.train_models.logger') as mock_logger:
            models = load_trained_models()
            
            assert models == {}
            mock_logger.warning.assert_called()

def test_load_trained_models_invalid_file(models_path):
    """Test loading models from a corrupted or invalid file."""
    with patch('scripts.train_models.Path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data=b'invalid data')):
            with patch('scripts.train_models.logger') as mock_logger:
                models = load_trained_models(models_dir=str(models_path.parent))
                
                assert models == {}
                mock_logger.error.assert_called()

def test_save_trained_models(models_path):
    """Test saving trained models."""
    models = {'lstm': MagicMock(), 'xgboost': MagicMock()}
    data_size = 100
    
    m = mock_open()
    with patch('builtins.open', m):
        with patch('scripts.train_models.logger'):
            result = save_trained_models(models, data_size, models_dir=str(models_path.parent))
            
            assert result is True
            m.assert_called_once_with(models_path, 'wb')
            handle = m()
            assert handle.write.called

def test_save_trained_models_failure():
    """Test saving trained models with a failure."""
    models = {'lstm': MagicMock()}
    data_size = 100
    
    with patch('builtins.open', side_effect=Exception("File error")):
        with patch('scripts.train_models.logger') as mock_logger:
            result = save_trained_models(models, data_size, models_dir="/nonexistent/directory")
            
            assert result is False
            mock_logger.error.assert_called()

@patch('scripts.train_models.train_all_models')
@patch('scripts.train_models.load_trained_models')
def test_model_retraining_logic(mock_load, mock_train, sample_data):
    """Test the logic for when to retrain models vs. loading existing ones."""
    # Mock existing models
    mock_models = {'lstm': MagicMock(), 'xgboost': MagicMock()}
    mock_load.return_value = mock_models
    mock_train.return_value = {'lstm': MagicMock(), 'xgboost': MagicMock(), 'catboost': MagicMock()}
    
    # Test with force_retrain=True
    train_all_models(sample_data, force_retrain=True)
    mock_train.assert_called_once()
    mock_load.assert_not_called()
    
    # Reset mocks
    mock_train.reset_mock()
    mock_load.reset_mock()
    
    # Test with force_retrain=False and existing models
    train_all_models(sample_data, force_retrain=False)
    mock_load.assert_called_once()
    mock_train.assert_not_called()
    
    # Reset mocks
    mock_train.reset_mock()
    mock_load.reset_mock()
    
    # Test with force_retrain=False and no existing models
    mock_load.return_value = {}
    train_all_models(sample_data, force_retrain=False)
    mock_load.assert_called_once()
    mock_train.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__]) 