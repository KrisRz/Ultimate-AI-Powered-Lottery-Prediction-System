import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root directory to the path to resolve imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

# Import the main module - we're using a try/except to handle potential import issues
try:
    from main import LotteryPredictor, main
except ImportError:
    try:
        from scripts.main import LotteryPredictor, main
    except ImportError:
        pytest.skip("Could not import LotteryPredictor from main.py. Skipping tests.", allow_module_level=True)

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def predictor():
    """Fixture to create a LotteryPredictor instance."""
    return LotteryPredictor()

@pytest.fixture
def sample_data():
    """Fixture to create a sample DataFrame for testing."""
    data = {
        'draw_date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)],
        'winning_numbers': [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv(tmp_path):
    """Fixture to create a sample CSV file."""
    data = {
        'draw_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'winning_numbers': ['1,2,3,4,5,6', '7,8,9,10,11,12', '13,14,15,16,17,18']
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'lottery_history.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

def test_init_predictor(predictor):
    """Test initialization of LotteryPredictor."""
    assert isinstance(predictor.models, dict)
    assert set(predictor.models.keys()) == {'xgboost', 'lightgbm', 'catboost', 'lstm', 'prophet'}
    assert predictor.data is None

def test_load_data_success(predictor, sample_csv):
    """Test loading data from a valid CSV file."""
    result = predictor.load_data(str(sample_csv))
    assert result is True
    assert isinstance(predictor.data, pd.DataFrame)
    assert len(predictor.data) == 3
    assert set(predictor.data.columns) == {'draw_date', 'winning_numbers'}

def test_load_data_failure(predictor):
    """Test loading data from a non-existent file."""
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        result = predictor.load_data('non_existent.csv')
        assert result is False
        assert predictor.data is None

def test_prepare_features(predictor, sample_data):
    """Test feature preparation."""
    predictor.data = sample_data.copy()
    result = predictor.prepare_features()
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    # After dropping NaN from lag features, we should have fewer rows
    assert len(result) < len(sample_data)  
    
    # Check expected columns exist
    expected_columns = {'draw_date', 'winning_numbers', 'day_of_week', 'month', 'year'}
    for lag in range(1, 4):
        expected_columns.add(f'number_lag_{lag}')
    
    assert set(result.columns).issuperset(expected_columns)
    
    # Check data types and values
    assert result['day_of_week'].iloc[0] in range(7)
    assert result['month'].iloc[0] in range(1, 13)
    assert result['year'].iloc[0] == 2023

def test_prepare_features_no_data(predictor):
    """Test prepare_features with no data."""
    result = predictor.prepare_features()
    assert result is None

@patch('xgboost.XGBRegressor.fit')
@patch('lightgbm.LGBMRegressor.fit')
@patch('catboost.CatBoostRegressor.fit')
@patch('tensorflow.keras.models.Sequential.fit')
@patch('prophet.Prophet.fit')
def test_train_models(mock_prophet_fit, mock_lstm_fit, mock_catboost_fit, mock_lightgbm_fit,
                      mock_xgboost_fit, predictor, sample_data):
    """Test training all models."""
    # Set up predictor with prepared data
    predictor.data = sample_data.copy()
    predictor.data = predictor.prepare_features()
    
    # Configure mocks
    mock_xgboost_fit.return_value = None
    mock_lightgbm_fit.return_value = None
    mock_catboost_fit.return_value = None
    mock_lstm_fit.return_value = None
    mock_prophet_fit.return_value = None
    
    # Train models
    result = predictor.train_models()
    
    # Verify result and that all models were trained
    assert result is True
    mock_xgboost_fit.assert_called_once()
    mock_lightgbm_fit.assert_called_once()
    mock_catboost_fit.assert_called_once()
    mock_lstm_fit.assert_called_once()
    mock_prophet_fit.assert_called_once()

def test_train_models_no_data(predictor):
    """Test train_models with no data."""
    result = predictor.train_models()
    assert result is False

@patch('xgboost.XGBRegressor.predict')
@patch('lightgbm.LGBMRegressor.predict')
@patch('catboost.CatBoostRegressor.predict')
@patch('tensorflow.keras.models.Sequential.predict')
@patch('prophet.Prophet.make_future_dataframe')
@patch('prophet.Prophet.predict')
def test_predict_next_numbers(mock_prophet_predict, mock_prophet_future, mock_lstm_predict, 
                             mock_catboost_predict, mock_lightgbm_predict, mock_xgboost_predict, 
                             predictor, sample_data):
    """Test generating predictions."""
    # Set up predictor with prepared data
    predictor.data = sample_data.copy()
    predictor.data = predictor.prepare_features()
    
    # Configure prophet mock 
    mock_prophet_future.return_value = pd.DataFrame({'ds': [datetime.now()]})
    mock_prophet_predict.return_value = pd.DataFrame({'yhat': [5]})
    
    # Configure other model mocks
    mock_xgboost_predict.return_value = np.array([1])
    mock_lightgbm_predict.return_value = np.array([2])
    mock_catboost_predict.return_value = np.array([3])
    mock_lstm_predict.return_value = np.array([[4]])
    
    # Generate predictions
    predictions = predictor.predict_next_numbers()
    
    # Verify predictions
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == {'xgboost', 'lightgbm', 'catboost', 'lstm', 'prophet'}
    assert predictions['xgboost'] == 1
    assert predictions['lightgbm'] == 2
    assert predictions['catboost'] == 3
    assert predictions['lstm'] == 4
    assert predictions['prophet'] == 5

def test_predict_next_numbers_error_handling(predictor, sample_data):
    """Test error handling in predict_next_numbers."""
    # Set up predictor with prepared data
    predictor.data = sample_data.copy()
    predictor.data = predictor.prepare_features()
    
    # Replace one model with something that will cause an error
    predictor.models['xgboost'] = None
    
    # Should not raise an exception
    predictions = predictor.predict_next_numbers()
    assert isinstance(predictions, dict)
    assert 'xgboost' not in predictions  # XGBoost prediction should be omitted due to error

def test_main():
    """Test the main function."""
    with patch('main.LotteryPredictor.load_data', return_value=True) as mock_load, \
         patch('main.LotteryPredictor.prepare_features', return_value=pd.DataFrame()) as mock_prepare, \
         patch('main.LotteryPredictor.train_models', return_value=True) as mock_train, \
         patch('main.LotteryPredictor.predict_next_numbers', return_value={'xgboost': 1}) as mock_predict, \
         patch('builtins.print') as mock_print:
        
        # Run main function
        main()
        
        # Verify all methods were called in the correct order
        mock_load.assert_called_once()
        mock_prepare.assert_called_once()
        mock_train.assert_called_once()
        mock_predict.assert_called_once()
        mock_print.assert_called()  # At least one print call

def test_main_data_load_failure():
    """Test main when data loading fails."""
    with patch('main.LotteryPredictor.load_data', return_value=False) as mock_load, \
         patch('logging.error') as mock_log:
        
        # Run main function
        main()
        
        # Verify logging error was called
        mock_load.assert_called_once()
        mock_log.assert_called_once()

def test_main_train_failure():
    """Test main when training fails."""
    with patch('main.LotteryPredictor.load_data', return_value=True) as mock_load, \
         patch('main.LotteryPredictor.prepare_features', return_value=pd.DataFrame()) as mock_prepare, \
         patch('main.LotteryPredictor.train_models', return_value=False) as mock_train, \
         patch('logging.error') as mock_log:
        
        # Run main function
        main()
        
        # Verify methods were called in the correct order and error was logged
        mock_load.assert_called_once()
        mock_prepare.assert_called_once()
        mock_train.assert_called_once()
        mock_log.assert_called_once()

def test_integrated_workflow(predictor, sample_csv):
    """Test an integrated workflow with minimal mocking."""
    # Load real data from CSV
    success = predictor.load_data(str(sample_csv))
    assert success is True
    
    # Prepare features with the actual implementation
    prepared_data = predictor.prepare_features()
    assert prepared_data is not None
    
    # Mock only the model training to avoid dependencies
    with patch.object(predictor.models['xgboost'], 'fit'), \
         patch.object(predictor.models['lightgbm'], 'fit'), \
         patch.object(predictor.models['catboost'], 'fit'), \
         patch.object(predictor.models['prophet'], 'fit'), \
         patch.object(predictor, '_create_lstm_model', return_value=MagicMock()):
        
        # Train the models
        success = predictor.train_models()
        assert success is True
        
        # Mock predictions to test end-to-end workflow
        with patch.object(predictor.models['xgboost'], 'predict', return_value=np.array([1])), \
             patch.object(predictor.models['lightgbm'], 'predict', return_value=np.array([2])), \
             patch.object(predictor.models['catboost'], 'predict', return_value=np.array([3])), \
             patch.object(predictor.models['lstm'], 'predict', return_value=np.array([[4]])), \
             patch.object(predictor.models['prophet'], 'make_future_dataframe', return_value=pd.DataFrame({'ds': [datetime.now()]})), \
             patch.object(predictor.models['prophet'], 'predict', return_value=pd.DataFrame({'yhat': [5]})):
            
            # Generate predictions
            predictions = predictor.predict_next_numbers()
            assert isinstance(predictions, dict)
            assert len(predictions) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 