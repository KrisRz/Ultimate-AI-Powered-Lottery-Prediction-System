import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
import os
import sys

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

try:
    from scripts.data_validation import DataValidator, validate_dataframe, validate_prediction
except ImportError:
    try:
        from data_validation import DataValidator, validate_dataframe, validate_prediction
    except ImportError:
        pytest.skip("Could not import data_validation module. Skipping tests.", allow_module_level=True)

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_valid_data():
    """Fixture to create a valid sample DataFrame for testing."""
    data = {
        'Draw Date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'Day': ['Sun', 'Mon'],
        'Balls': ['1 2 3 4 5 6 BONUS 7', '8 9 10 11 12 13 BONUS 14'],
        'Jackpot': ['£1,000,000', '£2,000,000'],
        'Winners': [0, 1],
        'Draw Details': ['Draw 1', 'Draw 2'],
        'Main_Numbers': [[1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13]],
        'Bonus': [7, 14]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_invalid_data():
    """Fixture to create an invalid sample DataFrame for testing."""
    data = {
        'Draw Date': [datetime(2023, 1, 1), 'invalid_date', datetime(2023, 1, 2)],
        'Day': ['Sun', 'Invalid', 'Mon'],
        'Balls': ['1 2 3 4 5 6 BONUS 7', 'invalid', '1 1 1 1 1 1 BONUS 1'],
        'Jackpot': ['£1,000,000', 'invalid', '£2,000,000'],
        'Winners': [0, -1, 1],
        'Draw Details': ['Draw 1', None, 'Draw 2'],
        'Main_Numbers': [[1, 2, 3, 4, 5, 6], [1], [1, 1, 1, 1, 1, 1]],
        'Bonus': [7, 60, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def validator():
    """Fixture to create a DataValidator instance."""
    return DataValidator()

def test_init_validator(validator):
    """Test initialization of DataValidator."""
    assert isinstance(validator.validation_rules, dict)
    assert 'Draw Date' in validator.validation_rules
    assert validator.history_file == Path('results/validation_log.json')

def test_validate_valid_data(validator, sample_valid_data):
    """Test validation of valid data."""
    is_valid, results = validator.validate_data(sample_valid_data, fix_issues=True)
    assert is_valid
    assert len(results['errors']) == 0
    assert len(results['warnings']) == 0
    assert results['data_shape']['rows'] == 2
    assert set(results['data_shape']['columns']) == set(sample_valid_data.columns)

def test_validate_invalid_data(validator, sample_invalid_data):
    """Test validation of invalid data."""
    is_valid, results = validator.validate_data(sample_invalid_data, fix_issues=True)
    assert not is_valid
    assert len(results['errors']) > 0
    assert any('Draw Date' in e for e in results['errors'])
    assert any('Day' in e for e in results['errors'])
    assert any('Balls' in e for e in results['errors'])
    assert any('Winners' in e for e in results['errors'])
    assert any('Bonus' in e for e in results['errors'])

def test_fix_issues(validator, sample_invalid_data):
    """Test automatic fixing of issues."""
    is_valid, results = validator.validate_data(sample_invalid_data, fix_issues=True)
    assert len(results['fixes_applied']) > 0
    assert any('Winners' in f for f in results['fixes_applied'])
    assert any('Draw Details' in f for f in results['fixes_applied'])

def test_validate_column_datetime(validator, sample_valid_data):
    """Test validation of datetime column."""
    results = validator._validate_column(sample_valid_data, 'Draw Date', validator.validation_rules['Draw Date'], True)
    assert len(results['errors']) == 0
    assert len(results['warnings']) == 0

def test_validate_column_invalid_datetime(validator):
    """Test validation of invalid datetime column."""
    df = pd.DataFrame({'Draw Date': ['invalid_date', '2023-01-01']})
    results = validator._validate_column(df, 'Draw Date', validator.validation_rules['Draw Date'], True)
    assert len(results['errors']) > 0
    assert any('datetime' in e.lower() for e in results['errors'])

def test_validate_balls_format(validator, sample_valid_data):
    """Test custom validation for Balls column."""
    errors, warnings, fixes = validator._validate_balls_format(sample_valid_data, 'Balls', True)
    assert len(errors) == 0
    assert len(warnings) == 0
    assert len(fixes) == 0

def test_validate_invalid_balls_format(validator):
    """Test validation of invalid Balls format."""
    df = pd.DataFrame({'Balls': ['1 2 3 4 5 6', 'invalid', '1 1 1 1 1 1 BONUS 1']})
    errors, warnings, fixes = validator._validate_balls_format(df, 'Balls', True)
    assert len(errors) > 0
    assert any('format' in e.lower() for e in errors)
    assert len(fixes) > 0

def test_validate_main_numbers(validator, sample_valid_data):
    """Test custom validation for Main_Numbers column."""
    errors, warnings, fixes = validator._validate_main_numbers(sample_valid_data, 'Main_Numbers', True)
    assert len(errors) == 0
    assert len(warnings) == 0
    assert len(fixes) == 0

def test_validate_invalid_main_numbers(validator):
    """Test validation of invalid Main_Numbers."""
    df = pd.DataFrame({'Main_Numbers': [[1], [1, 2, 3, 4, 5, 'invalid'], [1, 1, 1, 1, 1, 1]]})
    errors, warnings, fixes = validator._validate_main_numbers(df, 'Main_Numbers', True)
    assert len(errors) > 0
    assert any('length' in e.lower() for e in errors)
    assert any('integers' in e.lower() for e in errors)
    assert len(warnings) > 0
    assert any('duplicates' in w.lower() for w in warnings)

def test_check_consistency(validator, sample_valid_data):
    """Test consistency checks across columns."""
    results = validator._check_consistency(sample_valid_data)
    assert len(results['errors']) == 0
    assert len(results['warnings']) == 0

def test_validate_prediction_valid():
    """Test validation of a valid prediction."""
    prediction = [1, 2, 3, 4, 5, 6]
    is_valid, error_msg = validate_prediction(prediction)
    assert is_valid
    assert error_msg == ""

def test_validate_prediction_invalid():
    """Test validation of invalid predictions."""
    predictions = [
        [1, 2, 3, 4, 5],  # Too short
        [1, 2, 3, 4, 5, 60],  # Out of range
        [1, 1, 2, 3, 4, 5],  # Duplicates
        ['1', 2, 3, 4, 5, 6],  # Wrong type
    ]
    for pred in predictions:
        is_valid, error_msg = validate_prediction(pred)
        assert not is_valid
        assert error_msg != ""

def test_save_validation_results(validator, sample_valid_data, tmp_path):
    """Test saving validation results to file."""
    validator.history_file = tmp_path / 'validation_log.json'
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': {'rows': 2, 'columns': list(sample_valid_data.columns)},
        'checks': {},
        'errors': [],
        'warnings': [],
        'fixes_applied': []
    }
    validator._save_validation_results(results)
    assert validator.history_file.exists()
    with open(validator.history_file, 'r') as f:
        saved = json.load(f)
    assert len(saved) == 1
    assert saved[0]['timestamp'] == results['timestamp']

def test_get_validation_history(validator, tmp_path):
    """Test retrieving validation history."""
    validator.history_file = tmp_path / 'validation_log.json'
    history_data = [{'timestamp': '2023-01-01T00:00:00', 'data_shape': {'rows': 2, 'columns': []}}]
    with open(validator.history_file, 'w') as f:
        json.dump(history_data, f)
    history = validator.get_validation_history()
    assert len(history) == 1
    assert history[0]['timestamp'] == '2023-01-01T00:00:00'

def test_validate_dataframe(sample_valid_data):
    """Test the standalone validate_dataframe function."""
    is_valid, results = validate_dataframe(sample_valid_data, fix_issues=True)
    assert is_valid
    assert len(results['errors']) == 0
    assert results['data_shape']['rows'] == 2

def test_validate_dataframe_with_missing_columns(validator):
    """Test validation with missing required columns."""
    df = pd.DataFrame({
        'Draw Date': [datetime(2023, 1, 1)],
        'Day': ['Sun']
        # Missing Main_Numbers and Bonus
    })
    is_valid, results = validator.validate_data(df, fix_issues=True)
    assert not is_valid
    assert any('required' in e.lower() for e in results['errors'])

def test_validate_dataframe_with_extra_columns(validator, sample_valid_data):
    """Test validation with extra columns."""
    df = sample_valid_data.copy()
    df['Extra'] = [1, 2]
    is_valid, results = validator.validate_data(df, fix_issues=True)
    assert is_valid  # Extra columns should not cause validation failure
    assert len(results['warnings']) > 0
    assert any('extra' in w.lower() for w in results['warnings'])

def test_validate_empty_dataframe(validator):
    """Test validation with an empty DataFrame."""
    df = pd.DataFrame()
    is_valid, results = validator.validate_data(df, fix_issues=True)
    assert not is_valid
    assert len(results['errors']) > 0
    assert any('empty' in e.lower() for e in results['errors'])

def test_extraction_from_balls(validator):
    """Test extraction of Main_Numbers and Bonus from Balls column."""
    df = pd.DataFrame({
        'Balls': ['1 2 3 4 5 6 BONUS 7']
    })
    is_valid, results = validator.validate_data(df, fix_issues=True)
    assert 'Main_Numbers' in df.columns
    assert 'Bonus' in df.columns
    assert df['Main_Numbers'][0] == [1, 2, 3, 4, 5, 6]
    assert df['Bonus'][0] == 7

def test_jackpot_cleaning(validator):
    """Test cleaning of Jackpot column."""
    df = pd.DataFrame({
        'Jackpot': ['£1,234,567', 'not_a_number', '£0']
    })
    is_valid, results = validator._validate_column(df, 'Jackpot', {'type': 'string'}, True)
    assert len(results['fixes_applied']) > 0

def test_validate_prediction_edge_cases():
    """Test validation of edge case predictions."""
    # None
    is_valid, error_msg = validate_prediction(None)
    assert not is_valid
    assert 'None' in error_msg
    
    # Empty list
    is_valid, error_msg = validate_prediction([])
    assert not is_valid
    assert 'empty' in error_msg.lower()
    
    # Not a list
    is_valid, error_msg = validate_prediction(123)
    assert not is_valid
    assert 'list' in error_msg.lower()

if __name__ == "__main__":
    pytest.main([__file__]) 