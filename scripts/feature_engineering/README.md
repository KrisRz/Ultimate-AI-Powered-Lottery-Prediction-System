# Feature Engineering Module

This module provides comprehensive feature engineering functionality for the lottery prediction system.

## Available Functionality

### Core Feature Engineering (`feature_engineering.py`)

- `enhanced_time_series_features`: Generate comprehensive features for time series analysis
- `prepare_lstm_features`: Create sequence-based features specifically for LSTM models
- `expand_draw_sequences`: Include previous draws as features
- `create_temporal_features`: Generate date/time-based features
- `create_number_features`: Extract statistical properties from lottery numbers
- `create_frequency_features`: Analyze frequency patterns in lottery draws
- `create_lag_features`: Generate lagged versions of features
- `create_combination_features`: Extract combination patterns from lottery draws

### Feature Visualization (`feature_visualization.py`)

- `visualize_features`: Create visualizations of feature distributions, correlations, and importance

## Integration with Main System

To use these features in the lottery prediction system:

```python
# Basic usage
from scripts.feature_engineering import enhanced_time_series_features

# Generate enhanced features
enhanced_df = enhanced_time_series_features(lottery_data, look_back=100)

# Prepare data for LSTM models
from scripts.feature_engineering import prepare_lstm_features
X, y, scaler = prepare_lstm_features(lottery_data, look_back=50)

# Visualize features
from scripts.feature_engineering import visualize_features
visualize_features(X, y, feature_names)
```

## Development Guidelines

When extending this module:

1. Implement new functionality in the appropriate file
2. Update the `__init__.py` file to expose the new functionality
3. Add tests to ensure the functionality works as expected
4. Document the new functionality in this README

When using existing functionality:

1. Import from `scripts.feature_engineering` rather than from individual files
2. This ensures you're using the stable public API 