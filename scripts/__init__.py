from .predict_numbers import (
    predict_next_draw,
    ensemble_prediction,
    predict_with_model,
    monte_carlo_simulation,
    backtest,
    get_model_weights,
    save_predictions,
    calculate_prediction_metrics,
    score_combinations,
    ensure_valid_prediction,
    import_prediction_function
)

from .train_models import (
    train_model,
    validate_model_predictions,
    train_all_models,
    load_trained_models
)

from .fetch_data import (
    load_data,
    prepare_training_data,
    split_data
)

from .analyze_data import (
    analyze_lottery_data
)

__all__ = [
    'predict_next_draw',
    'ensemble_prediction',
    'predict_with_model',
    'monte_carlo_simulation',
    'backtest',
    'get_model_weights',
    'save_predictions',
    'calculate_prediction_metrics',
    'score_combinations',
    'ensure_valid_prediction',
    'import_prediction_function',
    'train_model',
    'validate_model_predictions',
    'train_all_models',
    'load_trained_models',
    'load_data',
    'prepare_training_data', 
    'split_data',
    'analyze_lottery_data'
]

# Initialize scripts package