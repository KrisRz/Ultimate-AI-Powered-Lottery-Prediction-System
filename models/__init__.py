"""Models package for lottery prediction."""

from .base import BaseModel, TimeSeriesModel, EnsembleModel
from .lstm_model import LSTMModel
from .cnn_lstm_model import CNNLSTMModel
from .meta_model import MetaModel
from .autoencoder_model import AutoencoderModel
from .arima_model import ARIMAModel
from .holtwinters_model import HoltWintersModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel
from .gradient_boosting_model import GradientBoostingModel
from .knn_model import KNNModel
from .linear_model import LinearModel
from .compatibility import *

__all__ = [
    'BaseModel',
    'TimeSeriesModel',
    'EnsembleModel',
    'LSTMModel',
    'CNNLSTMModel',
    'MetaModel',
    'AutoencoderModel',
    'ARIMAModel',
    'HoltWintersModel',
    'LightGBMModel',
    'XGBoostModel',
    'CatBoostModel',
    'GradientBoostingModel',
    'KNNModel',
    'LinearModel',
    # Compatibility functions
    'ensure_valid_prediction', 'import_prediction_function',
    'predict_with_model', 'score_combinations',
    'ensemble_prediction', 'monte_carlo_simulation',
    'save_predictions', 'predict_next_draw',
    'calculate_prediction_metrics', 'calculate_metrics',
    'backtest', 'get_model_weights'
] 