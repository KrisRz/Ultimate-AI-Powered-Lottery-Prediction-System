"""
Functions for training various prediction models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
import optuna
from typing import Dict, Any, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import psutil
import os

from scripts.analyze_data import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = "data/lottery_data_1995_2025.csv"
MODELS_DIR = Path("models")
RETRAIN_WINDOW = 52  # weeks

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series models."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def objective_lstm(trial):
    """Optimize LSTM hyperparameters."""
    try:
        units1 = trial.suggest_int("units1", 32, 128)
        units2 = trial.suggest_int("units2", 16, 64)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
        
        model = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(10, 1)),
            Dropout(dropout_rate),
            LSTM(units2),
            Dropout(dropout_rate),
            Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    except Exception as e:
        logger.error(f"Error in LSTM objective: {e}")
        raise

def train_lstm(data: np.ndarray, n_trials: int = 100) -> Tuple[Sequential, Dict]:
    """Train LSTM model with hyperparameter optimization."""
    try:
        X, y = create_sequences(data, 10)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_lstm, n_trials=n_trials)
        
        best_model = objective_lstm(study.best_trial)
        best_model.fit(X, y, epochs=100, batch_size=32, verbose=0)
        
        return best_model, study.best_params
    except Exception as e:
        logger.error(f"Error training LSTM: {e}")
        raise

def train_arima(data: np.ndarray) -> ARIMA:
    """Train ARIMA model."""
    try:
        model = ARIMA(data, order=(5,1,0))
        return model.fit()
    except Exception as e:
        logger.error(f"Error training ARIMA: {e}")
        raise

def train_holtwinters(data: np.ndarray) -> ExponentialSmoothing:
    """Train Holt-Winters model."""
    try:
        model = ExponentialSmoothing(data, seasonal_periods=52, trend='add', seasonal='add')
        return model.fit()
    except Exception as e:
        logger.error(f"Error training Holt-Winters: {e}")
        raise

def train_all_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train all prediction models."""
    logger.info("Starting model training...")
    log_memory_usage()
    
    models = {
        'scalers': {},
        'models': {},
        'params': {}
    }
    
    try:
        number_cols = [col for col in df.columns if col.startswith('Number')]
        
        for col in number_cols:
            logger.info(f"Training models for {col}")
            data = df[col].values
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data.reshape(-1, 1)).ravel()
            models['scalers'][col] = scaler
            
            # Train models
            lstm_model, lstm_params = train_lstm(scaled_data)
            models['models'][f'{col}_lstm'] = lstm_model
            models['params'][f'{col}_lstm'] = lstm_params
            
            arima_model = train_arima(data)
            models['models'][f'{col}_arima'] = arima_model
            
            hw_model = train_holtwinters(data)
            models['models'][f'{col}_hw'] = hw_model
            
            # Train ML models
            X, y = create_sequences(scaled_data, 10)
            
            xgb = XGBRegressor()
            xgb.fit(X.reshape(X.shape[0], -1), y)
            models['models'][f'{col}_xgb'] = xgb
            
            lgb = LGBMRegressor()
            lgb.fit(X.reshape(X.shape[0], -1), y)
            models['models'][f'{col}_lgb'] = lgb
            
            cat = CatBoostRegressor(verbose=False)
            cat.fit(X.reshape(X.shape[0], -1), y)
            models['models'][f'{col}_cat'] = cat
            
            gb = GradientBoostingRegressor()
            gb.fit(X.reshape(X.shape[0], -1), y)
            models['models'][f'{col}_gb'] = gb
            
            knn = KNeighborsRegressor()
            knn.fit(X.reshape(X.shape[0], -1), y)
            models['models'][f'{col}_knn'] = knn
            
            log_memory_usage()
            
        logger.info("Model training completed successfully")
        return models
        
    except Exception as e:
        logger.error(f"Error in train_all_models: {e}")
        raise

def update_models(force_retrain: bool = False) -> None:
    """Update models if necessary."""
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        
        df = load_data(Path(DATA_FILE))
        
        model_file = MODELS_DIR / "trained_models.joblib"
        if not force_retrain and model_file.exists():
            last_modified = model_file.stat().st_mtime
            last_data = df['Date'].max().timestamp()
            
            if last_modified > last_data:
                logger.info("Models are up to date")
                return
        
        logger.info("Training new models...")
        models = train_all_models(df)
        joblib.dump(models, model_file)
        logger.info("Models updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating models: {e}")
        raise

if __name__ == "__main__":
    update_models() 