import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Attention, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pmdarima import auto_arima
from catboost import CatBoostRegressor
import optuna
from tqdm import tqdm
import pickle
import os
import datetime
import logging
from typing import Dict, Tuple, List, Any, Union, Optional
from .utils import log_training_errors, log_memory_usage, LOOK_BACK, EPOCHS
from .base import TimeSeriesModel, BaseModel

def prepare_data(df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
    """
    Prepare appropriate data structures for different model types
    
    Args:
        df: DataFrame with lottery data
        model_type: One of 'time_series', 'feature_based', 'single_time_series', 'autoencoder'
        
    Returns:
        Dictionary with prepared data for the specified model type
    """
    # Extract core data from DataFrame
    all_numbers = df['Main_Numbers'].tolist()
    data = {}
    
    # Prepare data for different model types
    if model_type == 'time_series':
        # For LSTM, CNN-LSTM - sequence with LOOK_BACK windows
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(np.array([num for draw in all_numbers for num in draw]).reshape(-1, 1))
        X, y = [], []
        for i in range(len(series_scaled) - LOOK_BACK - 6):
            X.append(series_scaled[i:i + LOOK_BACK])
            y.append(series_scaled[i + LOOK_BACK:i + LOOK_BACK + 6])
        
        data['X'] = np.array(X)
        data['y'] = np.array(y)
        data['scaler'] = scaler
        
    elif model_type == 'single_time_series':
        # For ARIMA, Holt-Winters - separate time series for each position
        position_series = []
        for pos in range(6):
            series = [draw[pos] for draw in all_numbers]
            position_series.append(series)
        
        data['position_series'] = position_series
        
    elif model_type == 'feature_based':
        # For ML models - engineered features
        X = pd.DataFrame({
            'index': np.arange(len(all_numbers)),
            'dayofweek': pd.to_numeric(df['DayOfWeek'], errors='coerce').fillna(0),
            'sum': df['Sum'],
            'mean': df['Mean'],
            'unique': df['Unique'],
            'zscore_sum': df['ZScore_Sum'],
            'primes': df['Primes'],
            'odds': df['Odds'],
            'gaps': df['Gaps'],
            'freq_10': df['Freq_10'],
            'freq_20': df['Freq_20'],
            'freq_50': df['Freq_50'],
            'pair_freq': df['Pair_Freq'],
            'triplet_freq': df['Triplet_Freq']
        })
        
        # Prepare target variables for each number position
        y_positions = []
        for pos in range(6):
            y_positions.append(np.array([draw[pos] for draw in all_numbers]))
        
        data['X'] = X
        data['y_positions'] = y_positions
        data['X_array'] = X.values
    
    elif model_type == 'autoencoder':
        # For autoencoder - sequence data with prediction targets
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(np.array([num for draw in all_numbers for num in draw]).reshape(-1, 1))
        
        # For sequence reconstruction (original autoencoder)
        X_recon = []
        for i in range(len(series_scaled) - LOOK_BACK):
            X_recon.append(series_scaled[i:i + LOOK_BACK])
        
        # For predicting next draw (predictive autoencoder)
        X_pred, y_pred = [], []
        for i in range(len(all_numbers) - 1):
            X_pred.append(np.array(all_numbers[i]).flatten())
            y_pred.append(np.array(all_numbers[i+1]))
        
        data['X_recon'] = np.array(X_recon)
        data['X_pred'] = np.array(X_pred)
        data['y_pred'] = np.array(y_pred)
        data['scaler'] = scaler
    
    # Common data
    data['all_numbers'] = all_numbers
    data['df'] = df
    
    return data

def objective_lstm(trial, X, y):
    units1 = trial.suggest_int('units1', 32, 256)
    units2 = trial.suggest_int('units2', 16, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    inputs = Input(shape=(LOOK_BACK, 1))
    lstm1 = LSTM(units1, return_sequences=True, activation="tanh")(inputs)
    dropout1 = Dropout(dropout)(lstm1)
    attention = Attention()([dropout1, dropout1])
    lstm2 = LSTM(units2, return_sequences=False, activation="tanh")(attention)
    outputs = Dense(6, activation="linear")(lstm2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model.evaluate(X, y, verbose=0)

@log_training_errors
def train_lstm_models(df: pd.DataFrame) -> Tuple[Model, MinMaxScaler]:
    log_memory_usage()
    
    # Use standardized data preparation
    data = prepare_data(df, 'time_series')
    X, y, scaler = data['X'], data['y'], data['scaler']
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_lstm(trial, X, y), n_trials=20)
    best_params = study.best_params
    
    inputs = Input(shape=(LOOK_BACK, 1))
    lstm1 = LSTM(best_params['units1'], return_sequences=True, activation="tanh")(inputs)
    dropout1 = Dropout(best_params['dropout'])(lstm1)
    attention = Attention()([dropout1, dropout1])
    lstm2 = LSTM(best_params['units2'], return_sequences=False, activation="tanh")(attention)
    outputs = Dense(6, activation="linear")(lstm2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    
    for epoch in tqdm(range(EPOCHS), desc="Training LSTM seq2seq"):
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        log_memory_usage()
    
    return model, scaler

@log_training_errors
def train_arima_models(df: pd.DataFrame) -> List[ARIMA]:
    log_memory_usage()
    
    # Use standardized data preparation
    data = prepare_data(df, 'single_time_series')
    position_series = data['position_series']
    
    models = []
    for pos in tqdm(range(6), desc="Training ARIMA models"):
        series = position_series[pos]
        try:
            model = auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, d=1,
                               seasonal=False, trace=False, error_action='warn', 
                               suppress_warnings=True, stepwise=True)
            models.append(model)
        except Exception as e:
            models.append(None)
    log_memory_usage()
    return models

@log_training_errors
def train_holtwinters_models(df: pd.DataFrame) -> List[ExponentialSmoothing]:
    log_memory_usage()
    
    # Use standardized data preparation
    data = prepare_data(df, 'single_time_series')
    position_series = data['position_series']
    
    models = []
    for pos in tqdm(range(6), desc="Training Holt-Winters models"):
        series = position_series[pos]
        model = ExponentialSmoothing(series, trend="add").fit()
        models.append(model)
    log_memory_usage()
    return models

@log_training_errors
def train_linear_models(df: pd.DataFrame) -> List[LinearRegression]:
    log_memory_usage()
    
    # Use standardized data preparation
    data = prepare_data(df, 'feature_based')
    X, y_positions = data['X'], data['y_positions']
    
    models = []
    for pos in tqdm(range(6), desc="Training Linear Regression models"):
        y = y_positions[pos]
        model = LinearRegression()
        model.fit(X, y)
        models.append(model)
    log_memory_usage()
    return models, StandardScaler().fit(X)

@log_training_errors
def train_xgboost_models(df: pd.DataFrame) -> List[XGBRegressor]:
    log_memory_usage()
    
    # Use standardized data preparation
    data = prepare_data(df, 'feature_based')
    X, y_positions = data['X'], data['y_positions']
    
    models = []
    for pos in tqdm(range(6), desc="Training XGBoost models"):
        y = y_positions[pos]
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X, y)
        models.append(model)
    log_memory_usage()
    return models, StandardScaler().fit(X)

@log_training_errors
def train_lightgbm_models(df: pd.DataFrame) -> List[LGBMRegressor]:
    log_memory_usage()
    all_numbers = df['Main_Numbers'].tolist()
    X = pd.DataFrame({
        'index': np.arange(len(all_numbers)),
        'dayofweek': pd.to_numeric(df['DayOfWeek'], errors='coerce').fillna(0),
        'sum': df['Sum'],
        'mean': df['Mean'],
        'unique': df['Unique'],
        'zscore_sum': df['ZScore_Sum'],
        'primes': df['Primes'],
        'odds': df['Odds'],
        'gaps': df['Gaps'],
        'freq_10': df['Freq_10'],
        'freq_20': df['Freq_20'],
        'freq_50': df['Freq_50'],
        'pair_freq': df['Pair_Freq'],
        'triplet_freq': df['Triplet_Freq']
    })
    models = []
    for pos in tqdm(range(6), desc="Training LightGBM models"):
        y = np.array([draw[pos] for draw in all_numbers])
        model = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X, y)
        models.append(model)
    log_memory_usage()
    return models

@log_training_errors
def train_knn_models(df: pd.DataFrame) -> List[KNeighborsRegressor]:
    log_memory_usage()
    all_numbers = df['Main_Numbers'].tolist()
    X = pd.DataFrame({
        'index': np.arange(len(all_numbers)),
        'dayofweek': pd.to_numeric(df['DayOfWeek'], errors='coerce').fillna(0),
        'sum': df['Sum'],
        'mean': df['Mean'],
        'unique': df['Unique'],
        'zscore_sum': df['ZScore_Sum'],
        'primes': df['Primes'],
        'odds': df['Odds'],
        'gaps': df['Gaps'],
        'freq_10': df['Freq_10'],
        'freq_20': df['Freq_20'],
        'freq_50': df['Freq_50'],
        'pair_freq': df['Pair_Freq'],
        'triplet_freq': df['Triplet_Freq']
    })
    models = []
    for pos in tqdm(range(6), desc="Training KNN models"):
        y = np.array([draw[pos] for draw in all_numbers])
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, y)
        models.append(model)
    log_memory_usage()
    return models

@log_training_errors
def train_gradientboosting_models(df: pd.DataFrame) -> List[GradientBoostingRegressor]:
    log_memory_usage()
    all_numbers = df['Main_Numbers'].tolist()
    X = pd.DataFrame({
        'index': np.arange(len(all_numbers)),
        'dayofweek': pd.to_numeric(df['DayOfWeek'], errors='coerce').fillna(0),
        'sum': df['Sum'],
        'mean': df['Mean'],
        'unique': df['Unique'],
        'zscore_sum': df['ZScore_Sum'],
        'primes': df['Primes'],
        'odds': df['Odds'],
        'gaps': df['Gaps'],
        'freq_10': df['Freq_10'],
        'freq_20': df['Freq_20'],
        'freq_50': df['Freq_50'],
        'pair_freq': df['Pair_Freq'],
        'triplet_freq': df['Triplet_Freq']
    })
    models = []
    for pos in tqdm(range(6), desc="Training Gradient Boosting models"):
        y = np.array([draw[pos] for draw in all_numbers])
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X, y)
        models.append(model)
    log_memory_usage()
    return models

@log_training_errors
def train_catboost_models(df: pd.DataFrame) -> List[CatBoostRegressor]:
    log_memory_usage()
    all_numbers = df['Main_Numbers'].tolist()
    X = pd.DataFrame({
        'index': np.arange(len(all_numbers)),
        'dayofweek': pd.to_numeric(df['DayOfWeek'], errors='coerce').fillna(0),
        'sum': df['Sum'],
        'mean': df['Mean'],
        'unique': df['Unique'],
        'zscore_sum': df['ZScore_Sum'],
        'primes': df['Primes'],
        'odds': df['Odds'],
        'gaps': df['Gaps'],
        'freq_10': df['Freq_10'],
        'freq_20': df['Freq_20'],
        'freq_50': df['Freq_50'],
        'pair_freq': df['Pair_Freq'],
        'triplet_freq': df['Triplet_Freq']
    })
    models = []
    for pos in tqdm(range(6), desc="Training CatBoost models"):
        y = np.array([draw[pos] for draw in all_numbers])
        model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, verbose=False)
        model.fit(X, y)
        models.append(model)
    log_memory_usage()
    return models

@log_training_errors
def train_cnn_lstm_models(df: pd.DataFrame) -> Tuple[Model, MinMaxScaler]:
    log_memory_usage()
    all_numbers = df['Main_Numbers'].tolist()
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(np.array([num for draw in all_numbers for num in draw]).reshape(-1, 1))
    X, y = [], []
    for i in range(len(series_scaled) - LOOK_BACK - 6):
        X.append(series_scaled[i:i + LOOK_BACK])
        y.append(series_scaled[i + LOOK_BACK:i + LOOK_BACK + 6])
    X, y = np.array(X), np.array(y)
    
    inputs = Input(shape=(LOOK_BACK, 1))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    lstm1 = LSTM(128, return_sequences=True)(pool1)
    lstm2 = LSTM(64)(lstm1)
    outputs = Dense(6, activation="linear")(lstm2)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    
    for epoch in tqdm(range(EPOCHS), desc="Training CNN-LSTM"):
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        log_memory_usage()
    
    return model, scaler

@log_training_errors
def train_autoencoder(df: pd.DataFrame) -> Tuple[Model, MinMaxScaler]:
    log_memory_usage()
    
    # Use standardized data preparation for predictive autoencoder
    data = prepare_data(df, 'autoencoder')
    X_pred, y_pred, scaler = data['X_pred'], data['y_pred'], data['scaler']
    
    # Scale input and output for prediction
    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X_pred)
    
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y_pred)
    
    # Build predictive autoencoder model
    input_dim = X_scaled.shape[1]
    inputs = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Latent representation
    latent = Dense(16, activation='relu')(encoded)
    
    # Decoder for next draw prediction 
    decoded = Dense(32, activation='relu')(latent)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    outputs = Dense(6, activation='linear')(decoded)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    
    for epoch in tqdm(range(EPOCHS), desc="Training Predictive Autoencoder"):
        model.fit(
            X_scaled, y_scaled,
            epochs=1, 
            batch_size=32, 
            verbose=0,
            validation_split=0.2
        )
        log_memory_usage()
    
    return model, (X_scaler, y_scaler)

@log_training_errors
def train_meta_model(df: pd.DataFrame, preds_dict: Dict[str, List[List[int]]]) -> LinearRegression:
    log_memory_usage()
    X_meta = []
    for model_name, preds in preds_dict.items():
        for pred in preds:
            X_meta.append(pred)
    X_meta = np.array(X_meta)
    
    # Use the last 6 numbers as target
    y_meta = np.array([draw[-6:] for draw in df['Main_Numbers']])
    
    model = LinearRegression()
    model.fit(X_meta, y_meta)
    log_memory_usage()
    return model

@log_training_errors
def update_models(df: pd.DataFrame, retrain_window: int = None) -> Dict:
    """
    Update all models with new data.
    
    Args:
        df: DataFrame with lottery data
        retrain_window: If specified, only retrain if new data exceeds this number of rows
        
    Returns:
        Dictionary of trained models
    """
    log_memory_usage()
    
    # Check if we should retrain based on new data
    if retrain_window is not None:
        try:
            # Try to load existing models and their metadata
            if os.path.exists('models/trained_models.pkl'):
                with open('models/trained_models.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                    
                if 'metadata' in saved_data:
                    metadata = saved_data['metadata']
                    last_train_size = metadata.get('train_size', 0)
                    last_train_date = metadata.get('train_date')
                    
                    # Only retrain if we have enough new data or no last training date
                    if last_train_size and len(df) <= last_train_size + retrain_window:
                        logging.info(f"Not enough new data for retraining. Using existing models from {last_train_date}")
                        return saved_data['models']
        except Exception as e:
            logging.warning(f"Error checking retraining condition: {str(e)}. Proceeding with training.")
    
    models = {}
    
    # Train individual models
    models['lstm'] = train_lstm_models(df)
    models['arima'] = train_arima_models(df)
    models['holtwinters'] = train_holtwinters_models(df)
    models['linear'] = train_linear_models(df)
    models['xgboost'] = train_xgboost_models(df)
    models['lightgbm'] = train_lightgbm_models(df)
    models['knn'] = train_knn_models(df)
    models['gradientboosting'] = train_gradientboosting_models(df)
    models['catboost'] = train_catboost_models(df)
    models['cnn_lstm'] = train_cnn_lstm_models(df)
    models['autoencoder'] = train_autoencoder(df)
    
    # Generate predictions for meta model
    preds_dict = {}
    for model_name, model in models.items():
        if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
            preds = [model[0].predict(df) for _ in range(10)]
        else:
            preds = [model.predict(df) for _ in range(10)]
        preds_dict[model_name] = preds
    
    # Train meta model
    models['meta'] = train_meta_model(df, preds_dict)
    
    # Save models with metadata
    try:
        metadata = {
            'train_size': len(df),
            'train_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_count': len(models)
        }
        
        with open('models/trained_models.pkl', 'wb') as f:
            pickle.dump({
                'models': models,
                'metadata': metadata
            }, f)
            
        logging.info(f"Saved {len(models)} trained models with metadata")
    except Exception as e:
        logging.error(f"Error saving models: {str(e)}")
    
    log_memory_usage()
    return models 