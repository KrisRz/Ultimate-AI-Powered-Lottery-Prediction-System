import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
import time
import pickle
import concurrent.futures
import psutil
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import xgboost as xgb
import joblib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import cpu_count

# Import model training functions
from .lstm_model import train_lstm_model
from .arima_model import train_arima_model
from .holtwinters_model import train_holtwinters_model
from .linear_model import train_linear_models
from .xgboost_model import train_xgboost_model
from .lightgbm_model import train_lightgbm_model
from .knn_model import train_knn_model
from .gradient_boosting_model import train_gradient_boosting_model
from .catboost_model import train_catboost_model
from .cnn_lstm_model import train_cnn_lstm_model
from .autoencoder_model import train_autoencoder_model
from .meta_model import train_meta_model

# Import utils
from .utils import log_training_errors, log_memory_usage

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MODELS_CACHE_DIR = Path("models/cache")
MODELS_PATH = Path("models/trained_models.pkl")
MODEL_CACHE_EXPIRY = 7  # days

def memory_profiled(func):
    """Decorator to profile memory usage of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        duration = end_time - start_time
        mem_used = mem_after - mem_before
        
        logger.info(f"{func.__name__} completed in {duration:.2f}s, memory delta: {mem_used:.2f}MB")
        return result
    return wrapper

def create_model_cache_key(df: pd.DataFrame, model_name: str) -> str:
    """Create a cache key for a model based on data properties."""
    # Use data properties that would invalidate a model when changed
    data_hash = pd.util.hash_pandas_object(df.iloc[-10:]).sum()  # Use recent data for hash
    return f"{model_name}_{len(df)}_{data_hash}"

def save_model_to_cache(model: Any, model_name: str, cache_key: str) -> bool:
    """Save a model to the cache directory."""
    try:
        MODELS_CACHE_DIR.mkdir(exist_ok=True, parents=True)
        cache_file = MODELS_CACHE_DIR / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'model': model,
                'timestamp': time.time(),
                'model_name': model_name
            }, f)
        logger.info(f"Saved {model_name} model to cache with key {cache_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving model to cache: {e}")
        return False

def load_model_from_cache(model_name: str, cache_key: str) -> Optional[Any]:
    """Load a model from the cache directory if it exists and is not expired."""
    try:
        cache_file = MODELS_CACHE_DIR / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is expired
            cache_age = (time.time() - cached_data['timestamp']) / (60 * 60 * 24)  # days
            if cache_age > MODEL_CACHE_EXPIRY:
                logger.info(f"Cache for {model_name} is expired ({cache_age:.1f} days old)")
                return None
                
            logger.info(f"Loaded {model_name} model from cache ({cache_age:.1f} days old)")
            return cached_data['model']
        return None
    except Exception as e:
        logger.error(f"Error loading model from cache: {e}")
        return None

def prepare_data_optimized(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare data for model training more efficiently."""
    data = {}
    
    # Extract core data once (avoid multiple extractions)
    all_numbers = np.array(df['Main_Numbers'].tolist())
    data['all_numbers'] = all_numbers
    
    # Flatten all numbers for time series models
    flattened = np.array([num for draw in df['Main_Numbers'] for num in draw]).reshape(-1, 1)
    data['flattened'] = flattened
    
    # Prepare time series data
    lstm_scaler = MinMaxScaler()
    series_scaled = lstm_scaler.fit_transform(flattened)
    data['lstm_scaler'] = lstm_scaler
    data['series_scaled'] = series_scaled
    
    # Time series sequences (for LSTM, CNN-LSTM)
    look_back = 200  # from .utils import LOOK_BACK
    X_ts, y_ts = [], []
    for i in range(len(series_scaled) - look_back - 6):
        X_ts.append(series_scaled[i:i + look_back])
        y_ts.append(series_scaled[i + look_back:i + look_back + 6])
    data['X_ts'] = np.array(X_ts)
    data['y_ts'] = np.array(y_ts)
    
    # Feature data for ML models
    # Pre-compute features for faster access
    features = {
        'index': np.arange(len(all_numbers)),
        'dayofweek': pd.to_numeric(df['DayOfWeek'], errors='coerce').fillna(0),
        'sum': df['Sum'].values,
        'mean': df['Mean'].values,
        'unique': df['Unique'].values,
        'zscore_sum': df['ZScore_Sum'].values,
        'primes': df['Primes'].values,
        'odds': df['Odds'].values,
        'gaps': df['Gaps'].values,
        'freq_10': df['Freq_10'].values,
        'freq_20': df['Freq_20'].values,
        'freq_50': df['Freq_50'].values,
        'pair_freq': df['Pair_Freq'].values,
        'triplet_freq': df['Triplet_Freq'].values
    }
    
    # Convert to numpy arrays once
    X_feat = np.column_stack([features[col] for col in features])
    data['X_feat'] = X_feat
    data['features'] = features
    
    # Target variables for each position
    y_positions = []
    for pos in range(6):
        y_positions.append(np.array([draw[pos] for draw in all_numbers]))
    data['y_positions'] = y_positions
    
    # Split into train/validation for early stopping
    X_train, X_val, y_train, y_val = {}, {}, {}, {}
    
    # Time series data split
    train_idx, val_idx = train_test_split(np.arange(len(data['X_ts'])), test_size=0.2, random_state=42)
    X_train['ts'] = data['X_ts'][train_idx]
    X_val['ts'] = data['X_ts'][val_idx]
    y_train['ts'] = data['y_ts'][train_idx]
    y_val['ts'] = data['y_ts'][val_idx]
    
    # Feature data split
    train_idx, val_idx = train_test_split(np.arange(len(X_feat)), test_size=0.2, random_state=42)
    X_train['feat'] = X_feat[train_idx]
    X_val['feat'] = X_feat[val_idx]
    
    # Position-specific targets
    y_train['pos'] = []
    y_val['pos'] = []
    for pos in range(6):
        pos_arr = data['y_positions'][pos]
        y_train['pos'].append(pos_arr[train_idx])
        y_val['pos'].append(pos_arr[val_idx])
    
    data['X_train'] = X_train
    data['X_val'] = X_val
    data['y_train'] = y_train
    data['y_val'] = y_val
    
    return data

def detect_and_configure_gpu():
    """Detect and configure GPU usage for TensorFlow."""
    try:
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth - prevents TF from allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available: {len(gpus)} device(s)")
                
                # Log GPU details
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.name if hasattr(gpu, 'name') else gpu}")
                
                return True
            except RuntimeError as e:
                logger.warning(f"Error configuring GPU: {e}")
        
        logger.info("No GPU detected, using CPU")
        return False
    except Exception as e:
        logger.warning(f"Error during GPU detection: {e}")
        return False

# Check for GPU at module import time
HAS_GPU = detect_and_configure_gpu()

@memory_profiled
def train_lstm_optimized(data: Dict[str, Any]) -> Tuple[Any, Any]:
    """Optimized LSTM model training with early stopping."""
    X_train = data['X_train']['ts']
    y_train = data['y_train']['ts']
    X_val = data['X_val']['ts']
    y_val = data['y_val']['ts']
    scaler = data['lstm_scaler']
    
    # Create model with improved architecture
    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(6)
    ])
    
    # Use optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    # Add callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Log whether we're using GPU
    logger.info(f"Training LSTM model {'with GPU acceleration' if HAS_GPU else 'on CPU'}")
    
    # Train with validation
    history = model.fit(
        X_train, y_train,
        epochs=100,  # More epochs but with early stopping
        batch_size=64 if HAS_GPU else 32,  # Larger batch for GPU
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, scaler

@memory_profiled
def train_xgboost_optimized(data: Dict[str, Any]) -> Tuple[List[Any], Any]:
    """Optimized XGBoost training for each position."""
    X_train = data['X_train']['feat']
    X_val = data['X_val']['feat']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train a model for each position
    models = []
    
    for pos in range(6):
        y_train_pos = data['y_train']['pos'][pos]
        y_val_pos = data['y_val']['pos'][pos]
        
        # Create model with optimized parameters
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',  # faster algorithm
            random_state=42
        )
        
        # Train with early stopping
        model.fit(
            X_train_scaled, y_train_pos,
            eval_set=[(X_val_scaled, y_val_pos)],
            early_stopping_rounds=20,
            verbose=0
        )
        
        models.append(model)
    
    return models, scaler

def train_model_with_fallback(train_func, data, model_name, cache_key):
    """Train a model with caching and fallback to simpler model on failure."""
    # Try to load from cache first
    cached_model = load_model_from_cache(model_name, cache_key)
    if cached_model is not None:
        return cached_model
    
    try:
        # Try training the model
        start_time = time.time()
        logger.info(f"Training {model_name} model...")
        model = train_func(data)
        duration = time.time() - start_time
        logger.info(f"Successfully trained {model_name} model in {duration:.2f} seconds")
        
        # Cache successful model
        save_model_to_cache(model, model_name, cache_key)
        return model
    except Exception as e:
        logger.error(f"Error training {model_name} model: {e}")
        
        # Return fallback/dummy model
        logger.info(f"Using fallback model for {model_name}")
        return None

@memory_profiled
def train_all_models_parallel(df: pd.DataFrame, max_workers: int = None) -> Dict[str, Any]:
    """Train all models in parallel with optimized data preparation."""
    if max_workers is None:
        # Use half of available CPUs for parallel training
        max_workers = max(1, psutil.cpu_count(logical=False) // 2)
    
    logger.info(f"Preparing data for model training...")
    data = prepare_data_optimized(df)
    logger.info(f"Data preparation complete. Training with {max_workers} workers.")
    
    # Define model training tasks
    model_tasks = {
        'lstm': (lambda: train_lstm_optimized(data), 'lstm_model'),
        'xgboost': (lambda: train_xgboost_optimized(data), 'xgboost_model'),
        'lightgbm': (lambda: train_lightgbm_model(data['X_feat'], data['y_positions']), 'lightgbm_model'),
        'catboost': (lambda: train_catboost_model(data['X_feat'], data['y_positions']), 'catboost_model'),
        'knn': (lambda: train_knn_model(data['X_feat'], data['y_positions']), 'knn_model'),
        'linear': (lambda: train_linear_models(data['X_feat'], data['y_positions']), 'linear_model'),
        'gradient_boosting': (lambda: train_gradient_boosting_model(data['X_feat'], data['y_positions']), 'gradient_boosting_model'),
        'arima': (lambda: train_arima_optimized(data), 'arima_model'),
        'holtwinters': (lambda: train_holtwinters_model(data['all_numbers']), 'holtwinters_model'),
        'cnn_lstm': (lambda: train_cnn_lstm_optimized(data), 'cnn_lstm_model'),
        'autoencoder': (lambda: train_autoencoder_model(data['X_ts'], data['y_ts']), 'autoencoder_model')
    }
    
    # Create cache keys for each model
    cache_keys = {name: create_model_cache_key(df, name) for name in model_tasks}
    
    # Train models in parallel
    start_time = time.time()
    models = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                train_model_with_fallback, 
                task[0], data, name, cache_keys[name]
            ): name 
            for name, task in model_tasks.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                model = future.result()
                if model is not None:
                    models[model_name] = model
                    logger.info(f"Successfully trained {model_name} model")
                else:
                    logger.warning(f"Failed to train {model_name} model")
            except Exception as e:
                logger.error(f"Exception training {model_name} model: {e}")
    
    # Train meta model after all individual models are trained
    if len(models) >= 3:
        logger.info(f"Training meta model with {len(models)} base models")
        try:
            # Create meta model training data
            meta_preds = []
            for model_name, model in models.items():
                # Generate predictions from each base model
                if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
                    # Deep learning models
                    pred = model[0].predict(data['X_ts'][-1:])
                    meta_preds.append(pred.flatten())
                else:
                    # Traditional ML models 
                    for i in range(6):
                        if model_name in ['arima', 'holtwinters']:
                            # Time series models with direct forecasting
                            pred = model[i].forecast(1)[0] if model[i] is not None else 30
                        else:
                            # Feature-based models
                            pred = model[i].predict(data['X_feat'][-1:])[0] if model[i] is not None else 30
                        meta_preds.append(pred)
            
            # Train meta model
            meta_X = np.array(meta_preds).reshape(1, -1)
            meta_y = np.array(data['all_numbers'][-1:])
            meta_model = train_meta_model(meta_X, meta_y)
            models['meta'] = meta_model
            
        except Exception as e:
            logger.error(f"Error training meta model: {e}")
    
    duration = time.time() - start_time
    logger.info(f"Trained {len(models)} models in {duration:.2f} seconds")
    
    # Save all models
    try:
        MODELS_PATH.parent.mkdir(exist_ok=True)
        with open(MODELS_PATH, 'wb') as f:
            pickle.dump({
                'models': models,
                'timestamp': time.time(),
                'data_size': len(df)
            }, f)
        logger.info(f"Saved {len(models)} models to {MODELS_PATH}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
    
    return models

def load_optimized_models() -> Dict[str, Any]:
    """Load trained models with optimized checks."""
    try:
        if MODELS_PATH.exists():
            modified_time = os.path.getmtime(MODELS_PATH)
            cache_age = (time.time() - modified_time) / (60 * 60 * 24)  # days
            
            with open(MODELS_PATH, 'rb') as f:
                data = pickle.load(f)
            
            models = data.get('models', {})
            timestamp = data.get('timestamp', 'unknown')
            
            logger.info(f"Loaded {len(models)} models from {MODELS_PATH} (cache age: {cache_age:.1f} days)")
            return models
        else:
            logger.warning(f"No models file found at {MODELS_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}

@memory_profiled
def train_cnn_lstm_optimized(data: Dict[str, Any]) -> Tuple[Any, Any]:
    """Optimized CNN-LSTM model training with GPU acceleration when available."""
    try:
        X_train = data['X_train']['ts']
        y_train = data['y_train']['ts']
        X_val = data['X_val']['ts']
        y_val = data['y_val']['ts']
        scaler = data['lstm_scaler']
        
        # Create model with improved architecture
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=(X_train.shape[1], X_train.shape[2])),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(100, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dropout(0.3),
            Dense(6)
        ])
        
        # Use optimizer with learning rate scheduling
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Add callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Log whether we're using GPU
        logger.info(f"Training CNN-LSTM model {'with GPU acceleration' if HAS_GPU else 'on CPU'}")
        
        # Train with validation
        history = model.fit(
            X_train, y_train,
            epochs=150,  # More epochs but with early stopping
            batch_size=32 if HAS_GPU else 16,  # Larger batch for GPU
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, scaler
    except Exception as e:
        logger.error(f"Error training CNN-LSTM model: {e}")
        raise e 

@memory_profiled
def train_prophet_optimized(data: Dict[str, Any]) -> Any:
    """Optimized Prophet model training with parallelization and resource optimization."""
    try:
        from prophet import Prophet
        import pandas as pd
        from multiprocessing import cpu_count
        
        df = data['prophet_df'].copy()
        
        # Determine optimal number of threads based on system resources
        n_cores = max(1, int(cpu_count() * 0.75))
        
        # Configure prophet with optimized parameters and parallel processing
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            mcmc_samples=0,  # Disable MCMC for speed
            interval_width=0.95
        )
        
        # Set the number of cores to use
        if n_cores > 1:
            logger.info(f"Training Prophet model with {n_cores} cores")
            with joblib.parallel_backend('threading', n_jobs=n_cores):
                model.fit(df)
        else:
            model.fit(df)
        
        return model
    except Exception as e:
        logger.error(f"Error training Prophet model: {e}")
        raise e 

@memory_profiled
def train_neuralprophet_optimized(data: Dict[str, Any]) -> Any:
    """Optimized NeuralProphet model training with parallelization for speed."""
    try:
        from neuralprophet import NeuralProphet
        import pandas as pd
        from multiprocessing import cpu_count
        
        df = data['prophet_df'].copy()
        
        # Determine optimal number of threads based on system resources
        # If GPU is available, use fewer CPU cores as the GPU will handle the heavy lifting
        n_workers = max(1, int(cpu_count() * (0.5 if HAS_GPU else 0.75)))
        logger.info(f"Training NeuralProphet model with {n_workers} worker processes")
        
        # Configure model with improved parameters
        model = NeuralProphet(
            n_forecasts=6,
            n_lags=14,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            batch_size=64 if HAS_GPU else 32,
            epochs=200,
            learning_rate=0.005,
            loss_func='MSE',
            num_hidden_layers=4,
            d_hidden=64,
            trend_reg=0.05,
            n_changepoints=20,
            seasonality_reg=0.05,
            ar_sparsity=0.1,
            n_jobs=n_workers,
        )
        
        # Train the model
        metrics = model.fit(df, freq='W')
        
        return model
    except Exception as e:
        logger.error(f"Error training NeuralProphet model: {e}")
        raise e 

@memory_profiled
def train_arima_optimized(data: Dict[str, Any], parallel: bool = True) -> Any:
    """Optimized ARIMA model training with auto-configuration and parallel processing."""
    try:
        import pmdarima as pm
        import numpy as np
        from multiprocessing import cpu_count
        
        time_series = data['time_series'].copy()
        
        # Determine optimal number of cores to use
        n_cores = max(1, int(cpu_count() * 0.75)) if parallel else 1
        
        logger.info(f"Training auto ARIMA model with {n_cores} cores")
        
        # Create optimized ARIMA model with auto parameter selection
        model = pm.auto_arima(
            time_series,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,      # Auto-determine differencing
            seasonal=True,
            m=52,        # Weekly seasonality (52 weeks in a year)
            stepwise=True,
            error_action='ignore',
            suppress_warnings=True,
            n_jobs=n_cores,
            information_criterion='aic',
            trace=True if DEBUG_MODE else False
        )
        
        # Get model summary and log the best parameters
        order = model.order
        seasonal_order = model.seasonal_order
        logger.info(f"ARIMA model selected: ({order[0]},{order[1]},{order[2]})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})")
        
        return model
    except Exception as e:
        logger.error(f"Error training ARIMA model: {e}")
        raise e 

def train_model_optimized(model_name: str, data: Dict[str, Any]) -> Tuple[Any, str]:
    """Train the specified model with optimized resource allocation."""
    logger.info(f"Training optimized model: {model_name}")
    
    # Map model names to their training functions and return value keys
    model_map = {
        'linear': (lambda: train_linear_models(data['X_feat'], data['y_positions']), 'linear_model'),
        'gradient_boosting': (lambda: train_gradient_boosting_model(data['X_feat'], data['y_positions']), 'gradient_boosting_model'),
        'arima': (lambda: train_arima_optimized(data), 'arima_model'),
        'holtwinters': (lambda: train_holtwinters_model(data['all_numbers']), 'holtwinters_model'),
        'cnn_lstm': (lambda: train_cnn_lstm_optimized(data), 'cnn_lstm_model'),
        'xgboost': (lambda: train_xgboost_optimized(data['X_feat'], data['y_positions']), 'xgboost_model'),
        'lstm': (lambda: train_lstm_model(data['sequences'], data['targets'], epochs=200), 'lstm_model'),
        'prophet': (lambda: train_prophet_optimized(data), 'prophet_model'),
        'neural_prophet': (lambda: train_neuralprophet_optimized(data), 'neural_prophet_model'),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(model_map.keys())}")
    
    # Measure training time and resource usage
    start_time = time.time()
    model = model_map[model_name][0]()
    training_time = time.time() - start_time
    
    logger.info(f"Finished training {model_name} in {training_time:.2f} seconds")
    
    return model, model_map[model_name][1]

def predict_optimized(model: Any, model_type: str, data: Dict[str, Any], **kwargs) -> List[int]:
    """Make optimized predictions using the trained model."""
    try:
        if model_type == 'arima_model':
            # Predict next n_steps
            n_steps = kwargs.get('n_steps', 1)
            forecast = model.predict(n_periods=n_steps)
            # Round to nearest integers for lottery numbers
            return [max(1, min(int(round(x)), kwargs.get('max_number', 49))) for x in forecast]
        
        elif model_type == 'prophet_model':
            # Create future dataframe
            future = model.make_future_dataframe(periods=kwargs.get('periods', 1), 
                                               freq=kwargs.get('freq', 'W'))
            # Make prediction
            forecast = model.predict(future)
            # Get the predicted values for future dates
            predictions = forecast.iloc[-kwargs.get('periods', 1):]['yhat'].values
            # Round to nearest integers for lottery numbers
            return [max(1, min(int(round(x)), kwargs.get('max_number', 49))) for x in predictions]
        
        elif model_type == 'neural_prophet_model':
            # Create future dataframe
            future = model.make_future_dataframe(df=data['prophet_df'], 
                                               periods=kwargs.get('periods', 1))
            # Make prediction
            forecast = model.predict(future)
            # Get the predicted values for future dates
            predictions = forecast.iloc[-kwargs.get('periods', 1):]['yhat1'].values
            # Round to nearest integers for lottery numbers
            return [max(1, min(int(round(x)), kwargs.get('max_number', 49))) for x in predictions]
        
        elif model_type == 'xgboost_model' or model_type == 'gradient_boosting_model':
            # Predict using test features
            X_test = data.get('X_test', kwargs.get('X_test'))
            if X_test is None:
                raise ValueError("X_test is required for prediction with tree-based models")
            
            predictions = model.predict(X_test)
            # Round to nearest integers for lottery numbers
            return [max(1, min(int(round(x)), kwargs.get('max_number', 49))) for x in predictions]
        
        elif model_type == 'lstm_model' or model_type == 'cnn_lstm_model':
            # Get the sequences for prediction
            test_sequences = data.get('test_sequences', kwargs.get('test_sequences'))
            if test_sequences is None:
                raise ValueError("test_sequences are required for LSTM prediction")
            
            # Use the appropriate prediction function
            if model_type == 'lstm_model':
                from models.lstm_model import predict_lstm_model
                predictions = predict_lstm_model(model, test_sequences, data.get('scaler'))
            else:
                predictions = model.predict(test_sequences)
            
            # Round to nearest integers for lottery numbers
            return [max(1, min(int(round(x)), kwargs.get('max_number', 49))) for x in predictions.flatten()]
        
        else:
            logger.warning(f"No specific prediction method for {model_type}, using generic predict")
            # Generic predict call for models that follow sklearn API
            X_test = data.get('X_test', kwargs.get('X_test'))
            if X_test is None:
                raise ValueError("X_test is required for prediction")
            
            predictions = model.predict(X_test)
            # Round to nearest integers for lottery numbers
            return [max(1, min(int(round(x)), kwargs.get('max_number', 49))) for x in predictions]
        
    except Exception as e:
        logger.error(f"Error making prediction with {model_type}: {e}")
        raise e

def ensemble_predict_parallel(models: Dict[str, Any], data: Dict[str, Any], 
                              weights: Optional[Dict[str, float]] = None,
                              max_number: int = 49, 
                              n_predictions: int = 6) -> List[int]:
    """Generate ensemble predictions by combining results from multiple models in parallel.
    
    Args:
        models: Dictionary mapping model_types to model objects
        data: Dictionary with prediction data for each model type
        weights: Optional dictionary of weights for each model type
        max_number: Maximum lottery number
        n_predictions: Number of lottery numbers to predict
        
    Returns:
        List of predicted lottery numbers
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction")
    
    # Default to equal weights if not specified
    if weights is None:
        weights = {model_type: 1.0/len(models) for model_type in models.keys()}
    else:
        # Normalize weights to sum to 1
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
    
    logger.info(f"Generating ensemble prediction with {len(models)} models using weights: {weights}")
    
    # Create a pool of workers for parallel prediction
    with ThreadPoolExecutor(max_workers=min(len(models), cpu_count())) as executor:
        # Submit prediction tasks for each model
        future_to_model = {
            executor.submit(predict_optimized, model, model_type, data): (model_type, weights.get(model_type, 0))
            for model_type, model in models.items()
        }
        
        # Initialize frequency counter for each possible lottery number
        number_weights = {i: 0 for i in range(1, max_number + 1)}
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_type, weight = future_to_model[future]
            try:
                predictions = future.result()
                logger.info(f"Model {model_type} predicted: {predictions}")
                
                # Add weighted votes for each predicted number
                for num in predictions:
                    if 1 <= num <= max_number:
                        number_weights[num] += weight
            except Exception as e:
                logger.error(f"Model {model_type} prediction failed: {e}")
    
    # Sort numbers by their weighted frequency
    sorted_numbers = sorted(number_weights.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top n_predictions numbers
    return [num for num, _ in sorted_numbers[:n_predictions]]

def train_multiple_models_parallel(model_names: List[str], data: Dict[str, Any]) -> Dict[str, Tuple[Any, str]]:
    """Train multiple models in parallel using optimized resource allocation.
    
    Args:
        model_names: List of model names to train
        data: Dictionary containing all data needed for model training
        
    Returns:
        Dictionary mapping model names to tuples of (model, model_type)
    """
    if not model_names:
        raise ValueError("No models specified for training")
    
    logger.info(f"Training {len(model_names)} models in parallel: {model_names}")
    
    # Check for available models
    available_models = {
        'linear', 'gradient_boosting', 'arima', 'holtwinters', 
        'cnn_lstm', 'xgboost', 'lstm', 'prophet', 'neural_prophet'
    }
    
    # Validate requested models
    invalid_models = set(model_names) - available_models
    if invalid_models:
        raise ValueError(f"Invalid model names: {invalid_models}. Available models: {available_models}")
    
    # Determine optimal number of parallel processes based on system resources
    # Avoid overloading the system - each model might use multiple threads internally
    max_workers = min(len(model_names), max(1, int(cpu_count() * 0.5)))
    
    # Reserve at least 2 cores for system if available
    if cpu_count() > 4:
        max_workers = min(max_workers, cpu_count() - 2)
    
    logger.info(f"Training with {max_workers} parallel workers")
    
    # Dictionary to store results
    results = {}
    
    # Create a ProcessPoolExecutor for parallel training
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit training tasks
        future_to_model = {
            executor.submit(train_model_optimized, model_name, data): model_name
            for model_name in model_names
        }
        
        # Process results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                model, model_type = future.result()
                results[model_name] = (model, model_type)
                logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                # Continue with other models even if one fails
    
    # Report training summary
    successful = len(results)
    logger.info(f"Training complete: {successful}/{len(model_names)} models trained successfully")
    
    return results

def run_optimized_prediction_pipeline(input_data: Dict[str, Any], 
                                models_to_train: List[str] = None,
                                model_weights: Dict[str, float] = None,
                                n_predictions: int = 6,
                                max_number: int = 49) -> List[int]:
    """Run the full optimized prediction pipeline from data to ensemble prediction.
    
    Args:
        input_data: Dictionary containing all necessary data for training and prediction
        models_to_train: List of model names to train, or None to use default set
        model_weights: Optional dictionary of weights for each model type
        n_predictions: Number of lottery numbers to predict
        max_number: Maximum lottery number
        
    Returns:
        List of predicted lottery numbers
    """
    # Default model selection if none specified
    if models_to_train is None:
        # Choose models based on available hardware
        if HAS_GPU:
            # For systems with GPU, prioritize deep learning models
            models_to_train = ['lstm', 'cnn_lstm', 'xgboost', 'neural_prophet', 'arima']
        else:
            # For CPU-only systems, focus on efficient models
            models_to_train = ['xgboost', 'arima', 'prophet', 'gradient_boosting']
    
    # Start resource monitoring
    logger.info(f"Starting optimized prediction pipeline with models: {models_to_train}")
    start_time = time.time()
    
    try:
        # Train all models in parallel
        model_results = train_multiple_models_parallel(models_to_train, input_data)
        
        # Create dictionary of trained models
        trained_models = {
            model_type: model for model_name, (model, model_type) in model_results.items()
        }
        
        # Make ensemble prediction
        predictions = ensemble_predict_parallel(
            trained_models, 
            input_data,
            weights=model_weights,
            max_number=max_number,
            n_predictions=n_predictions
        )
        
        total_time = time.time() - start_time
        logger.info(f"Optimized prediction pipeline completed in {total_time:.2f} seconds")
        logger.info(f"Predicted numbers: {predictions}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in optimized prediction pipeline: {e}")
        raise e


if __name__ == "__main__":
    # Example usage
    import sys
    from data_processing.data_loader import load_lottery_data
    from data_processing.feature_engineering import prepare_features
    
    try:
        # Load data
        data_path = sys.argv[1] if len(sys.argv) > 1 else "data/lottery_results.csv"
        raw_data = load_lottery_data(data_path)
        
        # Prepare features for all models
        prepared_data = prepare_features(raw_data)
        
        # Choose models based on availability of GPU
        if HAS_GPU:
            models = ['lstm', 'cnn_lstm', 'xgboost', 'neural_prophet']
            # Give more weight to deep learning models when GPU is available
            weights = {'lstm_model': 0.3, 'cnn_lstm_model': 0.3, 'xgboost_model': 0.2, 'neural_prophet_model': 0.2}
        else:
            models = ['xgboost', 'arima', 'prophet', 'gradient_boosting']
            # Equal weights for CPU models
            weights = None
        
        # Run the pipeline
        predictions = run_optimized_prediction_pipeline(
            prepared_data,
            models_to_train=models,
            model_weights=weights
        )
        
        print("\n===== LOTTERY PREDICTION RESULTS =====")
        print(f"Predicted winning numbers: {sorted(predictions)}")
        print("======================================\n")
        
    except Exception as e:
        print(f"Error running prediction pipeline: {e}")
        sys.exit(1)