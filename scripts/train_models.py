import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union, Callable
import logging
import pickle
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
import warnings
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from models.utils import setup_logging, log_memory_usage
from models.feature_engineering import prepare_time_series_data, prepare_feature_data
from models.training_config import *

# Try to import utility functions
try:
    from scripts.utils import create_progress_bar, create_model_progress_tracker, update_model_progress
    # Import DataValidator fully qualified to avoid type conflicts
    from scripts.data_validation import DataValidator as ExternalDataValidator
except ImportError:
    try:
        # Try relative imports if we're inside the scripts package
        from .utils import create_progress_bar, create_model_progress_tracker, update_model_progress
        from .data_validation import DataValidator as ExternalDataValidator
    except ImportError:
        # Default implementation if imports fail
        def setup_logging():
            logging.basicConfig(filename='lottery.log', level=logging.INFO,
                             format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Dummy progress tracking functions with proper type handling
        class DummyTQDM:
            def __init__(self, iterable=None, total=None, **kwargs):
                self.iterable = iterable
                self.total = total or (len(iterable) if iterable is not None else None)
                self.n = 0
                self.kwargs = kwargs
            
            def update(self, n=1):
                self.n += n
                
            def close(self):
                pass
                
            def set_description(self, desc=None):
                pass
                
            def __iter__(self):
                if self.iterable is None:
                    return range(self.total).__iter__() if self.total is not None else iter([])
                return self.iterable.__iter__()
                
            def __enter__(self):
                return self
                
            def __exit__(self, *args, **kwargs):
                self.close()
        
        def create_progress_bar(*args, **kwargs):
            return DummyTQDM(*args, **kwargs)
        
        def create_model_progress_tracker(*args, **kwargs) -> Dict[str, Any]:
            return {"main": DummyTQDM(), "models": {}, "current": None}
        
        def update_model_progress(tracker: Dict[str, Any], model_name: str, desc: str = "", reset: bool = False) -> None:
            """Safe update for progress tracking that handles None trackers"""
            if tracker is not None:
                # Implementation would go here in a real implementation
                pass
        
        # Local fallback implementation
        class DataValidator:
            """Fallback implementation of DataValidator with minimal functionality"""
            
            def validate_prediction(self, prediction: Union[List[int], np.ndarray]) -> Tuple[bool, str]:
                """
                Validate a single prediction (6 unique integers, 1-59).
                
                Args:
                    prediction: List of predicted numbers
                
                Returns:
                    Tuple of (is_valid: bool, error_message: str)
                """
                try:
                    error_msg = ""
                    
                    # Check type
                    if not isinstance(prediction, (list, np.ndarray)):
                        error_msg = f"Prediction must be a list, got {type(prediction).__name__}"
                        return False, error_msg
                    
                    # Check length
                    if len(prediction) != 6:
                        error_msg = f"Prediction must contain exactly 6 numbers, got {len(prediction)}"
                        return False, error_msg
                    
                    # Check integer type
                    if not all(isinstance(n, (int, np.integer)) for n in prediction):
                        error_msg = "Prediction must contain only integers"
                        return False, error_msg
                    
                    # Check range
                    out_of_range = [n for n in prediction if not (1 <= n <= 59)]
                    if out_of_range:
                        error_msg = f"Prediction contains numbers outside range 1-59: {out_of_range}"
                        return False, error_msg
                    
                    # Check uniqueness
                    if len(set(prediction)) != 6:
                        error_msg = f"Prediction must contain 6 unique numbers, got {prediction}"
                        return False, error_msg
                    
                    return True, ""
                    
                except Exception as e:
                    error_msg = f"Error validating prediction: {str(e)}"
                    return False, error_msg
        
        # For type compatibility, use either the external or local version
        ExternalDataValidator = DataValidator

# Import data preparation functions
try:
    from scripts.fetch_data import load_data, prepare_training_data, prepare_feature_data, prepare_sequence_data
except ImportError:
    try:
        from .fetch_data import load_data, prepare_training_data, prepare_feature_data, prepare_sequence_data
    except ImportError:
        # Will be caught by the model imports error handling below
        pass

# Import model training functions
try:
    # First try from models package
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.lstm_model import train_lstm_model
    from models.arima_model import train_arima_model
    from models.holtwinters_model import train_holtwinters_model
    from models.linear_model import train_linear_models
    from models.xgboost_model import train_xgboost_model
    from models.lightgbm_model import train_lightgbm_model
    from models.knn_model import train_knn_model
    from models.gradient_boosting_model import train_gradient_boosting_model
    from models.catboost_model import train_catboost_model
    from models.cnn_lstm_model import train_cnn_lstm_model
    from models.autoencoder_model import train_autoencoder_model
    from models.meta_model import train_meta_model
    from models.feature_engineering import (
        enhanced_time_series_features, 
        prepare_lstm_features,
        expand_draw_sequences
    )
    from models.training_config import (
        LSTM_CONFIG, 
        CNN_LSTM_CONFIG,
        AUTOENCODER_CONFIG,
        HOLTWINTERS_CONFIG,
        LINEAR_CONFIG,
        XGBOOST_CONFIG,
        LIGHTGBM_CONFIG,
        KNN_CONFIG,
        GRADIENT_BOOSTING_CONFIG,
        CATBOOST_CONFIG,
        META_CONFIG,
        DATA_PATH, 
        MODELS_DIR
    )
    
    # Import prediction functions for validation
    from models.linear_model import predict_linear_models as predict_linear_model
    from models.lstm_model import predict_lstm_model
    from models.xgboost_model import predict_xgboost_model
    from models.catboost_model import predict_catboost_model
    from models.lightgbm_model import predict_lightgbm_model
    from models.gradient_boosting_model import predict_gradient_boosting_model
    from models.knn_model import predict_knn_model
except ImportError as e:
    # Try alternative import paths
    try:
        import sys
        sys.path.append(os.path.abspath("."))
        from models.lstm_model import train_lstm_model, predict_lstm_model
        from models.arima_model import train_arima_model
        from models.holtwinters_model import train_holtwinters_model
        from models.linear_model import train_linear_models
        from models.xgboost_model import train_xgboost_model
        from models.lightgbm_model import train_lightgbm_model
        from models.knn_model import train_knn_model
        from models.gradient_boosting_model import train_gradient_boosting_model
        from models.catboost_model import train_catboost_model
        from models.cnn_lstm_model import train_cnn_lstm_model
        from models.autoencoder_model import train_autoencoder_model
        from models.meta_model import train_meta_model
        from models.feature_engineering import (
            enhanced_time_series_features, 
            prepare_lstm_features,
            expand_draw_sequences
        )
        
        # Import prediction functions for validation
        from models.linear_model import predict_linear_models as predict_linear_model
        from models.xgboost_model import predict_xgboost_model
        from models.catboost_model import predict_catboost_model
        from models.lightgbm_model import predict_lightgbm_model
        from models.gradient_boosting_model import predict_gradient_boosting_model
        from models.knn_model import predict_knn_model
        
        # Define defaults if config not found
        DATA_PATH = "data/lottery_data_1995_2025.csv"
        MODELS_DIR = Path("models/checkpoints")
        
        # Default configs
        LSTM_CONFIG = {
            "look_back": 200,
            "lstm_units_1": 256,
            "lstm_units_2": 128,
            "dropout_rate": 0.3,
            "l2_reg": 0.001,
            "batch_size": 64,
            "epochs": 200,
        }
        
        CNN_LSTM_CONFIG = {
            **LSTM_CONFIG,
            "filters_1": 128,
            "kernel_size": 3,
        }
        
        # Default configs for other models
        AUTOENCODER_CONFIG = {
            "params": {
                "encoding_dim": 32,
                "epochs": 100,
            }
        }
        
        HOLTWINTERS_CONFIG = {
            "params": {
                "seasonal_periods": 12,
            }
        }
        
        LINEAR_CONFIG = {
            "params": {
                "fit_intercept": True,
            }
        }
        
        XGBOOST_CONFIG = {
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
            }
        }
        
        LIGHTGBM_CONFIG = {
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
            }
        }
        
        KNN_CONFIG = {
            "params": {
                "n_neighbors": 5,
            }
        }
        
        GRADIENT_BOOSTING_CONFIG = {
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
            }
        }
        
        CATBOOST_CONFIG = {
            "params": {
                "iterations": 100,
                "learning_rate": 0.1,
            }
        }
        
        META_CONFIG = {
            "params": {
                "n_estimators": 100,
            }
        }
        
    except ImportError as inner_e:
        print(f"Error importing model modules: {inner_e}")
        raise ImportError(f"Failed to import model modules. Original error: {e}, Alternate path error: {inner_e}")

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path("models/checkpoints")
MODELS_PATH = MODELS_DIR / "trained_models.pkl"

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def train_model(model_name: str, df: pd.DataFrame, params: Dict[str, Any], tracker: Optional[Dict[str, Any]] = None) -> Any:
    """
    Train a model using the specified parameters.
    
    Args:
        model_name: Name of the model to train
        df: DataFrame with lottery data
        params: Parameters for the model
        tracker: Progress tracker object
    
    Returns:
        Trained model or model tuple
    """
    try:
        # Prepare training data based on model type
        if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
            X, y = prepare_training_data(df)
        else:
            X, y = prepare_feature_data(df)
            
        # Ensure y is not None
        if y is None:
            y = np.zeros((X.shape[0], 6))  # Default placeholder for models that need y
        
        # Import and train the specified model
        # Each model function should return the trained model
        if model_name == 'lstm':
            from models.lstm_model import train_lstm_model
            update_model_status(model_name, "training", tracker)
            model = train_lstm_model(X, y, params)
            update_model_status(model_name, "trained", tracker)
            return model
        
        # Add other model cases here...
        
        # Default case
        update_model_status(model_name, "model not implemented", tracker)
        return None
    
    except Exception as e:
        update_model_status(model_name, f"error: {str(e)}", tracker)
        logger.error(f"Error training {model_name} model: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def validate_model_predictions(model_name: str, model_tuple: Union[Any, Tuple[Any, ...]], df: pd.DataFrame) -> bool:
    """Validate model predictions to ensure they are valid lottery numbers"""
    try:
        # Prepare appropriate data based on model type
        if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
            X, _ = prepare_time_series_data(df)
            model, scaler = model_tuple[:2]  # Unpack model and scaler
            predictions = model.predict(X)
        else:
            X, _ = prepare_feature_data(df)
            model, scaler = model_tuple[:2]  # Unpack model and scaler
            if isinstance(model, list):  # For models that return list of models
                predictions = np.zeros((len(X), 6))
                X_scaled = scaler.transform(X)
                for i, m in enumerate(model):
                    predictions[:, i] = m.predict(X_scaled)
            else:
                predictions = model.predict(scaler.transform(X))

        # Validate predictions
        validator = DataValidator()
        for pred in predictions[:10]:  # Check first 10 predictions
            is_valid, message = validator.validate_prediction(pred)
            if not is_valid:
                logging.error(f"Invalid prediction from {model_name}: {message}")
                return False
        return True
        
    except Exception as e:
        logging.error(f"Error validating {model_name} model: {str(e)}")
        return False

def train_all_models(df: pd.DataFrame, force_retrain: Union[bool, str] = False) -> Dict[str, Any]:
    """
    Train all models for lottery prediction using optimized configurations.
    
    Args:
        df: DataFrame with lottery data
        force_retrain: Whether to retrain models even if they exist (True/False or "yes"/"no")
    
    Returns:
        Dictionary of trained models
    """
    start_time = time.time()
    
    # Normalize force_retrain parameter
    if isinstance(force_retrain, str):
        force_retrain = force_retrain.lower() in ['yes', 'true', 'y', '1']
    
    # Check for existing models
    models_dict = {}
    if not force_retrain and MODELS_PATH.exists():
        try:
            logger.info(f"Loading existing models from {MODELS_PATH}")
            with open(MODELS_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                
            if isinstance(saved_data, dict) and 'models' in saved_data:
                models_dict = saved_data['models']
                timestamp = saved_data.get('timestamp', 'unknown')
                logger.info(f"Loaded {len(models_dict)} models trained on {timestamp}")
                return models_dict
            else:
                logger.warning("Saved models file has invalid format. Will retrain models.")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.info("Will retrain models due to loading error.")
    
    logger.info(f"{'Retraining' if force_retrain else 'Training'} all models with optimized configuration...")
    
    # Set up model progress tracker
    model_list = [
        'lstm', 'cnn_lstm', 'autoencoder', 'holtwinters', 'linear', 
        'xgboost', 'lightgbm', 'knn', 'gradient_boosting', 'catboost', 'meta'
    ]
    progress_tracker = create_model_progress_tracker(model_list, 
                                                   desc=f"{'Retraining' if force_retrain else 'Training'} models")
    
    # Prepare enhanced features with progress bar
    try:
        # Create enhanced features for better model performance
        logger.info("Generating enhanced time series features...")
        update_model_progress(progress_tracker, "preprocessing", desc="Generating enhanced features")
        
        # Create progress bar for data preparation
        pbar = create_progress_bar(total=5, desc="Preparing data", position=1, leave=False)
        
        # Generate enhanced features
        df_enhanced = enhanced_time_series_features(
            df, 
            look_back=LSTM_CONFIG.get("look_back", 200),
            use_tsfresh=True,
            cache_features=True
        )
        pbar.update(1)
        
        logger.info(f"Generated enhanced features: {len(df_enhanced.columns)} columns")
        
        # Prepare optimized data for LSTM
        logger.info("Preparing optimized LSTM features...")
        lstm_X, lstm_y, lstm_scaler = prepare_lstm_features(
            df, 
            look_back=LSTM_CONFIG.get("look_back", 200),
            scale_method=LSTM_CONFIG.get("scaling_method", "robust"),
            use_pca=LSTM_CONFIG.get("use_pca", False),
            pca_components=LSTM_CONFIG.get("pca_components", 30)
        )
        pbar.update(1)
        
        logger.info(f"Prepared LSTM features with shape X:{lstm_X.shape}, y:{lstm_y.shape}")
        
        # For regular models
        # For time-series models (LSTM, CNN-LSTM) - fallback if optimized features fail
        ts_X, ts_y = prepare_training_data(df, look_back=LSTM_CONFIG.get("look_back", 200))
        pbar.update(1)
        
        logger.info(f"Prepared fallback time-series data with shape X:{ts_X.shape}, y:{ts_y.shape}")
        
        # For feature-based models (XGBoost, LightGBM, etc.)
        feat_X, feat_y = prepare_feature_data(df_enhanced, use_all_features=True)
        pbar.update(1)
        
        logger.info(f"Prepared feature data with shape X:{feat_X.shape}, y:{feat_y.shape}")
        
        # For sequence models with expanded features
        seq_df = expand_draw_sequences(df_enhanced, n_prev_draws=5)
        seq_X, seq_y = prepare_sequence_data(seq_df, seq_length=10)
        pbar.update(1)
        
        logger.info(f"Prepared sequence data with shape X:{seq_X.shape}, y:{seq_y.shape}")
        
        # For HoltWinters (univariate time series)
        # Extract lottery numbers as 1D time-series
        hw_y = np.array([num for draw in df['Main_Numbers'] for num in draw])
        logger.info(f"Prepared HoltWinters data with shape y:{hw_y.shape}")
        
        # Close the progress bar
        pbar.close()
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}
    
    # Train all models
    models_dict = {}
    failed_models = []
    
    # Time-series models with optimized features
    logger.info("Training LSTM model with optimized features...")
    try:
        # Extract params from LSTM_CONFIG
        lstm_params = LSTM_CONFIG.get('params', {})
        # Convert params dictionary to expected types for the specific model function
        lstm_epochs = lstm_params.get('epochs', 100)  # Default to 100 if not specified
        models_dict['lstm'] = train_model('lstm', df, lstm_params, progress_tracker)
        if models_dict['lstm'] is None:
            # Fallback to regular features
            logger.warning("Optimized LSTM training failed, trying with standard features...")
            models_dict['lstm'] = train_model('lstm', df, lstm_params, progress_tracker)
            if models_dict['lstm'] is None:
                failed_models.append('lstm')
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
        failed_models.append('lstm')
    
    logger.info("Training CNN-LSTM model with optimized features...")
    try:
        # Extract params from CNN_LSTM_CONFIG
        cnn_lstm_params = CNN_LSTM_CONFIG.get('params', {})
        models_dict['cnn_lstm'] = train_model('cnn_lstm', df, cnn_lstm_params, progress_tracker)
        if models_dict['cnn_lstm'] is None:
            # Fallback to regular features
            logger.warning("Optimized CNN-LSTM training failed, trying with standard features...")
            models_dict['cnn_lstm'] = train_model('cnn_lstm', df, cnn_lstm_params, progress_tracker)
            if models_dict['cnn_lstm'] is None:
                failed_models.append('cnn_lstm')
    except Exception as e:
        logger.error(f"Error training CNN-LSTM model: {str(e)}")
        failed_models.append('cnn_lstm')
    
    # Train autoencoder with sequence data
    autoencoder_params = AUTOENCODER_CONFIG.get('params', {})
    models_dict['autoencoder'] = train_model('autoencoder', df, autoencoder_params, progress_tracker)
    if models_dict['autoencoder'] is None:
        failed_models.append('autoencoder')
    
    # Statistical model
    hw_params = HOLTWINTERS_CONFIG.get('params', {})
    models_dict['holtwinters'] = train_model('holtwinters', df, hw_params, progress_tracker)
    if models_dict['holtwinters'] is None:
        failed_models.append('holtwinters')
    
    # Feature-based models with enhanced features
    linear_params = LINEAR_CONFIG.get('params', {})
    models_dict['linear'] = train_model('linear', df, linear_params, progress_tracker)
    if models_dict['linear'] is None:
        failed_models.append('linear')
    
    xgboost_params = XGBOOST_CONFIG.get('params', {})
    models_dict['xgboost'] = train_model('xgboost', df, xgboost_params, progress_tracker)
    if models_dict['xgboost'] is None:
        failed_models.append('xgboost')
    
    lightgbm_params = LIGHTGBM_CONFIG.get('params', {})
    models_dict['lightgbm'] = train_model('lightgbm', df, lightgbm_params, progress_tracker)
    if models_dict['lightgbm'] is None:
        failed_models.append('lightgbm')
    
    knn_params = KNN_CONFIG.get('params', {})
    models_dict['knn'] = train_model('knn', df, knn_params, progress_tracker)
    if models_dict['knn'] is None:
        failed_models.append('knn')
    
    gb_params = GRADIENT_BOOSTING_CONFIG.get('params', {})
    models_dict['gradient_boosting'] = train_model('gradient_boosting', df, gb_params, progress_tracker)
    if models_dict['gradient_boosting'] is None:
        failed_models.append('gradient_boosting')
    
    catboost_params = CATBOOST_CONFIG.get('params', {})
    models_dict['catboost'] = train_model('catboost', df, catboost_params, progress_tracker)
    if models_dict['catboost'] is None:
        failed_models.append('catboost')
    
    # Train meta model if we have enough base models
    successful_models = {k: v for k, v in models_dict.items() if v is not None}
    if len(successful_models) >= 3:
        update_model_progress(progress_tracker, "meta", desc="Preparing meta model")
        logger.info(f"Training meta model based on {len(successful_models)} successful base models")
        
        # Generate base predictions for meta model
        base_predictions = []
        
        # Create progress bar for base predictions
        base_pbar = create_progress_bar(total=len(successful_models), desc="Generating base predictions", position=1, leave=False)
        
        for model_name, model in successful_models.items():
            try:
                valid = validate_model_predictions(model_name, model, df)
                if valid:
                    base_predictions.append(model_name)
            except Exception as e:
                logger.error(f"Error generating predictions for meta model input from {model_name}: {str(e)}")
            base_pbar.update(1)
        
        # Close the progress bar
        base_pbar.close()
        
        if len(base_predictions) >= 3:
            try:
                # Create meta features from base model predictions
                meta_X = np.array([successful_models[model_name] for model_name in base_predictions])
                meta_y = feat_y[-1:]  # Use last actual draw as target
                
                meta_params = META_CONFIG.get('params', {})
                models_dict['meta'] = train_model('meta', df, meta_params, progress_tracker)
                if models_dict['meta'] is None:
                    failed_models.append('meta')
            except Exception as e:
                logger.error(f"Error training meta model: {str(e)}")
                logger.debug(traceback.format_exc())
                failed_models.append('meta')
    else:
        logger.warning(f"Not enough successful models to train meta model. Need at least 3, got {len(successful_models)}")
    
    # Validate trained models
    validated_models = {}
    update_model_progress(progress_tracker, "validation", desc="Validating models")
    
    # Create progress bar for validation
    validation_pbar = create_progress_bar(total=len(models_dict), desc="Validating models", position=1, leave=False)
    
    for model_name, model in models_dict.items():
        if model is not None:
            validation_pbar.set_description(f"Validating {model_name}")
            valid = validate_model_predictions(model_name, model, df)
            if valid:
                validated_models[model_name] = model
            else:
                logger.warning(f"Model {model_name} failed validation and will not be included")
                failed_models.append(f"{model_name} (validation)")
        validation_pbar.update(1)
    
    # Close the progress bar
    validation_pbar.close()
    
    # Save models to file
    update_model_progress(progress_tracker, "saving", desc="Saving models")
    
    models_to_save = {
        'models': validated_models,
        'timestamp': datetime.now().isoformat(),
        'data_size': len(df),
        'training_time': time.time() - start_time,
        'failed_models': failed_models,
        'config': {
            'lstm': LSTM_CONFIG,
            'cnn_lstm': CNN_LSTM_CONFIG
        }
    }
    
    try:
        MODELS_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(MODELS_PATH, 'wb') as f:
            pickle.dump(models_to_save, f)
        logger.info(f"Saved {len(validated_models)} models to {MODELS_PATH}")
    except Exception as e:
        logger.error(f"Error saving models to {MODELS_PATH}: {str(e)}")
        logger.debug(traceback.format_exc())
    
    # Log summary
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    logger.info(f"Successfully trained {len(validated_models)} models")
    
    if failed_models:
        logger.warning(f"Failed models: {', '.join(failed_models)}")
    
    # Close progress tracker
    if progress_tracker and "main" in progress_tracker and hasattr(progress_tracker["main"], "close"):
        progress_tracker["main"].close()
    
    return validated_models

def load_trained_models() -> Dict[str, Any]:
    """
    Load pre-trained models from file.
    
    Returns:
        Dictionary of trained models or empty dict if no models found
    """
    try:
        if not MODELS_PATH.exists():
            logger.warning(f"No trained models found at {MODELS_PATH}")
            return {}
        
        with open(MODELS_PATH, 'rb') as f:
            saved_data = pickle.load(f)
        
        if isinstance(saved_data, dict) and 'models' in saved_data:
            models = saved_data['models']
            timestamp = saved_data.get('timestamp', 'unknown')
            data_size = saved_data.get('data_size', 'unknown')
            logger.info(f"Loaded {len(models)} models trained on {timestamp} with {data_size} samples")
            return models
        else:
            logger.warning(f"Invalid format in {MODELS_PATH}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading trained models: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}

def update_model_status(model_name: str, status: str, tracker: Optional[Dict[str, Any]] = None, save: bool = True) -> None:
    """
    Update the status of a model in the tracker.
    
    Args:
        model_name: Name of the model
        status: Status message to update
        tracker: Progress tracker object
        save: Whether to save the tracker to disk
    """
    # Update model status in the tracker if provided
    if tracker is not None:
        update_model_progress(tracker, model_name, status)
    
    # Log status in any case
    logger.info(f"Model {model_name}: {status}")
    
    # Save tracker to disk if requested
    if save and tracker is not None:
        try:
            with open("model_status.json", "w") as f:
                json.dump({k: str(v) for k, v in tracker.items() if k != "main"}, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save model status: {str(e)}")

if __name__ == "__main__":
    try:
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train lottery prediction models')
        parser.add_argument('--data', type=str, default=str(DATA_PATH),
                           help=f'Path to lottery data (default: {DATA_PATH})')
        parser.add_argument('--force', type=str, choices=['yes', 'no'], default='no',
                           help='Force retraining of models (yes/no)')
        parser.add_argument('--config', type=str, choices=['default', 'quick', 'deep'], default='default',
                           help='Configuration to use for training (default/quick/deep)')
        
        args = parser.parse_args()
        
        # Apply configuration
        if args.config == 'quick':
            from models.training_config import QUICK_TRAINING_CONFIG as CONFIG
            print("Using QUICK training configuration")
        elif args.config == 'deep':
            from models.training_config import DEEP_TRAINING_CONFIG as CONFIG
            print("Using DEEP training configuration")
        else:
            from models.training_config import LSTM_CONFIG as CONFIG
            print("Using DEFAULT training configuration")
        
        # Load data
        print(f"Loading lottery data from {args.data}...")
        df = load_data(args.data)
        print(f"Loaded {len(df)} lottery draws")
        
        # Train models
        print(f"Training models (force={args.force}, config={args.config})...")
        models = train_all_models(df, force_retrain=args.force)
        
        if models:
            print(f"Successfully trained {len(models)} models:")
            for model_name in models.keys():
                print(f"  - {model_name}")
        else:
            print("No models were successfully trained")
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc() 