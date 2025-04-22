import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union
import logging
import pickle
import os
import time
import traceback
from pathlib import Path
from datetime import datetime

# Try to import utility functions
try:
    from utils import setup_logging
    from data_validation import DataValidator
except ImportError:
    # Default implementation if imports fail
    def setup_logging():
        logging.basicConfig(filename='lottery.log', level=logging.INFO,
                           format='%(asctime)s - %(levelname)s - %(message)s')
    
    class DataValidator:
        def validate_prediction(self, prediction):
            return True

# Import data preparation functions
from fetch_data import load_data, prepare_training_data, prepare_feature_data, prepare_sequence_data

# Import model training functions
try:
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
except ImportError as e:
    # Try alternative import paths
    try:
        import sys
        sys.path.append(os.path.abspath("."))
        from lstm_model import train_lstm_model
        from arima_model import train_arima_model
        from holtwinters_model import train_holtwinters_model
        from linear_model import train_linear_models
        from xgboost_model import train_xgboost_model
        from lightgbm_model import train_lightgbm_model
        from knn_model import train_knn_model
        from gradient_boosting_model import train_gradient_boosting_model
        from catboost_model import train_catboost_model
        from cnn_lstm_model import train_cnn_lstm_model
        from autoencoder_model import train_autoencoder_model
        from meta_model import train_meta_model
    except ImportError as inner_e:
        print(f"Error importing model modules: {inner_e}")
        raise ImportError(f"Failed to import model modules. Original error: {e}, Alternate path error: {inner_e}")

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants
MODELS_PATH = Path("models/trained_models.pkl")
DATA_PATH = Path("data/lottery_data_1995_2025.csv")

def train_model(model_name: str, train_func: callable, X: np.ndarray, y: np.ndarray, 
                **kwargs) -> Optional[Any]:
    """
    Train a model and handle exceptions.
    
    Args:
        model_name: Name of the model for logging
        train_func: Function to train the model
        X: Training features
        y: Training targets
        **kwargs: Additional parameters for the training function
    
    Returns:
        Trained model or None if training fails
    """
    start_time = time.time()
    try:
        logger.info(f"Training {model_name} model...")
        model = train_func(X, y, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Successfully trained {model_name} model in {duration:.2f} seconds")
        return model
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed to train {model_name} model after {duration:.2f} seconds: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def validate_model_predictions(model_name: str, model: Any, df: pd.DataFrame) -> bool:
    """
    Generate and validate predictions from a trained model.
    
    Args:
        model_name: Name of the model
        model: Trained model
        df: DataFrame with lottery data
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        logger.info(f"Validating {model_name} model predictions...")
        
        # Prepare data for prediction based on model type
        if model_name in ['lstm', 'cnn_lstm']:
            X, _ = prepare_training_data(df)
            sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(sequence)[0]
        elif model_name == 'holtwinters':
            # HoltWinters works on time series
            prediction = model.forecast(6)
        elif model_name == 'autoencoder':
            X, _ = prepare_training_data(df)
            sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(sequence)[0]
        else:
            # Feature-based models
            X, _ = prepare_feature_data(df)
            prediction = model.predict(X[-1].reshape(1, -1))[0]
        
        # Round and format prediction
        prediction = np.round(prediction).astype(int)
        prediction = np.clip(prediction, 1, 59)
        
        # Ensure unique numbers
        unique_nums = set(prediction)
        while len(unique_nums) < 6:
            new_num = np.random.randint(1, 60)
            if new_num not in unique_nums:
                unique_nums.add(new_num)
        
        # Convert to list and sort
        prediction = sorted(list(unique_nums))[:6]
        
        # Validate prediction format
        validator = DataValidator()
        is_valid = validator.validate_prediction(prediction)
        
        if is_valid:
            logger.info(f"{model_name} model validation passed. Sample prediction: {prediction}")
            return True
        else:
            logger.warning(f"{model_name} model validation failed. Invalid prediction: {prediction}")
            return False
    
    except Exception as e:
        logger.error(f"Error validating {model_name} model: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def train_all_models(df: pd.DataFrame, force_retrain: Union[bool, str] = False) -> Dict[str, Any]:
    """
    Train all models for lottery prediction.
    
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
    
    logger.info(f"{'Retraining' if force_retrain else 'Training'} all models...")
    
    # Prepare different data formats required by different models
    try:
        # For time-series models (LSTM, CNN-LSTM)
        ts_X, ts_y = prepare_training_data(df)
        logger.info(f"Prepared time-series data with shape X:{ts_X.shape}, y:{ts_y.shape}")
        
        # For feature-based models (XGBoost, LightGBM, etc.)
        feat_X, feat_y = prepare_feature_data(df)
        logger.info(f"Prepared feature data with shape X:{feat_X.shape}, y:{feat_y.shape}")
        
        # For sequence models
        seq_X, seq_y = prepare_sequence_data(df)
        logger.info(f"Prepared sequence data with shape X:{seq_X.shape}, y:{seq_y.shape}")
        
        # For HoltWinters (univariate time series)
        # Extract lottery numbers as 1D time-series
        hw_y = np.array([num for draw in df['Main_Numbers'] for num in draw])
        logger.info(f"Prepared HoltWinters data with shape y:{hw_y.shape}")
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}
    
    # Train all models
    models_dict = {}
    failed_models = []
    
    # Time-series models
    models_dict['lstm'] = train_model('lstm', train_lstm_model, ts_X, ts_y)
    if models_dict['lstm'] is None:
        failed_models.append('lstm')
    
    models_dict['cnn_lstm'] = train_model('cnn_lstm', train_cnn_lstm_model, ts_X, ts_y)
    if models_dict['cnn_lstm'] is None:
        failed_models.append('cnn_lstm')
    
    models_dict['autoencoder'] = train_model('autoencoder', train_autoencoder_model, ts_X, ts_y)
    if models_dict['autoencoder'] is None:
        failed_models.append('autoencoder')
    
    # Statistical model
    models_dict['holtwinters'] = train_model('holtwinters', train_holtwinters_model, hw_y, None)
    if models_dict['holtwinters'] is None:
        failed_models.append('holtwinters')
    
    # Feature-based models
    models_dict['linear'] = train_model('linear', train_linear_models, feat_X, feat_y)
    if models_dict['linear'] is None:
        failed_models.append('linear')
    
    models_dict['xgboost'] = train_model('xgboost', train_xgboost_model, feat_X, feat_y)
    if models_dict['xgboost'] is None:
        failed_models.append('xgboost')
    
    models_dict['lightgbm'] = train_model('lightgbm', train_lightgbm_model, feat_X, feat_y)
    if models_dict['lightgbm'] is None:
        failed_models.append('lightgbm')
    
    models_dict['knn'] = train_model('knn', train_knn_model, feat_X, feat_y)
    if models_dict['knn'] is None:
        failed_models.append('knn')
    
    models_dict['gradient_boosting'] = train_model('gradient_boosting', train_gradient_boosting_model, feat_X, feat_y)
    if models_dict['gradient_boosting'] is None:
        failed_models.append('gradient_boosting')
    
    models_dict['catboost'] = train_model('catboost', train_catboost_model, feat_X, feat_y)
    if models_dict['catboost'] is None:
        failed_models.append('catboost')
    
    # Train meta model if we have enough base models
    successful_models = {k: v for k, v in models_dict.items() if v is not None}
    if len(successful_models) >= 3:
        logger.info(f"Training meta model based on {len(successful_models)} successful base models")
        
        # Generate base predictions for meta model
        base_predictions = []
        for model_name, model in successful_models.items():
            try:
                valid = validate_model_predictions(model_name, model, df)
                if valid:
                    base_predictions.append(model_name)
            except Exception as e:
                logger.error(f"Error generating predictions for meta model input from {model_name}: {str(e)}")
        
        if len(base_predictions) >= 3:
            try:
                # Create meta features from base model predictions
                meta_X = np.array([successful_models[model_name] for model_name in base_predictions])
                meta_y = feat_y[-1:]  # Use last actual draw as target
                
                models_dict['meta'] = train_model('meta', train_meta_model, meta_X, meta_y)
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
    for model_name, model in models_dict.items():
        if model is not None:
            valid = validate_model_predictions(model_name, model, df)
            if valid:
                validated_models[model_name] = model
            else:
                logger.warning(f"Model {model_name} failed validation and will not be included")
                failed_models.append(f"{model_name} (validation)")
    
    # Save models to file
    models_to_save = {
        'models': validated_models,
        'timestamp': datetime.now().isoformat(),
        'data_size': len(df),
        'training_time': time.time() - start_time,
        'failed_models': failed_models
    }
    
    try:
        MODELS_PATH.parent.mkdir(exist_ok=True)
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

if __name__ == "__main__":
    try:
        import argparse
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='Train lottery prediction models')
        parser.add_argument('--data', type=str, default=str(DATA_PATH),
                           help=f'Path to lottery data (default: {DATA_PATH})')
        parser.add_argument('--force', type=str, choices=['yes', 'no'], default='no',
                           help='Force retraining of models (yes/no)')
        
        args = parser.parse_args()
        
        # Load data
        print(f"Loading lottery data from {args.data}...")
        df = load_data(args.data)
        print(f"Loaded {len(df)} lottery draws")
        
        # Train models
        print(f"Training models (force={args.force})...")
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