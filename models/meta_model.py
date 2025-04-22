import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any, Union, Optional
import optuna
from sklearn.model_selection import KFold
from .utils import log_training_errors, ensure_valid_prediction

# Import optional meta-learners with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with 'pip install xgboost' for additional meta-learners.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with 'pip install lightgbm' for additional meta-learners.")

def get_meta_learner(meta_learner_type: str = 'random_forest', params: Optional[Dict] = None) -> Any:
    """
    Get the appropriate meta-learner model based on specified type
    
    Args:
        meta_learner_type: Type of meta-learner ('random_forest', 'xgboost', 'lightgbm')
        params: Optional parameters for the meta-learner
        
    Returns:
        Meta-learner model instance
    """
    if params is None:
        params = {}
    
    # Ensure random_state is set for reproducibility
    if 'random_state' not in params:
        params['random_state'] = 42
    
    # Create the requested meta-learner
    if meta_learner_type == 'random_forest':
        # Default parameters for RandomForest if not specified
        if 'n_estimators' not in params:
            params['n_estimators'] = 100
        return RandomForestRegressor(**params)
    
    elif meta_learner_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            logging.warning("XGBoost requested but not available. Falling back to RandomForest.")
            return get_meta_learner('random_forest', params)
        
        # Default parameters for XGBoost if not specified
        xgb_params = {
            'n_estimators': params.get('n_estimators', 100),
            'learning_rate': params.get('learning_rate', 0.1),
            'max_depth': params.get('max_depth', 6),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'gamma': params.get('gamma', 0),
            'random_state': params['random_state']
        }
        return xgb.XGBRegressor(**xgb_params)
    
    elif meta_learner_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            logging.warning("LightGBM requested but not available. Falling back to RandomForest.")
            return get_meta_learner('random_forest', params)
        
        # Default parameters for LightGBM if not specified
        lgb_params = {
            'n_estimators': params.get('n_estimators', 100),
            'learning_rate': params.get('learning_rate', 0.1),
            'max_depth': params.get('max_depth', 6),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'num_leaves': params.get('num_leaves', 31),
            'random_state': params['random_state']
        }
        return lgb.LGBMRegressor(**lgb_params)
    
    else:
        logging.warning(f"Unknown meta-learner type: {meta_learner_type}. Falling back to RandomForest.")
        return get_meta_learner('random_forest', params)

def objective(trial, X, y, meta_learner_type='random_forest', cv=5):
    """
    Optuna objective function for meta-learner parameter tuning
    
    Args:
        trial: Optuna trial object
        X: Training features (base model predictions)
        y: Target variable
        meta_learner_type: Type of meta-learner to optimize
        cv: Number of cross-validation folds
        
    Returns:
        Mean cross-validation score
    """
    # Define hyperparameters based on meta-learner type
    if meta_learner_type == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
    elif meta_learner_type == 'xgboost' and XGBOOST_AVAILABLE:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
    elif meta_learner_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
    else:
        # Fallback to RandomForest
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        meta_learner_type = 'random_forest'
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    # Get the meta-learner with current trial parameters
    model = get_meta_learner(meta_learner_type, params)
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.mean(scores)

@log_training_errors
def train_meta_model(base_predictions, y_train, meta_learner_type='random_forest', 
                    tune_hyperparams=True, n_trials=50):
    """
    Train a meta-learner model to combine predictions from base models
    
    Args:
        base_predictions: Stacked predictions from base models with shape [n_samples, n_models, 6]
                          or [n_samples, n_models*6]
        y_train: Target values (lottery numbers) with shape [n_samples, 6]
        meta_learner_type: Type of meta-learner ('random_forest', 'xgboost', 'lightgbm')
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter tuning trials
        
    Returns:
        Tuple of (trained meta-model, scaler, meta_learner_type)
    """
    # Validate input
    if not isinstance(base_predictions, np.ndarray):
        try:
            base_predictions = np.array(base_predictions)
        except Exception as e:
            raise ValueError(f"Cannot convert base_predictions to numpy array: {str(e)}")
    
    if not isinstance(y_train, np.ndarray):
        try:
            y_train = np.array(y_train)
        except Exception as e:
            raise ValueError(f"Cannot convert y_train to numpy array: {str(e)}")
    
    # Check shapes and reshape if needed
    if len(base_predictions.shape) == 3:
        # Shape is [n_samples, n_models, 6] - flatten to [n_samples, n_models*6]
        logging.info(f"Reshaping base predictions from {base_predictions.shape} to [n_samples, n_models*6]")
        base_predictions = base_predictions.reshape(base_predictions.shape[0], -1)
    elif len(base_predictions.shape) != 2:
        raise ValueError(f"base_predictions must have shape [n_samples, n_models*6] or [n_samples, n_models, 6], got {base_predictions.shape}")
    
    # Ensure y_train has shape [n_samples, 6]
    if len(y_train.shape) != 2 or y_train.shape[1] != 6:
        raise ValueError(f"y_train must have shape [n_samples, 6], got {y_train.shape}")
    
    # Ensure base_predictions and y_train have same number of samples
    if base_predictions.shape[0] != y_train.shape[0]:
        raise ValueError(f"base_predictions and y_train must have same number of samples, got {base_predictions.shape[0]} and {y_train.shape[0]}")
    
    # Scale the input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(base_predictions)
    
    # Initialize parameters
    best_params = None
    
    # Tune hyperparameters if requested
    if tune_hyperparams:
        try:
            logging.info(f"Tuning hyperparameters for {meta_learner_type} meta-learner")
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, X_scaled, y_train, meta_learner_type), 
                n_trials=n_trials
            )
            best_params = study.best_params
            logging.info(f"Best parameters: {best_params}")
        except Exception as e:
            logging.error(f"Error during hyperparameter tuning: {str(e)}")
            logging.info("Using default parameters due to tuning failure")
    
    # Create and train the meta-learner
    model = get_meta_learner(meta_learner_type, best_params)
    
    try:
        # Train for all 6 numbers together
        model.fit(X_scaled, y_train)
        logging.info(f"Meta-model ({meta_learner_type}) trained on {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
    except Exception as e:
        logging.error(f"Error training meta-model: {str(e)}")
        # Fallback to RandomForest with default parameters
        logging.info("Falling back to RandomForest with default parameters")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_train)
    
    return model, scaler, meta_learner_type

def predict_meta_model(model, scaler, base_predictions, meta_learner_type=None):
    """
    Generate predictions using the trained meta-model
    
    Args:
        model: Trained meta-model
        scaler: Fitted StandardScaler
        base_predictions: Stacked predictions from base models with shape [n_samples, n_models, 6]
                          or [n_samples, n_models*6]
        meta_learner_type: Type of meta-learner (for informational purposes only)
        
    Returns:
        Array of valid lottery numbers
    """
    try:
        # Validate input
        if not isinstance(base_predictions, np.ndarray):
            base_predictions = np.array(base_predictions)
        
        # Check shape and reshape if needed
        if len(base_predictions.shape) == 3:
            # Shape is [n_samples, n_models, 6] - flatten to [n_samples, n_models*6]
            base_predictions = base_predictions.reshape(base_predictions.shape[0], -1)
        elif len(base_predictions.shape) != 2:
            raise ValueError(f"base_predictions must have shape [n_samples, n_models*6] or [n_samples, n_models, 6], got {base_predictions.shape}")
        
        # Scale the input
        X_scaled = scaler.transform(base_predictions)
        
        # Generate predictions
        predictions = model.predict(X_scaled)
        
        # If predictions shape is [n_samples, 6]
        if len(predictions.shape) == 2 and predictions.shape[1] == 6:
            # For single prediction, process it
            if predictions.shape[0] == 1:
                return ensure_valid_prediction(predictions[0])
            
            # For multiple predictions, validate each
            valid_predictions = []
            for i in range(predictions.shape[0]):
                valid_predictions.append(ensure_valid_prediction(predictions[i]))
            return np.array(valid_predictions)
        
        # If predictions shape is [n_samples,], reshape to [n_samples, 6] if possible
        elif len(predictions.shape) == 1:
            if predictions.shape[0] % 6 == 0:
                predictions = predictions.reshape(-1, 6)
                if predictions.shape[0] == 1:
                    return ensure_valid_prediction(predictions[0])
                
                valid_predictions = []
                for i in range(predictions.shape[0]):
                    valid_predictions.append(ensure_valid_prediction(predictions[i]))
                return np.array(valid_predictions)
            else:
                return ensure_valid_prediction(predictions[:6])  # Take first 6 values
        
        # Unusual prediction shape, try to handle it
        logging.warning(f"Unusual prediction shape: {predictions.shape}, attempting to convert to valid lottery numbers")
        return ensure_valid_prediction(predictions.flatten()[:6])
        
    except Exception as e:
        logging.error(f"Error in meta-model prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 