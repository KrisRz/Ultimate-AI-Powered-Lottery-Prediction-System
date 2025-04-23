import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import optuna
from typing import Tuple, List, Dict, Any, Union, Optional, Callable, overload, TypeVar, Protocol, runtime_checkable
import logging
from .utils import log_training_errors, ensure_valid_prediction
from numpy.typing import NDArray
from optuna import Trial
from optuna.study import Study
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import ensure_valid_prediction  # Import directly from utils.py
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.sparse import spmatrix

T = TypeVar('T', bound=BaseEstimator | RegressorMixin)

@runtime_checkable
class Predictor(Protocol):
    def predict(self, X: Union[NDArray, spmatrix]) -> NDArray: ...

def get_linear_model(model_type: str = 'linear', params: Optional[Dict] = None) -> Any:
    """
    Get the appropriate linear model based on specified type
    
    Args:
        model_type: Type of linear model ('linear', 'ridge', 'lasso', 'elastic_net')
        params: Optional parameters for the model
        
    Returns:
        Linear model instance
    """
    if params is None:
        params = {}
    
    # Create the requested linear model
    if model_type == 'linear':
        return LinearRegression(**params)
    elif model_type == 'ridge':
        # Default alpha if not specified
        if 'alpha' not in params:
            params['alpha'] = 1.0
        return Ridge(**params)
    elif model_type == 'lasso':
        # Default alpha if not specified
        if 'alpha' not in params:
            params['alpha'] = 0.1
        return Lasso(**params)
    elif model_type == 'elastic_net':
        # Default parameters if not specified
        if 'alpha' not in params:
            params['alpha'] = 0.1
        if 'l1_ratio' not in params:
            params['l1_ratio'] = 0.5
        return ElasticNet(**params)
    else:
        logging.warning(f"Unknown model type: {model_type}. Falling back to LinearRegression.")
        return LinearRegression()

@overload
def validate_prediction(pred: Union[List, np.ndarray], as_array: bool = True) -> NDArray: ...

@overload
def validate_prediction(pred: Union[List, np.ndarray], as_array: bool = False) -> List[int]: ...

def validate_prediction(pred: Union[List, np.ndarray], as_array: bool = True) -> Union[NDArray, List[int]]:
    """
    Ensure predictions are 6 unique integers between 1 and 59.
    
    Args:
        pred: Raw model prediction
        as_array: Whether to return numpy array (True) or list (False)
        
    Returns:
        Array or list of 6 unique integers between 1 and 59
    """
    try:
        # Convert to numpy array if it's a list
        if isinstance(pred, list):
            pred = np.array(pred)
        
        # Handle NaN, infinity, or other invalid values
        if pred is None or not isinstance(pred, (list, np.ndarray)):
            logging.warning(f"Invalid prediction type: {type(pred)}. Generating random numbers.")
            result = sorted(np.random.choice(range(1, 60), size=6, replace=False))
            return np.array(result) if as_array else list(result)
        
        # Check for NaN or infinite values
        if np.isnan(pred).any() or np.isinf(pred).any():
            logging.warning(f"Prediction contains NaN or Inf values: {pred}. Fixing...")
            pred = np.nan_to_num(pred, nan=30.0, posinf=59.0, neginf=1.0)
        
        # Handle potential fractional values by rounding
        pred = np.round(pred).astype(int)
        
        # Ensure values are within valid range (1-59)
        pred = np.clip(pred, 1, 59)
        
        # Check if we have the right number of predictions
        if len(pred) != 6:
            logging.warning(f"Prediction has {len(pred)} numbers instead of 6. Adjusting...")
            if len(pred) > 6:
                # Take the first 6 values
                pred = pred[:6]
            else:
                # Add random numbers until we have 6
                current = set(pred)
                while len(current) < 6:
                    candidate = np.random.randint(1, 60)
                    if candidate not in current:
                        current.add(candidate)
                pred = np.array(list(current))
        
        # Ensure uniqueness
        unique_values = set(pred)
        if len(unique_values) < 6:
            logging.warning(f"Prediction has duplicate values: {pred}. Fixing...")
            while len(unique_values) < 6:
                candidate = np.random.randint(1, 60)
                if candidate not in unique_values:
                    unique_values.add(candidate)
            pred = np.array(list(unique_values))
        
        # Sort and return in requested format
        result = sorted([int(x) for x in pred])
        return np.array(result) if as_array else result
    
    except Exception as e:
        logging.error(f"Error processing prediction: {e}. Generating random numbers.")
        result = sorted(np.random.choice(range(1, 60), size=6, replace=False))
        return np.array(result) if as_array else list(result)

def objective(trial: Trial, X: NDArray, y: NDArray, model_type: str, cv: int = 5) -> float:
    """
    Optuna objective function for linear model parameter tuning
    
    Args:
        trial: Optuna trial object
        X: Training features
        y: Target variable
        model_type: Type of linear model to optimize
        cv: Number of cross-validation folds
        
    Returns:
        Mean cross-validation score
    """
    # Get model with trial parameters
    params = {}
    if model_type == 'ridge':
        params['alpha'] = trial.suggest_float('alpha', 0.01, 10.0, log=True)
    elif model_type == 'lasso':
        params['alpha'] = trial.suggest_float('alpha', 0.01, 10.0, log=True)
    elif model_type == 'elastic_net':
        params['alpha'] = trial.suggest_float('alpha', 0.01, 10.0, log=True)
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        
    model = get_linear_model(model_type, params)
    
    # Cross validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_val, y_fold_val)
        scores.append(score)
        
    return float(np.mean(scores))  # Explicitly convert to float

class LinearModel:
    def __init__(self, models: List[Any], scaler: Union[StandardScaler, MinMaxScaler]):
        if not models or len(models) != 6:
            raise ValueError(f"Must provide exactly 6 models, got {len(models) if models else 0}")
        self.models = models
        self.scaler = scaler
        self.prediction_history = []
        self.validation_scores = []
        
    def predict(self, X: NDArray, return_proba: bool = False) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Make predictions with enhanced validation and confidence scores
        
        Args:
            X: Input features
            return_proba: Whether to return prediction probabilities
            
        Returns:
            predictions: Array of predictions
            probabilities: Array of confidence scores (if return_proba=True)
        """
        try:
            # Input validation
            if not isinstance(X, np.ndarray):
                raise ValueError(f"X must be numpy array, got {type(X)}")
            
            # Scale input data
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                logging.error(f"Failed to scale input: {str(e)}")
                raise RuntimeError(f"Feature scaling failed: {str(e)}")
            
            # Make predictions with confidence scores
            predictions = np.zeros((X.shape[0], 6))
            probabilities = np.zeros((X.shape[0], 6))
            
            for i, model in enumerate(self.models):
                try:
                    pred = model.predict(X_scaled)
                    predictions[:, i] = pred
                    
                    # Calculate confidence score based on prediction range and model type
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)
                        probabilities[:, i] = np.max(prob, axis=1)
                    else:
                        # Use a simple confidence metric based on prediction range
                        probabilities[:, i] = 1.0 - np.abs(pred - 30) / 29  # Higher confidence for numbers closer to middle range
                        
                except Exception as e:
                    logging.error(f"Model {i+1} prediction failed: {str(e)}")
                    # Use fallback prediction
                    predictions[:, i] = np.random.randint(1, 60, size=X.shape[0])
                    probabilities[:, i] = 0.1  # Low confidence for fallback predictions
            
            # Validate and adjust predictions
            valid_predictions = []
            for i in range(predictions.shape[0]):
                pred_row = predictions[i]
                prob_row = probabilities[i]
                
                # Check for duplicates and adjust based on confidence
                unique_pred = set()
                final_pred = []
                final_prob = []
                
                # Sort by confidence
                indices = np.argsort(prob_row)[::-1]
                
                for idx in indices:
                    num = int(round(pred_row[idx]))
                    if num not in unique_pred and 1 <= num <= 59:
                        unique_pred.add(num)
                        final_pred.append(num)
                        final_prob.append(prob_row[idx])
                
                # Fill remaining slots with unused numbers
                while len(final_pred) < 6:
                    num = np.random.randint(1, 60)
                    if num not in unique_pred:
                        unique_pred.add(num)
                        final_pred.append(num)
                        final_prob.append(0.1)  # Low confidence for random fills
                
                valid_predictions.append(final_pred)
                probabilities[i] = final_prob
            
            valid_predictions = np.array(valid_predictions)
            
            # Store prediction history
            if valid_predictions.shape[0] == 1:
                self.prediction_history.append(valid_predictions[0])
            
            if return_proba:
                return valid_predictions, probabilities
            return valid_predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            # Return safe fallback predictions
            fallback = np.array([sorted(np.random.choice(range(1, 60), size=6, replace=False)) 
                               for _ in range(X.shape[0])])
            if return_proba:
                return fallback, np.full((X.shape[0], 6), 0.1)
            return fallback
            
    def validate_model_stability(self, X: NDArray, n_runs: int = 10) -> float:
        """
        Validate model stability through multiple prediction runs
        
        Args:
            X: Validation data
            n_runs: Number of validation runs
            
        Returns:
            stability_score: Score between 0 and 1 indicating prediction stability
        """
        predictions = []
        for _ in range(n_runs):
            pred = self.predict(X)
            predictions.append(pred)
        
        # Calculate stability score based on prediction variance
        predictions = np.array(predictions)
        stability = 1.0 - np.mean(np.std(predictions, axis=0)) / 29  # Normalize by max possible std
        return max(0.0, min(1.0, stability))  # Clip to [0, 1]

@log_training_errors
def train_linear_models(X_train: NDArray, y_train: NDArray, 
                       model_type: str = 'ridge', tune_hyperparams: bool = True,
                       n_trials: int = 50, scaler_type: str = 'standard',
                       validation_size: float = 0.2) -> LinearModel:
    """
    Train linear models for lottery number prediction with enhanced error handling
    
    Args:
        X_train: Training features
        y_train: Target variables (6 numbers per sample)
        model_type: Type of linear model ('linear', 'ridge', 'lasso', 'elastic_net')
        tune_hyperparams: Whether to tune hyperparameters with Optuna
        n_trials: Number of Optuna trials for hyperparameter tuning
        scaler_type: Type of feature scaling ('standard' or 'minmax')
        validation_size: Size of validation split for model checking
        
    Returns:
        LinearModel instance with trained models and scaler
    """
    try:
        # Input validation with detailed error messages
        if not isinstance(X_train, np.ndarray):
            raise ValueError(f"X_train must be numpy array, got {type(X_train)}")
        if not isinstance(y_train, np.ndarray):
            raise ValueError(f"y_train must be numpy array, got {type(y_train)}")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train must have same number of samples. Got {X_train.shape[0]} vs {y_train.shape[0]}")
        if y_train.shape[1] != 6:
            raise ValueError(f"y_train must have 6 columns, got {y_train.shape[1]}")
            
        # Check for NaN/Inf values
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            raise ValueError("X_train contains NaN or Inf values")
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            raise ValueError("y_train contains NaN or Inf values")
            
        # Scale features with error handling
        try:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_train)
        except Exception as e:
            logging.error(f"Failed to scale features: {str(e)}")
            raise RuntimeError(f"Feature scaling failed: {str(e)}")
        
        # Split data for validation
        n_val = int(X_train.shape[0] * validation_size)
        if n_val > 0:
            X_val = X_scaled[-n_val:]
            y_val = y_train[-n_val:]
            X_train_split = X_scaled[:-n_val]
            y_train_split = y_train[:-n_val]
        else:
            X_val = X_scaled
            y_val = y_train
            X_train_split = X_scaled
            y_train_split = y_train
        
        models = []
        failed_models = []
        
        # Train a separate model for each lottery number with enhanced error handling
        for i in range(6):
            try:
                if tune_hyperparams:
                    study = optuna.create_study(direction='maximize')
                    study.optimize(lambda trial: objective(trial, X_train_split, y_train_split[:, i], model_type),
                                 n_trials=n_trials, 
                                 catch=(Exception,))  # Catch exceptions during optimization
                    
                    if not study.trials:
                        raise RuntimeError(f"No successful trials for number {i+1}")
                        
                    best_params = study.best_params
                    model = get_linear_model(model_type, best_params)
                else:
                    model = get_linear_model(model_type)
                
                # Fit model with error handling
                model.fit(X_train_split, y_train_split[:, i])
                
                # Validate model performance
                val_pred = model.predict(X_val)
                val_score = model.score(X_val, y_val[:, i])
                
                if val_score < 0:  # Basic sanity check
                    logging.warning(f"Model {i+1} has poor validation score: {val_score}")
                    # Try retraining with different random seed
                    model = get_linear_model(model_type)
                    model.fit(X_train_split, y_train_split[:, i])
                    val_score = model.score(X_val, y_val[:, i])
                    
                    if val_score < 0:
                        raise RuntimeError(f"Model {i+1} failed validation after retry")
                
                # Validate predictions are in valid range
                if not np.all((val_pred >= 1) & (val_pred <= 59)):
                    raise RuntimeError(f"Model {i+1} predictions out of valid range")
                
                models.append(model)
                logging.info(f"Successfully trained and validated model {i+1}")
                
            except Exception as e:
                logging.error(f"Failed to train model {i+1}: {str(e)}")
                failed_models.append(i+1)
                # Create fallback model
                fallback = get_linear_model('linear')  # Use simple linear regression as fallback
                fallback.fit(X_train_split, y_train_split[:, i])
                models.append(fallback)
                logging.info(f"Using fallback model for number {i+1}")
        
        if failed_models:
            logging.warning(f"Used fallback models for numbers: {failed_models}")
        
        return LinearModel(models, scaler)
        
    except Exception as e:
        logging.error(f"Fatal error in train_linear_models: {str(e)}")
        raise

def predict_linear_models(models: Union[Tuple[List[Predictor], StandardScaler], List[Predictor]], 
                        X: Union[NDArray, spmatrix]) -> List[int]:
    """
    Generate predictions using trained linear models.
    
    Args:
        models: Either (list of models, scaler) tuple or just list of models
        X: Input features to predict on
        
    Returns:
        List of 6 valid lottery numbers
    """
    try:
        # Handle different input formats
        if isinstance(models, tuple):
            model_list, scaler = models
            X = scaler.transform(X)
        else:
            model_list = models
            
        if len(model_list) == 1:
            # Single model case
            logging.info("Using single model for prediction")
            predictions = model_list[0].predict(X)
            return list(validate_prediction(predictions[0], as_array=False))
            
        # Multiple models case
        valid_predictions = []
        for model in model_list:
            pred = model.predict(X)
            valid_predictions.append(pred[0])
            
        return list(validate_prediction(valid_predictions, as_array=False))
        
    except Exception as e:
        logging.error(f"Error making prediction: {e}. Returning random numbers.")
        return list(validate_prediction(np.random.randint(1, 60, size=6), as_array=False)) 