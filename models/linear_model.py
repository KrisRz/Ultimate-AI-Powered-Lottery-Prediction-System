import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import optuna
from typing import Tuple, List, Dict, Any, Union, Optional
import logging
from .utils import log_training_errors, ensure_valid_prediction

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

def objective(trial, X, y, model_type='ridge', cv=5):
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
    # Define hyperparameters based on model type
    if model_type == 'ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        }
    elif model_type == 'lasso':
        params = {
            'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'max_iter': trial.suggest_int('max_iter', 1000, 3000)
        }
    elif model_type == 'elastic_net':
        params = {
            'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'max_iter': trial.suggest_int('max_iter', 1000, 3000)
        }
    else:  # LinearRegression has few parameters to tune
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    # Get the model with current trial parameters
    model = get_linear_model(model_type, params)
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.mean(scores)

@log_training_errors
def train_linear_models(X_train: np.ndarray, y_train: np.ndarray, 
                       model_type: str = 'ridge', tune_hyperparams: bool = True,
                       n_trials: int = 50, scaler_type: str = 'standard') -> Tuple[List[Any], Any]:
    """
    Train multiple linear regression models for lottery number prediction
    
    Args:
        X_train: Training features
        y_train: Target numbers (array of shape [n_samples, 6])
        model_type: Type of linear model ('linear', 'ridge', 'lasso', 'elastic_net')
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter tuning trials
        scaler_type: Type of feature scaling ('standard' or 'minmax')
        
    Returns:
        Tuple of (list of trained models, scaler)
    """
    # Validate input
    if not isinstance(X_train, np.ndarray):
        try:
            X_train = np.array(X_train)
        except Exception as e:
            raise ValueError(f"Cannot convert X_train to numpy array: {str(e)}")
    
    if not isinstance(y_train, np.ndarray):
        try:
            y_train = np.array(y_train)
        except Exception as e:
            raise ValueError(f"Cannot convert y_train to numpy array: {str(e)}")
    
    # Check shapes
    if len(y_train.shape) != 2 or y_train.shape[1] != 6:
        raise ValueError(f"y_train must have shape [n_samples, 6], got {y_train.shape}")
    
    # Initialize scaler
    if scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Scale features
    X_scaled = scaler.fit_transform(X_train)
    
    # Train separate model for each number position
    models = []
    best_params_list = []
    
    for i in range(6):
        logging.info(f"Training linear model for number position {i+1}/6")
        
        try:
            # Set up hyperparameter tuning if requested
            if tune_hyperparams:
                try:
                    logging.info(f"Tuning hyperparameters for {model_type} model")
                    study = optuna.create_study(direction='maximize')
                    study.optimize(
                        lambda trial: objective(trial, X_scaled, y_train[:, i], model_type), 
                        n_trials=n_trials
                    )
                    best_params = study.best_params
                    best_params_list.append(best_params)
                    logging.info(f"Best parameters for model {i+1}: {best_params}")
                except Exception as e:
                    logging.error(f"Error during hyperparameter tuning: {str(e)}")
                    logging.info("Using default parameters due to tuning failure")
                    best_params = {}
            else:
                best_params = {}
            
            # Create and train model
            model = get_linear_model(model_type, best_params)
            model.fit(X_scaled, y_train[:, i])
            
            # Evaluate model on training data
            train_score = model.score(X_scaled, y_train[:, i])
            logging.info(f"Model {i+1} training R² score: {train_score:.4f}")
            
            models.append(model)
            
        except Exception as e:
            logging.error(f"Error training model for position {i+1}: {str(e)}")
            # Fallback to basic model
            model = LinearRegression()
            model.fit(X_scaled, y_train[:, i])
            models.append(model)
    
    return models, scaler

def predict_linear_models(models: Union[Tuple[Any, Any], List[Any], Any], 
                         X: np.ndarray) -> List[int]:
    """
    Generate predictions using trained linear regression models
    
    Args:
        models: Either a tuple of (models_list, scaler) or just the models_list
                If a single model is provided, it will be used for all positions
        X: Input features for prediction
        
    Returns:
        Array of 6 valid lottery numbers
    """
    try:
        # Handle different input formats
        if isinstance(models, tuple):
            if len(models) == 2:
                models_list, scaler = models
            else:
                raise ValueError("Model tuple must have 2 elements (models_list, scaler)")
        else:
            raise ValueError("Models must be a tuple of (models_list, scaler)")
        
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Generate predictions for each number position
        predictions = np.zeros((X.shape[0], 6))
        
        # Check if models_list is a list or a single model
        if isinstance(models_list, list):
            for i, model in enumerate(models_list):
                predictions[:, i] = model.predict(X_scaled)
        else:
            # If a single model is provided, use it for all positions
            logging.warning("Single model provided; using it for all positions")
            for i in range(6):
                predictions[:, i] = models_list.predict(X_scaled)
        
        # Validate predictions
        if predictions.shape[0] == 1:  # Single prediction
            return ensure_valid_prediction(predictions[0])
            
        # For multiple predictions, ensure each row is valid
        valid_predictions = []
        for i in range(predictions.shape[0]):
            valid_predictions.append(ensure_valid_prediction(predictions[i]))
        
        return valid_predictions
        
    except Exception as e:
        logging.error(f"Error in linear model prediction: {str(e)}")
        # Return random valid numbers if prediction fails
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 