import numpy as np
import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import optuna
from typing import List, Dict, Tuple, Any, Union
from .utils import log_training_errors, ensure_valid_prediction

def objective(trial, X, y, cv=3):
    """
    Optuna objective function for Gradient Boosting parameter tuning
    
    Args:
        trial: Optuna trial object
        X: Training features
        y: Target variable
        cv: Number of cross-validation folds
        
    Returns:
        Mean cross-validation score
    """
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.mean(scores)

@log_training_errors
def train_gradient_boosting_model(X_train, y_train, params=None, tune_hyperparams=True, n_trials=50):
    """
    Train Gradient Boosting Regressor models for lottery number prediction
    
    Args:
        X_train: Training features
        y_train: Target numbers (array of shape [n_samples, 6])
        params: Optional predefined parameters
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter tuning trials
        
    Returns:
        Tuple of (list of trained models, scaler)
    """
    # Validate input
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    
    # Check shapes
    if len(y_train.shape) != 2 or y_train.shape[1] != 6:
        raise ValueError(f"y_train must have shape [n_samples, 6], got {y_train.shape}")
    
    # Initialize scaler and scale input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train model for each number position
    models = []
    best_params_list = []
    
    for i in range(6):
        # Set up hyperparameter tuning if requested
        if tune_hyperparams and params is None:
            logging.info(f"Tuning hyperparameters for Gradient Boosting model {i+1}/6")
            try:
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_scaled, y_train[:, i]), n_trials=n_trials)
                best_params = study.best_params
                best_params_list.append(best_params)
                logging.info(f"Best parameters for model {i+1}: {best_params}")
            except Exception as e:
                logging.error(f"Error during hyperparameter tuning: {str(e)}")
                logging.info("Using default parameters due to tuning failure")
                best_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'subsample': 0.8
                }
        else:
            # Use provided params or defaults
            best_params = params or {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 0.8
            }
        
        # Train model with best parameters
        try:
            model = GradientBoostingRegressor(**best_params, random_state=42)
            model.fit(X_scaled, y_train[:, i])
            models.append(model)
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                top_features = np.argsort(model.feature_importances_)[-5:]
                logging.info(f"Top features for model {i+1}: {top_features}")
        except Exception as e:
            logging.error(f"Error training model {i+1}: {str(e)}")
            raise
    
    return models, scaler

def predict_gradient_boosting_model(models, scaler, X):
    """
    Generate predictions using trained Gradient Boosting models
    
    Args:
        models: List of trained Gradient Boosting models
        scaler: Fitted StandardScaler
        X: Input features for prediction
        
    Returns:
        Array of predicted numbers of shape [n_samples, 6]
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Scale input
        X_scaled = scaler.transform(X)
        
        # Generate predictions
        predictions = np.zeros((X.shape[0], 6))
        for i, model in enumerate(models):
            predictions[:, i] = model.predict(X_scaled)
        
        # Validate predictions
        if predictions.shape[0] == 1:  # Single prediction
            return ensure_valid_prediction(predictions[0])
            
        # For multiple predictions, ensure each row is valid
        valid_predictions = []
        for i in range(predictions.shape[0]):
            valid_predictions.append(ensure_valid_prediction(predictions[i]))
        
        return np.array(valid_predictions)
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 