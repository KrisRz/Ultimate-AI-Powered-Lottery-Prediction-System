import numpy as np
import logging
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import optuna
from typing import List, Dict, Tuple, Any, Union, Optional
from .utils import log_training_errors, ensure_valid_prediction

def objective(trial, X, y, cv=5):
    """
    Optuna objective function for XGBoost parameter tuning
    
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
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
        'random_state': 42
    }
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = xgb.XGBRegressor(**params)
        
        try:
            # Train with early stopping
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Score on validation set
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)
        except Exception as e:
            # If training fails, return a poor score
            logging.warning(f"Error in XGBoost training during tuning: {str(e)}")
            scores.append(-999)
    
    # Return mean score, ignoring failed runs
    valid_scores = [s for s in scores if s > -999]
    if valid_scores:
        return np.mean(valid_scores)
    else:
        return -999  # All runs failed

@log_training_errors
def train_xgboost_model(X_train, y_train, params=None, tune_hyperparams=True, n_trials=50):
    """
    Train XGBoost Regressor models for lottery number prediction
    
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
    
    # Initialize scaler and scale input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train model for each number position
    models = []
    feature_importance_list = []
    best_params_list = []
    
    # Default parameters if not specified or tuning fails
    default_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    
    for i in range(6):
        logging.info(f"Training XGBoost model for number position {i+1}/6")
        
        # Set up hyperparameter tuning if requested
        if tune_hyperparams and params is None:
            try:
                logging.info(f"Tuning hyperparameters for XGBoost model {i+1}")
                study = optuna.create_study(direction='maximize')
                study.optimize(
                    lambda trial: objective(trial, X_scaled, y_train[:, i]), 
                    n_trials=n_trials,
                    catch=(Exception,)
                )
                tuned_params = study.best_params
                
                # Add required parameters not included in tuning
                tuned_params['objective'] = 'reg:squarederror'
                tuned_params['random_state'] = 42
                
                best_params = tuned_params
                best_params_list.append(best_params)
                logging.info(f"Best parameters for model {i+1}: {best_params}")
            except Exception as e:
                logging.error(f"Error during hyperparameter tuning: {str(e)}")
                logging.info("Using default parameters due to tuning failure")
                best_params = default_params.copy()
        else:
            # Use provided params or defaults
            best_params = params.copy() if params is not None else default_params.copy()
        
        # Train model with best parameters
        try:
            model = xgb.XGBRegressor(**best_params)
            
            # Split data for early stopping
            if X_train.shape[0] > 1000:  # Only if we have enough data
                train_idx = np.random.choice(X_scaled.shape[0], int(X_scaled.shape[0] * 0.8), replace=False)
                val_idx = np.array([i for i in range(X_scaled.shape[0]) if i not in train_idx])
                
                X_train_split = X_scaled[train_idx]
                y_train_split = y_train[train_idx, i]
                X_val_split = X_scaled[val_idx]
                y_val_split = y_train[val_idx, i]
                
                model.fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_scaled, y_train[:, i])
            
            models.append(model)
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:]  # Top 10 features
                top_features = {f"feature_{idx}": float(importances[idx]) for idx in indices}
                feature_importance_list.append(top_features)
                logging.info(f"Top features for model {i+1}: {top_features}")
                
        except Exception as e:
            logging.error(f"Error training model {i+1}: {str(e)}")
            # Fallback to default model
            try:
                logging.info("Trying fallback with default parameters")
                model = xgb.XGBRegressor(**default_params)
                model.fit(X_scaled, y_train[:, i])
                models.append(model)
            except Exception as e2:
                logging.error(f"Fallback training also failed: {str(e2)}")
                # Create a dummy model that returns the mean
                mean_value = np.mean(y_train[:, i])
                
                class DummyModel:
                    def predict(self, X):
                        return np.full(X.shape[0], mean_value)
                    
                models.append(DummyModel())
                logging.warning(f"Using dummy model that returns mean value {mean_value} for position {i+1}")
    
    # Store feature importance for analysis
    if feature_importance_list:
        try:
            import json
            with open("logs/xgboost_feature_importance.json", "w") as f:
                json.dump(feature_importance_list, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save feature importance: {str(e)}")
    
    return models, scaler

def predict_xgboost_model(models, scaler, X):
    """
    Generate predictions using trained XGBoost models
    
    Args:
        models: List of trained XGBoost models
        scaler: Fitted StandardScaler
        X: Input features for prediction
        
    Returns:
        Array of predicted numbers of shape [n_samples, 6]
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
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
        logging.error(f"Error in XGBoost prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 