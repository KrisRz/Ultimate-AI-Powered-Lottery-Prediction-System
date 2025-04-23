import numpy as np
import logging
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import optuna
from typing import List, Dict, Tuple, Any, Union
from .utils import log_training_errors, ensure_valid_prediction

def objective(trial, X, y, cv=3) -> float:
    """
    Optuna objective function for LightGBM parameter tuning
    
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
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'verbosity': -1
    }
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(**params, random_state=42)
        # Create early stopping callback
        callbacks = [lgb.early_stopping(stopping_rounds=20)]
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='mse',
            callbacks=callbacks
        )
        
        # Evaluate on validation set
        pred = model.predict(X_val_fold)
        mse = np.mean((y_val_fold - pred) ** 2)
        scores.append(-mse)  # Negative because we want to maximize
    
    return float(np.mean(scores))

@log_training_errors
def train_lightgbm_model(X_train, y_train, params=None, tune_hyperparams=True, n_trials=50):
    """
    Train LightGBM Regressor models for lottery number prediction
    
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
    feature_importance_list = []
    
    for i in range(6):
        # Set up hyperparameter tuning if requested
        if tune_hyperparams and params is None:
            logging.info(f"Tuning hyperparameters for LightGBM model {i+1}/6")
            try:
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_scaled, y_train[:, i]), n_trials=n_trials)
                best_params = study.best_params
                
                # Add mandatory parameters
                best_params['objective'] = 'regression'
                best_params['metric'] = 'mse'
                best_params['verbosity'] = -1
                
                best_params_list.append(best_params)
                logging.info(f"Best parameters for model {i+1}: {best_params}")
            except Exception as e:
                logging.error(f"Error during hyperparameter tuning: {str(e)}")
                logging.info("Using default parameters due to tuning failure")
                best_params = {
                    'objective': 'regression',
                    'metric': 'mse',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'verbosity': -1
                }
        else:
            # Use provided params or defaults
            best_params = params or {
                'objective': 'regression',
                'metric': 'mse',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': 6,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbosity': -1
            }
        
        # Train model with best parameters
        try:
            model = lgb.LGBMRegressor(**best_params, random_state=42)
            
            # Use validation set if we have enough data
            if X_train.shape[0] > 1000:  # Only if we have enough data
                X_train_part, X_val_part = X_scaled[:-200], X_scaled[-200:]
                y_train_part, y_val_part = y_train[:-200, i], y_train[-200:, i]
                
                # Create early stopping callback
                callbacks = [lgb.early_stopping(stopping_rounds=20)]
                model.fit(
                    X_train_part, y_train_part,
                    eval_set=[(X_val_part, y_val_part)],
                    eval_metric='mse',
                    callbacks=callbacks
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
            raise
    
    # Store feature importance for potential analysis
    if feature_importance_list:
        try:
            import json
            with open("logs/lightgbm_feature_importance.json", "w") as f:
                json.dump(feature_importance_list, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save feature importance: {str(e)}")
    
    return models, scaler

def predict_lightgbm_model(models, scaler, X):
    """
    Generate predictions using trained LightGBM models
    
    Args:
        models: List of trained LightGBM models
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
        logging.error(f"Error in LightGBM prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 