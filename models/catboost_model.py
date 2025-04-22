import numpy as np
import logging
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
import optuna
from typing import List, Dict, Tuple, Any, Union, Optional
from .utils import log_training_errors, ensure_valid_prediction

def objective(trial, X, y, cat_features=None, cv=3):
    """
    Optuna objective function for CatBoost parameter tuning
    
    Args:
        trial: Optuna trial object
        X: Training features
        y: Target variable
        cat_features: Indices of categorical features
        cv: Number of cross-validation folds
        
    Returns:
        Mean validation score
    """
    # Define the hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False
    }
    
    # Prepare cross-validation
    scores = []
    
    # Create dataset with categorical features
    train_data = Pool(data=X, label=y, cat_features=cat_features)
    
    # Cross-validation with CatBoost's built-in function
    cv_results = CatBoostRegressor(**params).cv(
        train_data,
        fold_count=cv,
        early_stopping_rounds=20,
        stratified=False,
        partition_random_seed=42
    )
    
    # Get the best validation score (lower RMSE is better)
    best_score = min(cv_results['test-RMSE-mean'])
    
    return -best_score  # Negative because we want to maximize

def detect_categorical_features(X_train, column_names=None) -> List[int]:
    """
    Detect categorical features in the dataset
    
    Args:
        X_train: Training data
        column_names: Optional list of column names
        
    Returns:
        List of indices of categorical features
    """
    categorical_indices = []
    
    # If we have column names, use them to identify categorical columns
    if column_names:
        categorical_keywords = [
            'day', 'month', 'year', 'week', 'category', 'type', 'id', 'code',
            'dayofweek', 'day_of_week', 'weekday', 'weekend'
        ]
        
        for i, col_name in enumerate(column_names):
            col_lower = col_name.lower()
            # Check if the column name contains categorical keywords
            if any(keyword in col_lower for keyword in categorical_keywords):
                # Further verify by checking the number of unique values
                unique_count = len(np.unique(X_train[:, i]))
                if unique_count < 50:  # Fewer than 50 unique values suggest categorical
                    categorical_indices.append(i)
    else:
        # Without column names, use heuristics
        for i in range(X_train.shape[1]):
            unique_values = np.unique(X_train[:, i])
            # Features with few unique values that are integers are likely categorical
            if len(unique_values) < 20 and np.all(np.mod(unique_values, 1) == 0):
                categorical_indices.append(i)
    
    return categorical_indices

@log_training_errors
def train_catboost_model(X_train, y_train, params=None, tune_hyperparams=True, n_trials=50, 
                         cat_features=None, column_names=None):
    """
    Train CatBoost Regressor models for lottery number prediction
    
    Args:
        X_train: Training features
        y_train: Target numbers (array of shape [n_samples, 6])
        params: Optional predefined parameters
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter tuning trials
        cat_features: Indices of categorical features (auto-detected if None)
        column_names: Names of feature columns for better categorical detection
        
    Returns:
        Tuple of (list of trained models, scaler, cat_features)
    """
    # Validate input
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    
    # Check shapes
    if len(y_train.shape) != 2 or y_train.shape[1] != 6:
        raise ValueError(f"y_train must have shape [n_samples, 6], got {y_train.shape}")
    
    # Initialize scaler and scale input data (exclude categorical features from scaling)
    scaler = StandardScaler()
    X_scaled = X_train.copy()
    
    # Detect categorical features if not provided
    if cat_features is None:
        cat_features = detect_categorical_features(X_train, column_names)
        logging.info(f"Detected categorical features at indices: {cat_features}")
    
    # Scale non-categorical features
    non_cat_indices = [i for i in range(X_train.shape[1]) if i not in cat_features]
    if non_cat_indices:
        X_scaled[:, non_cat_indices] = scaler.fit_transform(X_train[:, non_cat_indices])
    
    # Train model for each number position
    models = []
    best_params_list = []
    feature_importance_list = []
    
    for i in range(6):
        # Set up hyperparameter tuning if requested
        if tune_hyperparams and params is None:
            logging.info(f"Tuning hyperparameters for CatBoost model {i+1}/6")
            try:
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: objective(trial, X_scaled, y_train[:, i], cat_features, cv=3), 
                               n_trials=n_trials)
                best_params = study.best_params
                best_params['verbose'] = False  # Ensure quiet training
                best_params_list.append(best_params)
                logging.info(f"Best parameters for model {i+1}: {best_params}")
            except Exception as e:
                logging.error(f"Error during hyperparameter tuning: {str(e)}")
                logging.info("Using default parameters due to tuning failure")
                best_params = {
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3.0,
                    'verbose': False
                }
        else:
            # Use provided params or defaults
            best_params = params or {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'verbose': False
            }
        
        # Train model with best parameters
        try:
            model = CatBoostRegressor(**best_params)
            
            # Create proper dataset with categorical features
            train_pool = Pool(data=X_scaled, label=y_train[:, i], cat_features=cat_features)
            
            # Train the model
            model.fit(train_pool, plot=False)
            models.append(model)
            
            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:]  # Top 10 features
                
                # Create feature importance dict with names if available
                if column_names:
                    top_features = {column_names[idx]: float(importances[idx]) for idx in indices}
                else:
                    top_features = {f"feature_{idx}": float(importances[idx]) for idx in indices}
                
                feature_importance_list.append(top_features)
                logging.info(f"Top features for model {i+1}: {top_features}")
                
                # Check if categorical features are important
                cat_importance = [(idx, float(importances[idx])) for idx in cat_features if idx in indices]
                if cat_importance:
                    logging.info(f"Important categorical features: {cat_importance}")
                
        except Exception as e:
            logging.error(f"Error training model {i+1}: {str(e)}")
            raise
    
    # Store feature importance for analysis
    if feature_importance_list:
        try:
            import json
            with open("logs/catboost_feature_importance.json", "w") as f:
                json.dump(feature_importance_list, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not save feature importance: {str(e)}")
    
    return models, scaler, cat_features

def predict_catboost_model(models, scaler, X, cat_features=None):
    """
    Generate predictions using trained CatBoost models
    
    Args:
        models: List of trained CatBoost models
        scaler: Fitted StandardScaler for non-categorical features
        X: Input features for prediction
        cat_features: Indices of categorical features
        
    Returns:
        Array of predicted numbers of shape [n_samples, 6]
    """
    try:
        # Validate input
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Scale non-categorical features
        X_scaled = X.copy()
        if cat_features is None:
            cat_features = []
            
        non_cat_indices = [i for i in range(X.shape[1]) if i not in cat_features]
        if non_cat_indices:
            X_scaled[:, non_cat_indices] = scaler.transform(X[:, non_cat_indices])
        
        # Generate predictions
        predictions = np.zeros((X.shape[0], 6))
        
        for i, model in enumerate(models):
            # Create proper prediction pool with categorical features
            pred_pool = Pool(data=X_scaled, cat_features=cat_features)
            predictions[:, i] = model.predict(pred_pool)
        
        # Validate predictions
        if predictions.shape[0] == 1:  # Single prediction
            return ensure_valid_prediction(predictions[0])
            
        # For multiple predictions, ensure each row is valid
        valid_predictions = []
        for i in range(predictions.shape[0]):
            valid_predictions.append(ensure_valid_prediction(predictions[i]))
        
        return np.array(valid_predictions)
        
    except Exception as e:
        logging.error(f"Error in CatBoost prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 