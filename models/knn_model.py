import numpy as np
import logging
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import optuna
from typing import List, Dict, Tuple, Any, Union, Optional
from .utils import log_training_errors, ensure_valid_prediction

def find_optimal_k(X_train, y_train, cv=5, k_range=None):
    """
    Find the optimal value of k for KNN using cross-validation
    
    Args:
        X_train: Training features
        y_train: Target variable
        cv: Number of cross-validation folds
        k_range: Range of k values to test (default: [3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
        
    Returns:
        Optimal value of k
    """
    if k_range is None:
        k_range = list(range(3, 22, 2))  # Odd numbers from 3 to 21
    
    best_score = -np.inf
    best_k = 5  # Default
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for k in k_range:
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            model = KNeighborsRegressor(n_neighbors=k, weights='distance')
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)
        
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    
    return best_k

def objective(trial, X, y, cv=5):
    """
    Optuna objective function for KNN parameter tuning
    
    Args:
        trial: Optuna trial object
        X: Training features
        y: Target variable
        cv: Number of cross-validation folds
        
    Returns:
        Mean cross-validation score
    """
    # Define the hyperparameter search space
    k = trial.suggest_int('n_neighbors', 3, 25)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)  # 1 = Manhattan, 2 = Euclidean
    leaf_size = trial.suggest_int('leaf_size', 10, 100)
    
    # Cross-validation setup
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = KNeighborsRegressor(
            n_neighbors=k,
            weights=weights,
            p=p,
            leaf_size=leaf_size,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.mean(scores)

@log_training_errors
def train_knn_model(X_train, y_train, params=None, tune_hyperparams=True, 
                   n_trials=50, weights='distance', n_neighbors=None):
    """
    Train KNN Regressor models for lottery number prediction
    
    Args:
        X_train: Training features
        y_train: Target numbers (array of shape [n_samples, 6])
        params: Optional predefined parameters
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter tuning trials
        weights: Weight function ('uniform' or 'distance')
        n_neighbors: Number of neighbors (auto-tuned if None)
        
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
        logging.info(f"Training KNN model for number position {i+1}/6")
        
        # Set up hyperparameter tuning if requested
        if tune_hyperparams and params is None:
            try:
                if n_neighbors is None:
                    # Simple approach: just find the optimal k
                    best_k = find_optimal_k(X_scaled, y_train[:, i])
                    best_params = {
                        'n_neighbors': best_k,
                        'weights': weights,
                        'p': 2,  # Euclidean distance
                        'leaf_size': 30,
                        'n_jobs': -1
                    }
                    logging.info(f"Optimal k for position {i+1}: {best_k}")
                else:
                    # Comprehensive tuning with Optuna
                    study = optuna.create_study(direction='maximize')
                    study.optimize(
                        lambda trial: objective(trial, X_scaled, y_train[:, i]), 
                        n_trials=n_trials
                    )
                    best_params = study.best_params
                    best_params['n_jobs'] = -1  # Use all available cores
                    
                best_params_list.append(best_params)
                logging.info(f"Best parameters for model {i+1}: {best_params}")
                
            except Exception as e:
                logging.error(f"Error during hyperparameter tuning: {str(e)}")
                logging.info("Using default parameters due to tuning failure")
                best_params = {
                    'n_neighbors': 5 if n_neighbors is None else n_neighbors,
                    'weights': weights,
                    'p': 2,
                    'leaf_size': 30,
                    'n_jobs': -1
                }
        else:
            # Use provided params or defaults
            best_params = params or {
                'n_neighbors': 5 if n_neighbors is None else n_neighbors,
                'weights': weights,
                'p': 2,
                'leaf_size': 30,
                'n_jobs': -1
            }
        
        # Train model with best parameters
        try:
            model = KNeighborsRegressor(**best_params)
            model.fit(X_scaled, y_train[:, i])
            models.append(model)
            
            # Evaluate on training data
            train_score = model.score(X_scaled, y_train[:, i])
            logging.info(f"Model {i+1} training R² score: {train_score:.4f}")
            
        except Exception as e:
            logging.error(f"Error training model {i+1}: {str(e)}")
            raise
    
    return models, scaler

def predict_knn_model(models, scaler, X):
    """
    Generate predictions using trained KNN models
    
    Args:
        models: List of trained KNN models
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
        logging.error(f"Error in KNN prediction: {str(e)}")
        # Return random valid prediction as fallback
        return ensure_valid_prediction(np.random.randint(1, 60, size=6)) 