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
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
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
def train_xgboost_model(X, y, params=None):
    """
    Train XGBoost model for lottery prediction.
    
    Args:
        X: Input features
        y: Target values
        params: Model parameters
        
    Returns:
        Tuple of (model, scaler)
    """
    # Validate parameters
    if params:
        if 'learning_rate' in params and params['learning_rate'] <= 0:
            raise ValueError("Learning rate must be positive")
            
        if 'n_estimators' in params and params['n_estimators'] <= 0:
            raise ValueError("Number of estimators must be positive")
            
        if 'max_depth' in params and params['max_depth'] <= 0:
            raise ValueError("Max depth must be positive")
    
    # Use default parameters if none provided
    default_params = {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 6,
        'objective': 'reg:squarederror'
    }
    
    params = {**default_params, **(params or {})}
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model for each target
    models = []
    for i in range(y.shape[1]):
        model = xgb.XGBRegressor(**params)
        model.fit(X_scaled, y[:, i])
        models.append(model)
    
    return models, scaler

def predict_xgboost_model(model, X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Generate predictions using trained XGBoost models
    
    Args:
        model: List of trained XGBoost models (one for each number position)
        X: Input features to predict on
        scaler: Fitted StandardScaler
        
    Returns:
        Array of predicted numbers
    """
    # Validate input
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array")
    
    # Scale input features
    X_scaled = scaler.transform(X)
    
    # Make predictions for each number position
    predictions = []
    for i, m in enumerate(model):
        pred = m.predict(X_scaled)
        predictions.append(pred)
    
    # Stack predictions and round to integers
    predictions = np.round(np.column_stack(predictions)).astype(int)
    
    # Ensure predictions are within valid range
    predictions = np.clip(predictions, 1, 59)
    
    return predictions 

class XGBoostModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model"""
        self.models, self.scaler = train_xgboost_model(X, y, params=self.config)
        self.is_trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return predict_xgboost_model(self.models, X, self.scaler)
        
    def save(self, path: str) -> None:
        """Save the model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        import joblib
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'config': self.config
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        """Load a saved model from disk"""
        import joblib
        data = joblib.load(path)
        model = cls(data['config'])
        model.models = data['models']
        model.scaler = data['scaler']
        model.is_trained = True
        return model 