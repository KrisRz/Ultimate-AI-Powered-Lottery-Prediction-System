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
def train_gradient_boosting_model(X_train, y_train, params=None):
    """
    Train Gradient Boosting models for lottery number prediction
    
    Args:
        X_train: Training features
        y_train: Target values of shape [n_samples, 6]
        params: Optional dictionary of model parameters
        
    Returns:
        Tuple of (list of trained models, fitted scaler)
    """
    try:
        # Validate input shapes
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
            
        if y_train.shape[1] != 6:
            raise ValueError(f"Expected y_train to have shape [n_samples, 6], got {y_train.shape}")
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Update with custom params if provided
        if params:
            default_params.update(params)
            
        # Train models for each target
        models = []
        for i in range(6):
            model = GradientBoostingRegressor(**default_params)
            model.fit(X_scaled, y_train[:, i])
            models.append(model)
            
        return models, scaler
    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        raise

def predict_gradient_boosting_model(model, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions using trained Gradient Boosting models
    
    Args:
        model: List of trained Gradient Boosting models (one for each number position)
        X: Input features to predict on
        
    Returns:
        Array of predicted numbers
    """
    # Validate input
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    # Make predictions for each number position
    predictions = []
    for i, m in enumerate(model):
        pred = m.predict(X)
        predictions.append(pred)
    
    # Stack predictions and round to integers
    predictions = np.round(np.column_stack(predictions)).astype(int)
    
    # Ensure predictions are within valid range
    predictions = np.clip(predictions, 1, 59)
    
    return predictions 

class GradientBoostingModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Gradient Boosting model"""
        self.models, self.scaler = train_gradient_boosting_model(X, y, params=self.config)
        self.is_trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return predict_gradient_boosting_model(self.models, X)
        
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
    def load(cls, path: str) -> 'GradientBoostingModel':
        """Load a saved model from disk"""
        import joblib
        data = joblib.load(path)
        model = cls(data['config'])
        model.models = data['models']
        model.scaler = data['scaler']
        model.is_trained = True
        return model 