import numpy as np
import logging
from catboost import CatBoostRegressor, Pool, cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import optuna
from typing import List, Dict, Tuple, Any, Union, Optional
from .utils import log_training_errors, ensure_valid_prediction

def preprocess_categorical_features(X, cat_features):
    """
    Preprocess categorical features to ensure they are strings
    
    Args:
        X: Features array
        cat_features: List of categorical feature indices
        
    Returns:
        X with categorical features converted to strings
    """
    X_processed = X.copy()
    if cat_features:
        for cat_idx in cat_features:
            # Convert to integer first to avoid float strings
            X_processed[:, cat_idx] = X_processed[:, cat_idx].astype(int).astype(str)
    return X_processed

def objective(trial, X, y, cat_features=None, cv_folds=3) -> float:
    """
    Optuna objective function for CatBoost parameter tuning
    
    Args:
        trial: Optuna trial object
        X: Training features
        y: Target variable
        cat_features: Indices of categorical features
        cv_folds: Number of cross-validation folds
        
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
        'verbose': 0
    }
    
    # Preprocess categorical features
    X_processed = preprocess_categorical_features(X, cat_features)
    
    # Create dataset with categorical features
    train_data = Pool(data=X_processed, label=y, cat_features=cat_features)
    
    # Use catboost's built-in CV function
    cv_results = cv(
        train_data,
        params,
        fold_count=cv_folds,
        early_stopping_rounds=20,
        stratified=False,
        partition_random_seed=42
    )
    
    # Get the best validation score (lower RMSE is better)
    best_score = min(cv_results['test-RMSE-mean'])
    
    return -float(best_score)  # Negative because we want to maximize

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
def train_catboost_model(X_train: np.ndarray, y_train: np.ndarray, cat_features: Optional[List[int]] = None, params: Optional[Dict] = None) -> Tuple[List[CatBoostRegressor], StandardScaler]:
    """
    Train CatBoost models for lottery number prediction
    
    Args:
        X_train: Training features array of shape [n_samples, n_features]
        y_train: Target values array of shape [n_samples, 6]
        cat_features: Optional list of indices indicating categorical features
        params: Optional dictionary of model parameters:
            - iterations: Number of boosting iterations (default: 1000)
            - learning_rate: Learning rate (default: 0.03)
            - depth: Tree depth (default: 6)
            - l2_leaf_reg: L2 regularization (default: 3.0)
            - verbose: Verbosity during training (default: False)
            
    Returns:
        Tuple of (trained models list, fitted scaler)
    """
    try:
        # Validate input shapes
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match")
        if y_train.shape[1] != 6:
            raise ValueError("y_train must have 6 target columns")
            
        # Set default params if not provided
        if params is None:
            params = {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'verbose': False
            }
            
        # Convert numeric features to float and keep categorical features as strings
        X_train_processed = X_train.copy()
        if cat_features:
            non_cat_mask = np.ones(X_train.shape[1], dtype=bool)
            non_cat_mask[cat_features] = False
            X_train_processed[:, non_cat_mask] = X_train_processed[:, non_cat_mask].astype(float)
        else:
            X_train_processed = X_train_processed.astype(float)
            
        # Scale features except categorical ones
        scaler = StandardScaler()
        if cat_features:
            # Save categorical features
            cat_values = {i: X_train_processed[:, i].copy() for i in cat_features}
            # Scale only non-categorical features
            X_train_processed[:, non_cat_mask] = scaler.fit_transform(X_train_processed[:, non_cat_mask])
            # Restore categorical features
            for i, values in cat_values.items():
                X_train_processed[:, i] = values
        else:
            X_train_processed = scaler.fit_transform(X_train_processed)
        
        # Train a model for each target
        models = []
        for i in range(6):
            model = CatBoostRegressor(
                iterations=params['iterations'],
                learning_rate=params['learning_rate'],
                depth=params['depth'],
                l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
                verbose=params.get('verbose', False)
            )
            model.fit(X_train_processed, y_train[:, i], cat_features=cat_features)
            models.append(model)
            
        return models, scaler
        
    except Exception as e:
        logging.error(f"Error in CatBoost training: {str(e)}")
        raise

def predict_catboost_model(model, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions using trained CatBoost models
    
    Args:
        model: List of trained CatBoost models (one for each number position)
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

class CatBoostModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = None
        self.scaler = None
        self.is_trained = False
        self.cat_features = None
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the CatBoost model"""
        # Detect categorical features if not specified
        if 'cat_features' not in self.config:
            self.cat_features = detect_categorical_features(X)
        else:
            self.cat_features = self.config['cat_features']
            
        self.models, self.scaler = train_catboost_model(
            X, 
            y, 
            cat_features=self.cat_features,
            params=self.config
        )
        self.is_trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return predict_catboost_model(self.models, X)
        
    def save(self, path: str) -> None:
        """Save the model to disk"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        import joblib
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'config': self.config,
            'cat_features': self.cat_features
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'CatBoostModel':
        """Load a saved model from disk"""
        import joblib
        data = joblib.load(path)
        model = cls(data['config'])
        model.models = data['models']
        model.scaler = data['scaler']
        model.cat_features = data['cat_features']
        model.is_trained = True
        return model 