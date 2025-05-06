import numpy as np
import logging
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import optuna
from typing import List, Dict, Tuple, Any, Union
from .utils import log_training_errors, ensure_valid_prediction
from sklearn.metrics import mean_squared_error

def objective(trial, X, y):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'objective': 'regression',
        'metric': 'mse',
        'verbosity': -1
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10)],
        eval_metric='mse'
    )
    
    val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    return mse

def train_lightgbm_model(X, y, n_trials=20, params=None):
    """Train LightGBM models for lottery number prediction.

    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        n_trials (int, optional): Number of trials for hyperparameter tuning. Defaults to 20.
        params (dict, optional): Override default parameters. Defaults to None.

    Returns:
        tuple: (list of trained models, StandardScaler)
    """
    try:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Inputs X and y must be numpy arrays")

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")

        if len(y.shape) != 2 or y.shape[1] != 6:
            raise ValueError("Target y must have shape (n_samples, 6)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = []
        for i in range(6):
            y_train_single = y[:, i]
            study = optuna.create_study(direction='minimize')
            try:
                study.optimize(lambda trial: objective(trial, X_scaled, y_train_single), n_trials=n_trials)
                best_params = study.best_params
                if params:
                    best_params.update(params)
                best_params.update({
                    'objective': 'regression',
                    'metric': 'mse',
                    'verbosity': -1
                })
                
                # Train final model with best parameters and validation
                X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_train_single, test_size=0.2)
                model = lgb.LGBMRegressor(**best_params)
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10)],
                    eval_metric='mse'
                )
                
                # Convert to Booster for compatibility
                model = model.booster_
                models.append(model)
            except Exception as e:
                logging.error(f"Error during hyperparameter tuning: {str(e)}")
                raise

        return models, scaler
    except Exception as e:
        logging.error(f"Error in LightGBM training: {str(e)}")
        raise

def predict_lightgbm_model(model, X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Generate predictions using trained LightGBM models
    
    Args:
        model: List of trained LightGBM models (one for each number position)
        X: Input features to predict on
        scaler: Fitted StandardScaler
        
    Returns:
        Array of predicted numbers
    """
    # Validate input
    if not isinstance(X, np.ndarray):
        raise ValueError("Input X must be a numpy array")
    
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

class LightGBMModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LightGBM model"""
        self.models, self.scaler = train_lightgbm_model(
            X, 
            y, 
            n_trials=self.config.get('n_trials', 20),
            params=self.config.get('params', None)
        )
        self.is_trained = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return predict_lightgbm_model(self.models, X, self.scaler)
        
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
    def load(cls, path: str) -> 'LightGBMModel':
        """Load a saved model from disk"""
        import joblib
        data = joblib.load(path)
        model = cls(data['config'])
        model.models = data['models']
        model.scaler = data['scaler']
        model.is_trained = True
        return model 