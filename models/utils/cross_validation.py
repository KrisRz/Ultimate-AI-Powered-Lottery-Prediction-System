"""Cross validation utilities for model evaluation."""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)

def monte_carlo_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Perform Monte Carlo cross-validation.
    
    Args:
        X: Input features
        y: Target values
        model_fn: Function that returns a new model instance
        n_splits: Number of splits
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with validation metrics
    """
    np.random.seed(random_state)
    metrics = {'mse': [], 'mae': [], 'r2': []}
    
    for i in range(n_splits):
        # Random split indices
        indices = np.random.permutation(len(X))
        test_size_int = int(len(X) * test_size)
        test_indices = indices[:test_size_int]
        train_indices = indices[test_size_int:]
        
        # Split data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train and evaluate model
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        
        logger.info(f"Split {i+1}/{n_splits}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return metrics

def rolling_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    n_splits: int = 5,
    test_size: int = None
) -> Dict[str, List[float]]:
    """
    Perform rolling-window cross-validation for time series data.
    
    Args:
        X: Input features
        y: Target values
        model_fn: Function that returns a new model instance
        n_splits: Number of splits
        test_size: Size of test set (if None, calculated from n_splits)
        
    Returns:
        Dictionary with validation metrics
    """
    if test_size is None:
        test_size = len(X) // (n_splits + 1)
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    metrics = {'mse': [], 'mae': [], 'r2': []}
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and evaluate model
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        metrics['mse'].append(mse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        
        logger.info(f"Split {i+1}/{n_splits}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return metrics 