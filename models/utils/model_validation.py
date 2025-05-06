"""Model validation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .model_metrics import ModelMetrics

logger = logging.getLogger(__name__)

def validate_model_performance(model: Any, X: np.ndarray, y: np.ndarray,
                             feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Validate model performance using standard metrics.
    
    Args:
        model: Model to validate
        X: Input features
        y: Target values
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = ModelMetrics()
    y_pred = model.predict(X)
    return metrics.calculate_metrics(y, y_pred)

class ModelValidator:
    """Class for validating models."""
    
    def __init__(self, n_splits: int = 5, test_size: int = 1):
        """Initialize ModelValidator.
        
        Args:
            n_splits: Number of splits for cross-validation
            test_size: Size of test set for each split
        """
        self.n_splits = n_splits
        self.test_size = test_size
        # Initialize with dummy data that will be updated during validation
        self.metrics = ModelMetrics(
            y_true=np.zeros((1, 6)),  # Dummy data
            y_pred=np.zeros((1, 6)),  # Dummy data
            task='regression'
        )
        
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Perform time series cross-validation.
        
        Args:
            model: Model to validate
            X: Input features
            y: Target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        fold_metrics = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.metrics.calculate_metrics(y_test, y_pred)
            fold_metrics.append(metrics)
            
        # Aggregate metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in fold_metrics])
            
        return avg_metrics
        
    def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive model validation.
        
        Args:
            model: Model to validate
            X: Input features
            y: Target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of validation results
        """
        # Cross-validation
        cv_metrics = self.cross_validate(model, X, y, feature_names)
        
        # Time series metrics
        y_pred = model.predict(X)
        ts_metrics = self.metrics.calculate_time_series_metrics(y, y_pred)
        
        # Combine metrics
        validation_results = {
            'cross_validation': cv_metrics,
            'time_series': ts_metrics,
            'overall': {**cv_metrics, **ts_metrics}
        }
        
        return validation_results
        
    def validate_ensemble(self, ensemble: Any, X: np.ndarray, y: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate ensemble model.
        
        Args:
            ensemble: Ensemble model to validate
            X: Input features
            y: Target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of validation results
        """
        # Validate overall ensemble
        ensemble_results = self.validate_model(ensemble, X, y, feature_names)
        
        # Validate individual models
        individual_results = {}
        for name, model in ensemble.models.items():
            individual_results[name] = self.validate_model(model, X, y, feature_names)
            
        return {
            'ensemble': ensemble_results,
            'individual_models': individual_results
        } 