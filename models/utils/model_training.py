"""Model training utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.model_selection import train_test_split
from .model_metrics import ModelMetrics
from .model_validation import ModelValidator
from .model_visualization import ModelVisualizer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating models."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize ModelTrainer.
        
        Args:
            test_size: Size of test set
            random_state: Random seed
        """
        self.test_size = test_size
        self.random_state = random_state
        self.metrics = ModelMetrics()
        self.validator = ModelValidator()
        self.visualizer = ModelVisualizer()
        
    def train_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train and evaluate a model.
        
        Args:
            model: Model to train
            X: Input features
            y: Target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(y_test, y_pred)
        
        # Validate model
        validation_results = self.validator.validate_model(model, X, y, feature_names)
        
        # Visualize results
        self.visualizer.plot_predictions(y_test, y_pred)
        self.visualizer.plot_residuals(y_test, y_pred)
        self.visualizer.plot_validation_results(validation_results)
        
        if feature_names is not None and hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            self.visualizer.plot_feature_importance(feature_importance)
            
        return {
            'metrics': metrics,
            'validation': validation_results,
            'model': model
        }
        
    def train_ensemble(self, ensemble: Any, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train and evaluate an ensemble model.
        
        Args:
            ensemble: Ensemble model to train
            X: Input features
            y: Target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        y_pred = ensemble.predict(X_test)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(y_test, y_pred)
        
        # Validate ensemble
        validation_results = self.validator.validate_ensemble(ensemble, X, y, feature_names)
        
        # Visualize results
        self.visualizer.plot_predictions(y_test, y_pred)
        self.visualizer.plot_residuals(y_test, y_pred)
        self.visualizer.plot_ensemble_predictions(ensemble, X, y)
        self.visualizer.plot_validation_results(validation_results)
        
        if feature_names is not None:
            for name, model in ensemble.models.items():
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(feature_names, model.feature_importances_))
                    self.visualizer.plot_feature_importance(
                        feature_importance,
                        title=f'{name} Feature Importance'
                    )
                    
        return {
            'metrics': metrics,
            'validation': validation_results,
            'ensemble': ensemble
        }
        
    def train_time_series_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train and evaluate a time series model.
        
        Args:
            model: Model to train
            X: Input features
            y: Target values
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of training results
        """
        # Split data (preserving time order)
        train_size = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(y_test, y_pred)
        ts_metrics = self.metrics.calculate_time_series_metrics(y_test, y_pred)
        
        # Validate model
        validation_results = self.validator.validate_model(model, X, y, feature_names)
        
        # Visualize results
        self.visualizer.plot_time_series(y_test, y_pred)
        self.visualizer.plot_residuals(y_test, y_pred)
        self.visualizer.plot_validation_results(validation_results)
        
        if feature_names is not None and hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            self.visualizer.plot_feature_importance(feature_importance)
            
        return {
            'metrics': {**metrics, **ts_metrics},
            'validation': validation_results,
            'model': model
        } 