"""Model visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from .model_metrics import ModelMetrics

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Class for visualizing model performance and predictions."""
    
    def __init__(self):
        """Initialize ModelVisualizer."""
        self.metrics = ModelMetrics()
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = 'Model Predictions') -> None:
        """Plot true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = 'Residuals') -> None:
        """Plot residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    def plot_time_series(self, y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = 'Time Series Predictions') -> None:
        """Plot time series predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='True Values', alpha=0.7)
        plt.plot(y_pred, label='Predictions', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                              title: str = 'Feature Importance') -> None:
        """Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            title: Plot title
        """
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features)
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    def plot_metrics_history(self, metrics_history: Dict[int, Dict[str, float]],
                           metric_name: str) -> None:
        """Plot history of a specific metric.
        
        Args:
            metrics_history: Dictionary of metrics history
            metric_name: Name of metric to plot
        """
        steps = list(metrics_history.keys())
        values = [metrics[metric_name] for metrics in metrics_history.values()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker='o')
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} History')
        plt.grid(True)
        plt.show()
        
    def plot_ensemble_predictions(self, ensemble: Any, X: np.ndarray, y: np.ndarray,
                                title: str = 'Ensemble Predictions') -> None:
        """Plot predictions from ensemble and individual models.
        
        Args:
            ensemble: Ensemble model
            X: Input features
            y: Target values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y, label='True Values', alpha=0.7)
        
        # Plot ensemble predictions
        y_pred_ensemble = ensemble.predict(X)
        plt.plot(y_pred_ensemble, label='Ensemble', alpha=0.7)
        
        # Plot individual model predictions
        for name, model in ensemble.models.items():
            y_pred = model.predict(X)
            plt.plot(y_pred, label=name, alpha=0.5)
            
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """Plot validation results.
        
        Args:
            validation_results: Dictionary of validation results
        """
        # Plot cross-validation metrics
        cv_metrics = validation_results['cross_validation']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(cv_metrics.keys()), y=list(cv_metrics.values()))
        plt.title('Cross-Validation Metrics')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
        # Plot time series metrics
        ts_metrics = validation_results['time_series']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(ts_metrics.keys()), y=list(ts_metrics.values()))
        plt.title('Time Series Metrics')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
        # Plot individual model metrics if available
        if 'individual_models' in validation_results:
            for name, metrics in validation_results['individual_models'].items():
                plt.figure(figsize=(10, 6))
                sns.barplot(x=list(metrics['overall'].keys()),
                           y=list(metrics['overall'].values()))
                plt.title(f'{name} Metrics')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.show() 