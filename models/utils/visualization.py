"""Model visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def plot_training_history(history: Dict[str, List[float]], title: str = "Training History") -> None:
    """Plot training and validation metrics over epochs."""
    fig = make_subplots(rows=1, cols=1)
    
    for metric in history.keys():
        if 'val_' in metric:
            continue
        fig.add_trace(
            go.Scatter(
                y=history[metric],
                name=metric,
                mode='lines'
            ),
            row=1, col=1
        )
        if f'val_{metric}' in history:
            fig.add_trace(
                go.Scatter(
                    y=history[f'val_{metric}'],
                    name=f'val_{metric}',
                    mode='lines'
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Value",
        showlegend=True
    )
    fig.show()

def plot_feature_importance(model: Any, feature_names: List[str], title: str = "Feature Importance") -> None:
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model does not have feature importance attribute")
        return
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=importances,
            name="Feature Importance"
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Importance",
        showlegend=False
    )
    fig.show()

def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Prediction Distribution") -> None:
    """Plot distribution of true vs predicted values."""
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Histogram(
            x=y_true,
            name="True Values",
            opacity=0.75
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=y_pred,
            name="Predicted Values",
            opacity=0.75
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        barmode='overlay'
    )
    fig.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residuals Plot") -> None:
    """Plot residuals against predicted values."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name="Residuals"
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        showlegend=False
    )
    fig.show()

def plot_learning_curves(model: Any, X: np.ndarray, y: np.ndarray, 
                        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10)) -> None:
    """Plot learning curves for model performance vs training size."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean,
            name="Training Score",
            line=dict(color="blue")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=val_mean,
            name="Validation Score",
            line=dict(color="red")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean + train_std,
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean - train_std,
            line=dict(width=0),
            fill='tonexty',
            showlegend=False
        )
    )
    
    fig.update_layout(
        title="Learning Curves",
        xaxis_title="Training Examples",
        yaxis_title="Score",
        showlegend=True
    )
    fig.show()

def plot_model_comparison(models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> None:
    """Plot comparison of different models' performance."""
    scores = {}
    for name, model in models.items():
        scores[name] = model.score(X, y)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            name="Model Scores"
        )
    )
    
    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        showlegend=False
    )
    fig.show()

def plot_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray, 
                            confidence: float = 0.95) -> None:
    """Plot prediction confidence intervals."""
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    
    z_score = 1.96  # For 95% confidence
    upper_bound = y_pred + z_score * std_residuals
    lower_bound = y_pred - z_score * std_residuals
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(y_true)),
            y=y_true,
            name="True Values",
            mode='markers'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(y_pred)),
            y=y_pred,
            name="Predictions",
            mode='lines'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(upper_bound)),
            y=upper_bound,
            name="Upper Bound",
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(lower_bound)),
            y=lower_bound,
            name="Lower Bound",
            line=dict(width=0),
            fill='tonexty',
            showlegend=False
        )
    )
    
    fig.update_layout(
        title="Prediction Confidence Intervals",
        xaxis_title="Sample",
        yaxis_title="Value",
        showlegend=True
    )
    fig.show()

class ModelVisualizer:
    """Class for visualizing model performance and predictions."""
    
    def __init__(self, model, feature_names=None):
        """Initialize ModelVisualizer.
        
        Args:
            model: The trained model to visualize
            feature_names: List of feature names for the model
        """
        self.model = model
        self.feature_names = feature_names
        
    def plot_training_history(self, history, title="Training History"):
        """Plot training metrics over epochs."""
        plot_training_history(history, title)
        
    def plot_feature_importance(self, title="Feature Importance"):
        """Plot feature importance if available."""
        if self.feature_names is None:
            raise ValueError("Feature names must be provided to plot feature importance")
        plot_feature_importance(self.model, self.feature_names, title)
        
    def plot_predictions(self, y_true, y_pred, title="Prediction Distribution"):
        """Plot distribution of true vs predicted values."""
        plot_prediction_distribution(y_true, y_pred, title)
        
    def plot_residuals(self, y_true, y_pred, title="Residuals Plot"):
        """Plot residuals against predicted values."""
        plot_residuals(y_true, y_pred, title)
        
    def plot_learning_curves(self, X, y, train_sizes=np.linspace(0.1, 1.0, 10)):
        """Plot learning curves showing model performance vs training size."""
        plot_learning_curves(self.model, X, y, train_sizes)
        
    def plot_confidence_intervals(self, y_true, y_pred, confidence=0.95):
        """Plot predictions with confidence intervals."""
        plot_confidence_intervals(y_true, y_pred, confidence)
        
    def plot_all_metrics(self, X, y, y_pred, history=None):
        """Plot all available visualization metrics."""
        if history:
            self.plot_training_history(history)
            
        if self.feature_names is not None:
            self.plot_feature_importance()
            
        self.plot_predictions(y, y_pred)
        self.plot_residuals(y, y_pred)
        self.plot_learning_curves(X, y)
        self.plot_confidence_intervals(y, y_pred) 