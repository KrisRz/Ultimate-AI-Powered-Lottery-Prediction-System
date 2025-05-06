"""Model metrics utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    average_precision_score, log_loss, brier_score_loss,
    mean_squared_log_error, median_absolute_error, max_error
)
from scipy.stats import entropy, kendalltau, spearmanr, pearsonr
import pandas as pd
from .model_utils import set_random_seed

logger = logging.getLogger(__name__)

class ModelMetrics:
    """Class for calculating and tracking model metrics."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, task: str = 'classification'):
        """Initialize model metrics."""
        self.y_true = y_true
        self.y_pred = y_pred
        self.task = task
        
        # Set random seed for reproducibility
        set_random_seed(42)
        
        self.metrics_history = {}
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all relevant metrics for the task."""
        if self.task == 'classification':
            return self._calculate_classification_metrics()
        elif self.task == 'regression':
            return self._calculate_regression_metrics()
        else:
            raise ValueError(f"Unsupported task: {self.task}")
    
    def _calculate_classification_metrics(self) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='weighted')
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='weighted')
        metrics['f1'] = f1_score(self.y_true, self.y_pred, average='weighted')
        
        # Probability-based metrics
        if len(self.y_pred.shape) > 1 and self.y_pred.shape[1] > 1:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred, multi_class='ovr')
            metrics['log_loss'] = log_loss(self.y_true, self.y_pred)
            metrics['brier_score'] = brier_score_loss(self.y_true, self.y_pred)
        
        # Additional metrics
        metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred)
        metrics['classification_report'] = classification_report(self.y_true, self.y_pred, output_dict=True)
        
        return metrics
    
    def _calculate_regression_metrics(self) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(self.y_true, self.y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(self.y_true, self.y_pred)
        metrics['r2'] = r2_score(self.y_true, self.y_pred)
        metrics['explained_variance'] = explained_variance_score(self.y_true, self.y_pred)
        metrics['max_error'] = max_error(self.y_true, self.y_pred)
        
        # Additional metrics
        metrics['msle'] = mean_squared_log_error(self.y_true, self.y_pred)
        metrics['medae'] = median_absolute_error(self.y_true, self.y_pred)
        
        # Correlation metrics
        metrics['pearson_corr'] = pearsonr(self.y_true, self.y_pred)[0]
        metrics['spearman_corr'] = spearmanr(self.y_true, self.y_pred)[0]
        metrics['kendall_corr'] = kendalltau(self.y_true, self.y_pred)[0]
        
        return metrics
    
    def calculate_feature_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance."""
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
        else:
            # Use permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X, self.y_true, n_repeats=10)
            importances = result.importances_mean
        
        return dict(zip(range(len(importances)), importances))
    
    def calculate_prediction_uncertainty(self, model: Any, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate prediction uncertainty."""
        if isinstance(model, tf.keras.Model):
            # Monte Carlo Dropout
            predictions = []
            for _ in range(100):
                predictions.append(model.predict(X))
            predictions = np.array(predictions)
            
            return {
                'mean': np.mean(predictions, axis=0),
                'std': np.std(predictions, axis=0),
                'confidence_interval': np.percentile(predictions, [2.5, 97.5], axis=0)
            }
        else:
            # Bootstrap
            from sklearn.utils import resample
            predictions = []
            for _ in range(100):
                X_resampled, y_resampled = resample(X, self.y_true)
                model.fit(X_resampled, y_resampled)
                predictions.append(model.predict(X))
            predictions = np.array(predictions)
            
            return {
                'mean': np.mean(predictions, axis=0),
                'std': np.std(predictions, axis=0),
                'confidence_interval': np.percentile(predictions, [2.5, 97.5], axis=0)
            }
    
    def calculate_residuals(self) -> np.ndarray:
        """Calculate model residuals."""
        return self.y_true - self.y_pred
    
    def analyze_residuals(self) -> Dict[str, Any]:
        """Analyze model residuals."""
        residuals = self.calculate_residuals()
        
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'shapiro_test': self._shapiro_test(residuals),
            'durbin_watson': self._durbin_watson_test(residuals)
        }
    
    def _shapiro_test(self, residuals: np.ndarray) -> Dict[str, float]:
        """Perform Shapiro-Wilk test for normality."""
        from scipy.stats import shapiro
        stat, p_value = shapiro(residuals)
        return {'statistic': stat, 'p_value': p_value}
    
    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic."""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
    
    def calculate_calibration(self) -> Dict[str, Any]:
        """Calculate model calibration."""
        if self.task == 'classification':
            return self._calculate_classification_calibration()
        else:
            return self._calculate_regression_calibration()
    
    def _calculate_classification_calibration(self) -> Dict[str, Any]:
        """Calculate classification calibration."""
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_pred, n_bins=10)
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'brier_score': brier_score_loss(self.y_true, self.y_pred)
        }
    
    def _calculate_regression_calibration(self) -> Dict[str, Any]:
        """Calculate regression calibration."""
        residuals = self.calculate_residuals()
        quantiles = np.percentile(residuals, [25, 50, 75])
        
        return {
            'residual_quantiles': quantiles,
            'residual_std': np.std(residuals),
            'residual_mean': np.mean(residuals)
        }
    
    def calculate_feature_correlation(self, X: np.ndarray) -> pd.DataFrame:
        """Calculate feature correlations."""
        if self.task == 'classification':
            # Point-biserial correlation
            correlations = []
            for i in range(X.shape[1]):
                correlations.append(pearsonr(X[:, i], self.y_true)[0])
        else:
            # Pearson correlation
            correlations = []
            for i in range(X.shape[1]):
                correlations.append(pearsonr(X[:, i], self.y_true)[0])
        
        return pd.DataFrame({
            'feature': range(X.shape[1]),
            'correlation': correlations
        })
    
    def calculate_prediction_stability(self, model: Any, X: np.ndarray, n_samples: int = 100) -> Dict[str, float]:
        """Calculate prediction stability."""
        predictions = []
        for _ in range(n_samples):
            # Add small noise to input
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            predictions.append(model.predict(X_noisy))
        
        predictions = np.array(predictions)
        stability = np.std(predictions, axis=0)
        
        return {
            'mean_stability': np.mean(stability),
            'std_stability': np.std(stability),
            'max_stability': np.max(stability),
            'min_stability': np.min(stability)
        }
    
    def calculate_model_complexity(self, model: Any) -> Dict[str, int]:
        """Calculate model complexity."""
        if isinstance(model, tf.keras.Model):
            return {
                'num_layers': len(model.layers),
                'num_parameters': model.count_params(),
                'num_trainable_parameters': sum(
                    np.prod(w.shape) for w in model.trainable_weights
                )
            }
        else:
            return {
                'num_parameters': len(model.get_params()),
                'num_features': getattr(model, 'n_features_in_', 0)
            }
    
    def calculate_training_time(self, training_time: float) -> Dict[str, float]:
        """Calculate training time metrics."""
        return {
            'total_time': training_time,
            'time_per_sample': training_time / len(self.y_true),
            'time_per_epoch': training_time / getattr(self, 'num_epochs', 1)
        }
    
    def calculate_memory_usage(self, model: Any) -> Dict[str, float]:
        """Calculate model memory usage."""
        if isinstance(model, tf.keras.Model):
            memory = sum(
                np.prod(w.shape) * 4  # 4 bytes per float32
                for w in model.weights
            )
        else:
            memory = sum(
                np.prod(p.shape) * 4
                for p in model.get_params().values()
                if isinstance(p, np.ndarray)
            )
        
        return {
            'total_memory_bytes': memory,
            'total_memory_mb': memory / (1024 * 1024),
            'memory_per_parameter': memory / self.calculate_model_complexity(model)['num_parameters']
        }
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        report = {
            'basic_metrics': self.calculate_metrics(),
            'residual_analysis': self.analyze_residuals(),
            'calibration': self.calculate_calibration()
        }
        
        return report
    
    def calculate_time_series_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    window_size: int = 10) -> Dict[str, float]:
        """Calculate time series specific metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            window_size: Size of rolling window
            
        Returns:
            Dictionary of time series metrics
        """
        # Calculate rolling metrics
        rolling_mse = pd.Series(y_true - y_pred).rolling(window=window_size).apply(
            lambda x: np.mean(x**2)
        ).values
        
        rolling_mae = pd.Series(np.abs(y_true - y_pred)).rolling(window=window_size).mean().values
        
        return {
            'rolling_mse': np.mean(rolling_mse),
            'rolling_mae': np.mean(rolling_mae),
            'trend_accuracy': self._calculate_trend_accuracy(y_true, y_pred)
        }
        
    def _calculate_trend_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate trend prediction accuracy.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Trend accuracy score
        """
        true_trends = np.sign(np.diff(y_true))
        pred_trends = np.sign(np.diff(y_pred))
        
        return np.mean(true_trends == pred_trends)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to history.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if step is None:
            step = len(self.metrics_history)
            
        self.metrics_history[step] = metrics
        
    def get_metrics_history(self) -> Dict[int, Dict[str, float]]:
        """Get history of logged metrics.
        
        Returns:
            Dictionary of metrics history
        """
        return self.metrics_history
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from history.
        
        Returns:
            Dictionary of best metrics
        """
        if not self.metrics_history:
            return {}
            
        best_metrics = {}
        for metric in self.metrics_history[0].keys():
            if metric in ['mse', 'rmse', 'mae', 'max_error']:
                best_metrics[metric] = min(
                    metrics[metric] for metrics in self.metrics_history.values()
                )
            else:
                best_metrics[metric] = max(
                    metrics[metric] for metrics in self.metrics_history.values()
                )
                
        return best_metrics 