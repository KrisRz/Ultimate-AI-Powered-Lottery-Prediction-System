"""Model monitoring utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from .model_metrics import ModelMetrics
from .model_utils import set_random_seed
import time
import psutil
import GPUtil
from functools import wraps

logger = logging.getLogger(__name__)

def monitor_gpu(func):
    """Decorator to monitor GPU usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                logger.info(f"GPU Memory before {func.__name__}: {gpus[0].memoryUsed}MB")
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                logger.info(f"GPU Memory after {func.__name__}: {gpus[0].memoryUsed}MB")
                logger.info(f"Execution time: {end_time - start_time:.2f}s")
                return result
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error monitoring GPU: {e}")
            return func(*args, **kwargs)
    return wrapper

class ModelMonitor:
    """Monitor model performance and data drift over time."""
    
    def __init__(self, model: Any, model_name: str, monitoring_dir: str):
        """Initialize model monitor."""
        self.model = model
        self.model_name = model_name
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize monitoring history
        self.history_file = self.monitoring_dir / "monitoring_history.json"
        self.history = self._load_history()
        
        # Set random seed for reproducibility
        set_random_seed(42)
        
        self.metrics_history = {}
        self.resource_history = {}
    
    def _load_history(self) -> Dict[str, Any]:
        """Load monitoring history from disk."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {
            'performance': [],
            'feature_distribution': [],
            'prediction_distribution': [],
            'drift_detection': []
        }
    
    def _save_history(self) -> None:
        """Save monitoring history to disk."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def monitor_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Monitor model performance."""
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = ModelMetrics(y, y_pred)
        performance = metrics.calculate_metrics()
        
        # Record performance
        timestamp = datetime.now().isoformat()
        self.history['performance'].append({
            'timestamp': timestamp,
            'metrics': performance
        })
        self._save_history()
        
        return performance
    
    def monitor_feature_distribution(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Monitor feature distribution drift."""
        # Handle 3D data (samples, timesteps, features)
        if len(X.shape) == 3:
            # Reshape to 2D by combining samples and timesteps
            X = X.reshape(-1, X.shape[-1])
        
        # Handle 2D data (samples, features)
        if len(X.shape) == 2:
            # Calculate feature statistics for each feature
            feature_stats = {}
            for i in range(X.shape[1]):
                feature_name = feature_names[i] if feature_names else f'feature_{i}'
                feature_stats[feature_name] = {
                    'mean': np.mean(X[:, i]),
                    'std': np.std(X[:, i]),
                    'min': np.min(X[:, i]),
                    'max': np.max(X[:, i]),
                    'skewness': pd.Series(X[:, i].flatten()).skew(),
                    'kurtosis': pd.Series(X[:, i].flatten()).kurtosis()
                }
        else:
            raise ValueError(f"Expected 2D or 3D data, got shape {X.shape}")
        
        # Record feature distribution
        timestamp = datetime.now().isoformat()
        self.history['feature_distribution'].append({
            'timestamp': timestamp,
            'statistics': feature_stats
        })
        self._save_history()
        
        return feature_stats
    
    def monitor_prediction_distribution(self, X: np.ndarray) -> Dict[str, Any]:
        """Monitor prediction distribution drift."""
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate prediction statistics
        prediction_stats = {
            'mean': np.mean(y_pred),
            'std': np.std(y_pred),
            'min': np.min(y_pred),
            'max': np.max(y_pred),
            'skewness': pd.Series(y_pred).skew(),
            'kurtosis': pd.Series(y_pred).kurtosis()
        }
        
        # Record prediction distribution
        timestamp = datetime.now().isoformat()
        self.history['prediction_distribution'].append({
            'timestamp': timestamp,
            'statistics': prediction_stats
        })
        self._save_history()
        
        return prediction_stats
    
    def detect_drift(self, window_size: int = 10) -> Dict[str, Any]:
        """Detect drift in model performance and data distribution."""
        if len(self.history['performance']) < window_size:
            return {
                'performance_drift': None,
                'feature_drift': None,
                'prediction_drift': None
            }
        
        # Get recent history
        recent_performance = self.history['performance'][-window_size:]
        recent_features = self.history['feature_distribution'][-window_size:]
        recent_predictions = self.history['prediction_distribution'][-window_size:]
        
        # Detect performance drift
        performance_drift = self._detect_performance_drift(recent_performance)
        
        # Detect feature drift
        feature_drift = self._detect_feature_drift(recent_features)
        
        # Detect prediction drift
        prediction_drift = self._detect_prediction_drift(recent_predictions)
        
        # Record drift detection
        timestamp = datetime.now().isoformat()
        self.history['drift_detection'].append({
            'timestamp': timestamp,
            'performance_drift': performance_drift,
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift
        })
        self._save_history()
        
        return {
            'performance_drift': performance_drift,
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift
        }
    
    def _detect_performance_drift(self, recent_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect performance drift."""
        # Extract metrics
        metrics = {}
        for entry in recent_performance:
            for metric, value in entry['metrics'].items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(value)
        
        # Calculate drift
        drift = {}
        for metric, values in metrics.items():
            # Calculate mean and std of first half
            n = len(values) // 2
            mean1 = np.mean(values[:n])
            std1 = np.std(values[:n])
            
            # Calculate mean and std of second half
            mean2 = np.mean(values[n:])
            std2 = np.std(values[n:])
            
            # Calculate drift score
            drift_score = abs(mean2 - mean1) / (std1 + 1e-10)
            
            drift[metric] = {
                'drift_score': drift_score,
                'mean_change': mean2 - mean1,
                'std_change': std2 - std1
            }
        
        return drift
    
    def _detect_feature_drift(self, recent_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect feature distribution drift."""
        # Extract feature statistics
        feature_stats = {}
        for entry in recent_features:
            for feature, stats in entry['statistics'].items():
                if feature not in feature_stats:
                    feature_stats[feature] = {k: [] for k in stats.keys()}
                for stat, value in stats.items():
                    feature_stats[feature][stat].append(value)
        
        # Calculate drift
        drift = {}
        for feature, stats in feature_stats.items():
            feature_drift = {}
            for stat, values in stats.items():
                # Calculate mean and std of first half
                n = len(values) // 2
                mean1 = np.mean(values[:n])
                std1 = np.std(values[:n])
                
                # Calculate mean and std of second half
                mean2 = np.mean(values[n:])
                std2 = np.std(values[n:])
                
                # Calculate drift score
                drift_score = abs(mean2 - mean1) / (std1 + 1e-10)
                
                feature_drift[stat] = {
                    'drift_score': drift_score,
                    'mean_change': mean2 - mean1,
                    'std_change': std2 - std1
                }
            
            drift[feature] = feature_drift
        
        return drift
    
    def _detect_prediction_drift(self, recent_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect prediction distribution drift."""
        # Extract prediction statistics
        pred_stats = {k: [] for k in recent_predictions[0]['statistics'].keys()}
        for entry in recent_predictions:
            for stat, value in entry['statistics'].items():
                pred_stats[stat].append(value)
        
        # Calculate drift
        drift = {}
        for stat, values in pred_stats.items():
            # Calculate mean and std of first half
            n = len(values) // 2
            mean1 = np.mean(values[:n])
            std1 = np.std(values[:n])
            
            # Calculate mean and std of second half
            mean2 = np.mean(values[n:])
            std2 = np.std(values[n:])
            
            # Calculate drift score
            drift_score = abs(mean2 - mean1) / (std1 + 1e-10)
            
            drift[stat] = {
                'drift_score': drift_score,
                'mean_change': mean2 - mean1,
                'std_change': std2 - std1
            }
        
        return drift
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        return {
            'performance_history': self.history['performance'],
            'feature_distribution_history': self.history['feature_distribution'],
            'prediction_distribution_history': self.history['prediction_distribution'],
            'drift_detection_history': self.history['drift_detection']
        }
    
    def save_monitoring_report(self, path: str) -> str:
        """Save monitoring report to disk."""
        report = self.generate_monitoring_report()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return str(path)
    
    def plot_monitoring_history(self) -> None:
        """Plot monitoring history."""
        # Plot performance history
        if self.history['performance']:
            self._plot_performance_history()
        
        # Plot feature distribution history
        if self.history['feature_distribution']:
            self._plot_feature_distribution_history()
        
        # Plot prediction distribution history
        if self.history['prediction_distribution']:
            self._plot_prediction_distribution_history()
        
        # Plot drift detection history
        if self.history['drift_detection']:
            self._plot_drift_detection_history()
    
    def _plot_performance_history(self) -> None:
        """Plot performance history."""
        timestamps = [entry['timestamp'] for entry in self.history['performance']]
        metrics = list(self.history['performance'][0]['metrics'].keys())
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = [entry['metrics'][metric] for entry in self.history['performance']]
            plt.plot(timestamps, values, label=metric)
        
        plt.title('Model Performance History')
        plt.xlabel('Timestamp')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _plot_feature_distribution_history(self) -> None:
        """Plot feature distribution history."""
        timestamps = [entry['timestamp'] for entry in self.history['feature_distribution']]
        features = list(self.history['feature_distribution'][0]['statistics'].keys())
        stats = list(self.history['feature_distribution'][0]['statistics'][features[0]].keys())
        
        for stat in stats:
            plt.figure(figsize=(12, 6))
            for feature in features:
                values = [
                    entry['statistics'][feature][stat]
                    for entry in self.history['feature_distribution']
                ]
                plt.plot(timestamps, values, label=feature)
            
            plt.title(f'Feature {stat.capitalize()} History')
            plt.xlabel('Timestamp')
            plt.ylabel(stat.capitalize())
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def _plot_prediction_distribution_history(self) -> None:
        """Plot prediction distribution history."""
        timestamps = [entry['timestamp'] for entry in self.history['prediction_distribution']]
        stats = list(self.history['prediction_distribution'][0]['statistics'].keys())
        
        plt.figure(figsize=(12, 6))
        for stat in stats:
            values = [
                entry['statistics'][stat]
                for entry in self.history['prediction_distribution']
            ]
            plt.plot(timestamps, values, label=stat)
        
        plt.title('Prediction Distribution History')
        plt.xlabel('Timestamp')
        plt.ylabel('Statistic Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _plot_drift_detection_history(self) -> None:
        """Plot drift detection history."""
        timestamps = [entry['timestamp'] for entry in self.history['drift_detection']]
        
        # Plot performance drift
        if self.history['drift_detection'][0]['performance_drift']:
            plt.figure(figsize=(12, 6))
            for metric, drift in self.history['drift_detection'][0]['performance_drift'].items():
                values = [
                    entry['performance_drift'][metric]['drift_score']
                    for entry in self.history['drift_detection']
                ]
                plt.plot(timestamps, values, label=metric)
            
            plt.title('Performance Drift History')
            plt.xlabel('Timestamp')
            plt.ylabel('Drift Score')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Plot feature drift
        if self.history['drift_detection'][0]['feature_drift']:
            features = list(self.history['drift_detection'][0]['feature_drift'].keys())
            stats = list(self.history['drift_detection'][0]['feature_drift'][features[0]].keys())
            
            for stat in stats:
                plt.figure(figsize=(12, 6))
                for feature in features:
                    values = [
                        entry['feature_drift'][feature][stat]['drift_score']
                        for entry in self.history['drift_detection']
                    ]
                    plt.plot(timestamps, values, label=feature)
                
                plt.title(f'Feature {stat.capitalize()} Drift History')
                plt.xlabel('Timestamp')
                plt.ylabel('Drift Score')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
        
        # Plot prediction drift
        if self.history['drift_detection'][0]['prediction_drift']:
            plt.figure(figsize=(12, 6))
            for stat, drift in self.history['drift_detection'][0]['prediction_drift'].items():
                values = [
                    entry['prediction_drift'][stat]['drift_score']
                    for entry in self.history['drift_detection']
                ]
                plt.plot(timestamps, values, label=stat)
            
            plt.title('Prediction Drift History')
            plt.xlabel('Timestamp')
            plt.ylabel('Drift Score')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def alert_on_drift(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Generate alerts for significant drift."""
        alerts = []
        
        # Check performance drift
        if self.history['drift_detection']:
            latest_drift = self.history['drift_detection'][-1]
            
            # Check performance drift
            if latest_drift['performance_drift']:
                for metric, drift in latest_drift['performance_drift'].items():
                    if drift['drift_score'] > threshold:
                        alerts.append({
                            'type': 'performance',
                            'metric': metric,
                            'drift_score': drift['drift_score'],
                            'timestamp': latest_drift['timestamp']
                        })
            
            # Check feature drift
            if latest_drift['feature_drift']:
                for feature, stats in latest_drift['feature_drift'].items():
                    for stat, drift in stats.items():
                        if drift['drift_score'] > threshold:
                            alerts.append({
                                'type': 'feature',
                                'feature': feature,
                                'statistic': stat,
                                'drift_score': drift['drift_score'],
                                'timestamp': latest_drift['timestamp']
                            })
            
            # Check prediction drift
            if latest_drift['prediction_drift']:
                for stat, drift in latest_drift['prediction_drift'].items():
                    if drift['drift_score'] > threshold:
                        alerts.append({
                            'type': 'prediction',
                            'statistic': stat,
                            'drift_score': drift['drift_score'],
                            'timestamp': latest_drift['timestamp']
                        })
        
        return alerts
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log model metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if step is None:
            step = len(self.metrics_history)
            
        self.metrics_history[step] = metrics
        
    def log_resources(self) -> Dict[str, float]:
        """Log system resources.
        
        Returns:
            Dictionary of resource metrics
        """
        resources = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory': 0
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                resources['gpu_memory'] = gpus[0].memoryUsed
        except:
            pass
            
        self.resource_history[time.time()] = resources
        return resources
        
    def get_metrics_history(self) -> Dict[int, Dict[str, float]]:
        """Get history of logged metrics.
        
        Returns:
            Dictionary of metrics history
        """
        return self.metrics_history
        
    def get_resource_history(self) -> Dict[float, Dict[str, float]]:
        """Get history of resource usage.
        
        Returns:
            Dictionary of resource history
        """
        return self.resource_history
        
    def plot_metrics(self) -> None:
        """Plot metrics history."""
        if not self.metrics_history:
            return
            
        metrics = list(self.metrics_history[0].keys())
        steps = list(self.metrics_history.keys())
        
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            values = [self.metrics_history[step][metric] for step in steps]
            plt.plot(steps, values, label=metric)
            
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Model Metrics History')
        plt.legend()
        plt.show()
        
    def plot_resources(self) -> None:
        """Plot resource usage history."""
        if not self.resource_history:
            return
            
        times = list(self.resource_history.keys())
        resources = list(self.resource_history[times[0]].keys())
        
        plt.figure(figsize=(10, 6))
        for resource in resources:
            values = [self.resource_history[time][resource] for time in times]
            plt.plot(times, values, label=resource)
            
        plt.xlabel('Time')
        plt.ylabel('Usage')
        plt.title('Resource Usage History')
        plt.legend()
        plt.show() 