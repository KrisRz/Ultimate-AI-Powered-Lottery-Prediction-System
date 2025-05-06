"""Model interpretation utilities."""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .model_utils import set_random_seed

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """Class for interpreting model predictions."""
    
    def __init__(self):
        """Initialize ModelInterpreter."""
        self.explainer = None
        self.feature_names = None
        
    def explain_prediction(self, model: Any, X: np.ndarray, feature_names: List[str], 
                          instance_idx: int = 0) -> Dict[str, float]:
        """Explain a single prediction using LIME.
        
        Args:
            model: Trained model
            X: Input features
            feature_names: List of feature names
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.explainer is None or self.feature_names != feature_names:
            self.feature_names = feature_names
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode='regression'
            )
            
        exp = self.explainer.explain_instance(
            X[instance_idx],
            model.predict,
            num_features=len(feature_names)
        )
        
        return dict(exp.as_list())
        
    def get_global_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                            feature_names: List[str]) -> Dict[str, float]:
        """Get global feature importance using permutation importance.
        
        Args:
            model: Trained model
            X: Input features
            y: Target values
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42
        )
        
        return dict(zip(feature_names, result.importances_mean))
        
    def get_shap_values(self, model: Any, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Get SHAP values for model predictions.
        
        Args:
            model: Trained model
            X: Input features
            feature_names: List of feature names
            
        Returns:
            Array of SHAP values
        """
        if isinstance(model, tf.keras.Model):
            explainer = shap.DeepExplainer(model, X)
        else:
            explainer = shap.TreeExplainer(model)
            
        shap_values = explainer.shap_values(X)
        return np.array(shap_values)
        
    def plot_shap_summary(self, shap_values: np.ndarray, X: np.ndarray,
                         feature_names: List[str]) -> None:
        """Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X: Input features
            feature_names: List of feature names
        """
        shap.summary_plot(shap_values, X, feature_names=feature_names)
        
    def plot_shap_dependence(self, shap_values: np.ndarray, X: np.ndarray,
                           feature_names: List[str], feature_idx: int) -> None:
        """Plot SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values
            X: Input features
            feature_names: List of feature names
            feature_idx: Index of feature to plot
        """
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_names
        )
    
    def interpret_feature_importance(self, method: str = 'permutation') -> Dict[str, float]:
        """Interpret feature importance."""
        if method == 'permutation':
            return self.get_global_importance(self.model, self.X, self.y, self.feature_names)
        elif method == 'shap':
            return self.get_shap_values(self.model, self.X, self.feature_names)
        elif method == 'tree':
            return self._interpret_tree_importance()
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _interpret_tree_importance(self) -> Dict[str, float]:
        """Interpret feature importance for tree-based models."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        else:
            raise ValueError("Model does not support tree-based feature importance")
    
    def interpret_shap_values(self) -> Dict[str, np.ndarray]:
        """Calculate SHAP values for model predictions."""
        return {
            'shap_values': self.get_shap_values(self.model, self.X, self.feature_names),
            'expected_value': self.get_shap_values(self.model, self.X, self.feature_names).mean()
        }
    
    def interpret_lime_explanations(self, instance_idx: int) -> Dict[str, Any]:
        """Generate LIME explanations for a specific instance."""
        return self.explain_prediction(self.model, self.X, self.feature_names, instance_idx)
    
    def interpret_partial_dependence(self, feature_idx: int) -> Dict[str, np.ndarray]:
        """Calculate partial dependence for a specific feature."""
        pd_values = partial_dependence(
            self.model,
            self.X,
            features=[feature_idx],
            percentiles=(0, 1)
        )
        
        return {
            'feature_values': pd_values['values'][0],
            'partial_dependence': pd_values['average'][0]
        }
    
    def interpret_decision_boundary(self, feature1_idx: int, feature2_idx: int) -> Dict[str, np.ndarray]:
        """Interpret decision boundary for two features."""
        # Create grid
        x_min, x_max = self.X[:, feature1_idx].min(), self.X[:, feature1_idx].max()
        y_min, y_max = self.X[:, feature2_idx].min(), self.X[:, feature2_idx].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Create test points
        test_points = np.zeros((xx.ravel().shape[0], self.X.shape[1]))
        test_points[:, feature1_idx] = xx.ravel()
        test_points[:, feature2_idx] = yy.ravel()
        
        # Get predictions
        Z = self.model.predict(test_points)
        Z = Z.reshape(xx.shape)
        
        return {
            'xx': xx,
            'yy': yy,
            'Z': Z
        }
    
    def interpret_model_behavior(self) -> Dict[str, Any]:
        """Provide comprehensive interpretation of model behavior."""
        return {
            'feature_importance': self.interpret_feature_importance(),
            'shap_values': self.interpret_shap_values(),
            'partial_dependence': {
                feature: self.interpret_partial_dependence(i)
                for i, feature in enumerate(self.feature_names)
            }
        }
    
    def interpret_prediction_uncertainty(self) -> Dict[str, np.ndarray]:
        """Interpret prediction uncertainty."""
        if isinstance(self.model, tf.keras.Model):
            # Monte Carlo Dropout
            predictions = []
            for _ in range(100):
                predictions.append(self.model.predict(self.X))
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
                X_resampled, y_resampled = resample(self.X, self.y)
                self.model.fit(X_resampled, y_resampled)
                predictions.append(self.model.predict(self.X))
            predictions = np.array(predictions)
            
            return {
                'mean': np.mean(predictions, axis=0),
                'std': np.std(predictions, axis=0),
                'confidence_interval': np.percentile(predictions, [2.5, 97.5], axis=0)
            }
    
    def plot_feature_importance(self, method: str = 'permutation') -> None:
        """Plot feature importance."""
        importance = self.interpret_feature_importance(method)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=list(importance.values()),
            y=list(importance.keys())
        )
        plt.title(f'Feature Importance ({method})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self) -> None:
        """Plot SHAP summary plot."""
        shap_values = self.interpret_shap_values()
        
        self.plot_shap_summary(shap_values['shap_values'], self.X, self.feature_names)
    
    def plot_partial_dependence(self, feature_idx: int) -> None:
        """Plot partial dependence for a specific feature."""
        pd_values = self.interpret_partial_dependence(feature_idx)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            pd_values['feature_values'],
            pd_values['partial_dependence']
        )
        plt.title(f'Partial Dependence Plot: {self.feature_names[feature_idx]}')
        plt.xlabel(self.feature_names[feature_idx])
        plt.ylabel('Partial Dependence')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, feature1_idx: int, feature2_idx: int) -> None:
        """Plot decision boundary for two features."""
        boundary = self.interpret_decision_boundary(feature1_idx, feature2_idx)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(
            boundary['xx'],
            boundary['yy'],
            boundary['Z'],
            alpha=0.8
        )
        plt.scatter(
            self.X[:, feature1_idx],
            self.X[:, feature2_idx],
            c=self.y,
            edgecolors='k'
        )
        plt.title(
            f'Decision Boundary: {self.feature_names[feature1_idx]} vs '
            f'{self.feature_names[feature2_idx]}'
        )
        plt.xlabel(self.feature_names[feature1_idx])
        plt.ylabel(self.feature_names[feature2_idx])
        plt.colorbar()
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_uncertainty(self) -> None:
        """Plot prediction uncertainty."""
        uncertainty = self.interpret_prediction_uncertainty()
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            range(len(uncertainty['mean'])),
            uncertainty['mean'],
            yerr=uncertainty['std'],
            fmt='o',
            capsize=5
        )
        plt.title('Prediction Uncertainty')
        plt.xlabel('Sample')
        plt.ylabel('Prediction')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_tree_structure(self, tree_idx: int = 0) -> None:
        """Plot structure of a decision tree."""
        if hasattr(self.model, 'estimators_'):
            # Random Forest
            tree = self.model.estimators_[tree_idx]
        elif hasattr(self.model, 'tree_'):
            # Single Decision Tree
            tree = self.model
        else:
            raise ValueError("Model does not support tree visualization")
        
        dot_data = export_graphviz(
            tree,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            special_characters=True
        )
        
        graph = graphviz.Source(dot_data)
        return graph
    
    def generate_interpretation_report(self) -> Dict[str, Any]:
        """Generate comprehensive interpretation report."""
        return {
            'feature_importance': self.interpret_feature_importance(),
            'shap_values': self.interpret_shap_values(),
            'partial_dependence': {
                feature: self.interpret_partial_dependence(i)
                for i, feature in enumerate(self.feature_names)
            },
            'prediction_uncertainty': self.interpret_prediction_uncertainty()
        } 