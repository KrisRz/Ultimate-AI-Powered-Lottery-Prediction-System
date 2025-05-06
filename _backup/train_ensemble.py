"""
Ensemble model trainer for lottery prediction.

This file contains the training logic for lottery prediction ensembles:
1. The EnsembleTrainer class which manages training of all component models
2. Utilities for monitoring, interpreting and evaluating the ensemble
3. Methods for saving/loading trained ensembles
4. Visualization and metrics reporting

This is the TRAINING LOGIC file, whereas models/ensemble.py 
contains the CORE MODEL DEFINITION.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import sys
from pathlib import Path
import time
import traceback
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import gc

from models.lstm_model import LSTMModel
from models.utils.model_utils import (
    setup_gpu_memory_growth,
    set_random_seed,
    create_sequences,
    normalize_data,
    denormalize_data,
    create_callbacks,
    validate_predictions,
    ensure_valid_prediction,
    validate_prediction_format,
    monitor_gpu
)
from models.utils.model_monitoring import ModelMonitor
from models.utils.model_interpretation import ModelInterpreter
from models.utils.model_metrics import ModelMetrics
from models.utils.model_serialization import ModelSerializer
from models.utils.model_versioning import ModelVersioner
from models.utils.model_deployment import ModelDeployer
from models.utils.model_validation import ModelValidator
from models.utils.visualization import ModelVisualizer
from models.utils.optimization import ModelOptimizer

from scripts.train_models import train_ensemble, predict_next_numbers
from scripts.performance_tracking import get_model_weights, calculate_metrics

logger = logging.getLogger(__name__)

class EnsembleTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.n_splits = config.get('n_splits', 5)
        
        # Initialize utilities
        self.monitor = None  # Will be initialized after model is created
        self.interpreter = ModelInterpreter()
        self.metrics = None  # Will be initialized during training
        self.serializer = None  # Will be initialized after model is created
        self.versioner = None  # Will be initialized after model is created
        self.deployer = None  # Will be initialized after model is created
        self.validator = ModelValidator()
        self.visualizer = None  # Will be initialized after model is created
        self.optimizer = ModelOptimizer()
        
        # Initialize base models
        self._init_models()
        
        # Initialize monitor, serializer, versioner, deployer, and visualizer after models are created
        self.monitor = ModelMonitor(
            model=self.models.get('lstm'),  # Use LSTM as the main model for monitoring
            model_name='lstm',
            monitoring_dir='models/monitoring'
        )
        
        self.serializer = ModelSerializer(
            model=self.models.get('lstm'),  # Use LSTM as the main model for serialization
            model_name='lstm'
        )
        
        self.versioner = ModelVersioner(
            model=self.models.get('lstm'),  # Use LSTM as the main model for versioning
            model_name='lstm',
            versioning_dir='models/versions'
        )
        
        self.deployer = ModelDeployer(
            model=self.models.get('lstm'),  # Use LSTM as the main model for deployment
            model_name='lstm',
            deployment_dir='models/deployment'
        )
        
        self.visualizer = ModelVisualizer(
            model=self.models.get('lstm')  # Use LSTM as the main model for visualization
        )
        
    def _init_models(self):
        """Initialize all base models."""
        try:
            # LSTM model
            self.models['lstm'] = LSTMModel(self.config.get('lstm_config', {}))
            
            # XGBoost model
            self.models['xgboost'] = [
                xgb.XGBRegressor(
                    **self.config.get('xgboost', {
                        'max_depth': 6,
                        'learning_rate': 0.01,
                        'n_estimators': 1000,
                        'objective': 'reg:squarederror'
                    })
                ) for _ in range(6)  # One model per lottery number
            ]
            
            # LightGBM model
            self.models['lightgbm'] = [
                lgb.LGBMRegressor(
                    **self.config.get('lightgbm', {
                        'num_leaves': 31,
                        'learning_rate': 0.01,
                        'n_estimators': 1000
                    })
                ) for _ in range(6)  # One model per lottery number
            ]
            
            # Meta-model (Random Forest)
            self.meta_model = RandomForestRegressor(
                **self.config.get('meta', {
                    'n_estimators': 100,
                    'max_depth': 10
                })
            )
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
            
    @monitor_gpu
    def train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train the ensemble model with monitoring and interpretation."""
        try:
            # Monitor feature distribution before training
            self.monitor.monitor_feature_distribution(X)
            
            # Train ensemble
            self.ensemble = train_ensemble(
                X,
                timesteps=self.config.get('timesteps', 10),
                mc_splits=self.config.get('mc_splits', 5),
                rolling_splits=self.config.get('rolling_splits', 5),
                save_path=self.config.get('save_path', 'models/saved_models')
            )
            
            # Make predictions and initialize metrics
            predictions = predict_next_numbers(self.ensemble, X[-1:])
            self.metrics = ModelMetrics(y[-1:], predictions, task='regression')
            
            # Monitor performance
            self.monitor.monitor_performance(self.ensemble, X, y)
            
            # Calculate metrics
            metrics = self.metrics.calculate_metrics()
            logger.info(f"Training metrics: {metrics}")
            
            # Interpret model behavior
            self.interpreter.interpret_model_behavior(self.ensemble, X, y)
            
            # Optimize model
            self.optimizer.optimize_model(self.ensemble, X, y)
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions with visualization and monitoring."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        try:
            # Generate predictions
            predictions = predict_next_numbers(self.ensemble, X)
            
            # Monitor prediction distribution
            self.monitor.monitor_prediction_distribution(predictions)
            
            # Visualize predictions
            self.visualizer.plot_predictions(predictions, X)
            
            # Validate predictions
            self.validator.validate_predictions(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
            
    def save(self, save_dir: Path):
        """Save the ensemble model with versioning and deployment."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Version the model
            version = self.versioner.create_version(self.ensemble, save_dir)
            
            # Serialize model
            self.serializer.save(self.ensemble, save_dir / 'ensemble.joblib')
            
            # Deploy model
            self.deployer.deploy_model(self.ensemble, version, save_dir)
            
            # Save metadata
            metadata = {
                'config': self.config,
                'is_trained': self.is_trained,
                'n_splits': self.n_splits,
                'version': version
            }
            with open(save_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load(self, load_dir: Path):
        """Load the ensemble model with validation."""
        try:
            load_dir = Path(load_dir)
            
            # Load metadata
            with open(load_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                
            # Load model
            self.ensemble = self.serializer.load(load_dir / 'ensemble.joblib')
            
            # Validate loaded model
            self.validator.validate_model(self.ensemble)
            
            self.config = metadata['config']
            self.is_trained = metadata['is_trained']
            self.n_splits = metadata['n_splits']
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def get_model_insights(self) -> Dict[str, Any]:
        """Get comprehensive model insights."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        try:
            insights = {
                'performance': self.monitor.generate_monitoring_report(),
                'interpretation': self.interpreter.interpret_model_behavior(self.ensemble),
                'metrics': self.metrics.generate_metrics_report(),
                'validation': self.validator.get_validation_report()
            }
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model insights: {str(e)}")
            raise 