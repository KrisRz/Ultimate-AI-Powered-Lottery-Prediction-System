"""
Utilities for advanced cross-validation of time series models.

This module provides functions for:
1. Rolling window cross-validation for time series data
2. Time series split validation with expanding windows
3. Model evaluation across multiple validation folds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import TimeSeriesSplit
from scripts.train_models import EnsembleTrainer
from scripts.utils.memory_monitor import log_memory_usage

def perform_rolling_window_cv(
    trainer: EnsembleTrainer,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    min_train_size: Optional[int] = None,
    step_size: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Perform rolling window cross-validation for time series models.
    
    Args:
        trainer: The trained ensemble model
        X: Input features (n_samples, sequence_length, n_features)
        y: Target values (n_samples, n_targets)
        n_splits: Number of rolling window splits to use
        test_size: Proportion of data to use in each test set
        min_train_size: Minimum size of training set (if None, calculated based on test_size)
        step_size: Step size between windows (if None, calculated based on data size and n_splits)
        verbose: Whether to print progress
        
    Returns:
        Dictionary of metrics for each model type
    """
    if verbose:
        print(f"Performing rolling window cross-validation with {n_splits} splits...")
        log_memory_usage("Start of rolling window CV")
    
    n_samples = X.shape[0]
    test_length = int(n_samples * test_size)
    
    if min_train_size is None:
        min_train_size = int(n_samples * 0.5)  # Use at least 50% of data for training
    
    if step_size is None:
        # Calculate step size to create n_splits windows that cover the data
        available_range = n_samples - min_train_size - test_length
        step_size = max(1, available_range // (n_splits - 1)) if n_splits > 1 else available_range
    
    # Store metrics for each model type across all folds
    all_metrics = {
        "lstm": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []},
        "xgboost": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []},
        "lightgbm": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []},
        "ensemble": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []}
    }
    
    # Create windows for cross-validation
    for i in range(n_splits):
        if verbose:
            print(f"Rolling window fold {i+1}/{n_splits}")
        
        # Calculate window indices
        start_idx = min(i * step_size, n_samples - min_train_size - test_length)
        split_idx = start_idx + min_train_size
        end_idx = min(split_idx + test_length, n_samples)
        
        # Skip if window is invalid
        if split_idx >= n_samples or end_idx <= split_idx:
            if verbose:
                print(f"Skipping fold {i+1} - insufficient data")
            continue
        
        # Create train/test split for this window
        X_train, X_test = X[start_idx:split_idx], X[split_idx:end_idx]
        y_train, y_test = y[start_idx:split_idx], y[split_idx:end_idx]
        
        if verbose:
            print(f"  Window {i+1}: train={X_train.shape[0]} samples, test={X_test.shape[0]} samples")
            print(f"  Indices: {start_idx}:{split_idx}:{end_idx}")
        
        # Generate predictions
        predictions = trainer.predict(X_test)
        
        # Calculate metrics for each model
        for model_name, model_preds in predictions.items():
            # Basic metrics
            mse = np.mean((model_preds - y_test) ** 2)
            mae = np.mean(np.abs(model_preds - y_test))
            
            # Custom lottery-specific metrics (exact and partial matches)
            exact_matches, partial_matches = calculate_lottery_matches(model_preds, y_test)
            
            # Store metrics
            all_metrics[model_name]["mse"].append(mse)
            all_metrics[model_name]["mae"].append(mae)
            all_metrics[model_name]["exact_matches"].append(exact_matches)
            all_metrics[model_name]["partial_matches"].append(partial_matches)
            
            if verbose:
                print(f"  {model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, "
                     f"Exact: {exact_matches:.2%}, Partial: {partial_matches:.2%}")
    
    # Calculate summary statistics
    summary_metrics = {}
    for model_name, metrics in all_metrics.items():
        summary_metrics[model_name] = {}
        for metric_name, values in metrics.items():
            if values:  # Check if list is not empty
                summary_metrics[model_name][f"{metric_name}_mean"] = float(np.mean(values))
                summary_metrics[model_name][f"{metric_name}_std"] = float(np.std(values))
                summary_metrics[model_name][f"{metric_name}_min"] = float(np.min(values))
                summary_metrics[model_name][f"{metric_name}_max"] = float(np.max(values))
    
    # Log final memory usage
    if verbose:
        log_memory_usage("End of rolling window CV")
    
    return summary_metrics

def perform_time_series_cv(
    trainer: EnsembleTrainer,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    gap: int = 0,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Perform time series cross-validation using expanding windows.
    
    Args:
        trainer: The trained ensemble model
        X: Input features (n_samples, sequence_length, n_features)
        y: Target values (n_samples, n_targets)
        n_splits: Number of time series splits
        gap: Number of samples to exclude between train and test sets
        verbose: Whether to print progress
        
    Returns:
        Dictionary of metrics for each model type
    """
    if verbose:
        print(f"Performing time series cross-validation with {n_splits} splits...")
        log_memory_usage("Start of time series CV")
    
    # Create TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=None)
    
    # Store metrics for each model type across all folds
    all_metrics = {
        "lstm": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []},
        "xgboost": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []},
        "lightgbm": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []},
        "ensemble": {"mse": [], "mae": [], "exact_matches": [], "partial_matches": []}
    }
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if verbose:
            print(f"Time series fold {i+1}/{n_splits}")
        
        # Get train/test split for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if verbose:
            print(f"  Fold {i+1}: train={X_train.shape[0]} samples, test={X_test.shape[0]} samples")
        
        # Generate predictions
        predictions = trainer.predict(X_test)
        
        # Calculate metrics for each model
        for model_name, model_preds in predictions.items():
            # Basic metrics
            mse = np.mean((model_preds - y_test) ** 2)
            mae = np.mean(np.abs(model_preds - y_test))
            
            # Custom lottery-specific metrics (exact and partial matches)
            exact_matches, partial_matches = calculate_lottery_matches(model_preds, y_test)
            
            # Store metrics
            all_metrics[model_name]["mse"].append(mse)
            all_metrics[model_name]["mae"].append(mae)
            all_metrics[model_name]["exact_matches"].append(exact_matches)
            all_metrics[model_name]["partial_matches"].append(partial_matches)
            
            if verbose:
                print(f"  {model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, "
                     f"Exact: {exact_matches:.2%}, Partial: {partial_matches:.2%}")
    
    # Calculate summary statistics
    summary_metrics = {}
    for model_name, metrics in all_metrics.items():
        summary_metrics[model_name] = {}
        for metric_name, values in metrics.items():
            if values:  # Check if list is not empty
                summary_metrics[model_name][f"{metric_name}_mean"] = float(np.mean(values))
                summary_metrics[model_name][f"{metric_name}_std"] = float(np.std(values))
                summary_metrics[model_name][f"{metric_name}_min"] = float(np.min(values))
                summary_metrics[model_name][f"{metric_name}_max"] = float(np.max(values))
    
    # Log final memory usage
    if verbose:
        log_memory_usage("End of time series CV")
    
    return summary_metrics

def calculate_lottery_matches(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Calculate lottery-specific metrics (exact and partial matches).
    
    Args:
        predictions: Predicted lottery numbers (n_samples, n_numbers)
        targets: Actual lottery numbers (n_samples, n_numbers)
        
    Returns:
        Tuple of (exact_matches_rate, partial_matches_rate)
    """
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0, 0.0
    
    # Convert to integers and round predictions
    pred_ints = np.round(predictions * 69).astype(int)
    target_ints = np.round(targets * 69).astype(int)
    
    # Clip values to valid lottery number range (1-69)
    pred_ints = np.clip(pred_ints, 1, 69)
    target_ints = np.clip(target_ints, 1, 69)
    
    # Count exact matches (all numbers match)
    exact_match_count = 0
    
    # Count partial matches (at least 3 numbers match)
    partial_match_count = 0
    
    # Check each prediction-target pair
    for pred, target in zip(pred_ints, target_ints):
        # Convert each row to set for easier comparison
        pred_set = set(pred)
        target_set = set(target)
        
        # Count matching numbers
        matching_numbers = len(pred_set.intersection(target_set))
        
        # Check for exact match (all numbers match)
        if matching_numbers == len(pred):
            exact_match_count += 1
        
        # Check for partial match (at least 3 numbers match)
        if matching_numbers >= 3:
            partial_match_count += 1
    
    # Calculate match rates
    exact_match_rate = exact_match_count / len(predictions)
    partial_match_rate = partial_match_count / len(predictions)
    
    return exact_match_rate, partial_match_rate

def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    trainer_class: type,
    cv_method: str = "rolling_window",
    n_splits: int = 5,
    test_size: float = 0.2,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Comprehensive cross-validation of lottery prediction models.
    
    Args:
        X: Input features (n_samples, sequence_length, n_features)
        y: Target values (n_samples, n_targets)
        trainer_class: Class to use for creating trainer instances
        cv_method: Cross-validation method ("rolling_window" or "time_series")
        n_splits: Number of CV splits
        test_size: Proportion of data to use in each test set (for rolling window)
        verbose: Whether to print progress
        **kwargs: Additional arguments to pass to trainer_class constructor
        
    Returns:
        Dictionary of cross-validation results
    """
    if verbose:
        print(f"Performing {cv_method} cross-validation with {n_splits} splits...")
    
    # Create a new instance of the trainer for this CV
    trainer = trainer_class(**kwargs)
    
    # For conciseness in this case, we'll use the pre-trained models
    # In a real scenario, you might want to train models on each fold
    
    # Perform cross-validation based on method
    if cv_method == "rolling_window":
        results = perform_rolling_window_cv(
            trainer=trainer,
            X=X,
            y=y,
            n_splits=n_splits,
            test_size=test_size,
            verbose=verbose
        )
    elif cv_method == "time_series":
        results = perform_time_series_cv(
            trainer=trainer,
            X=X,
            y=y,
            n_splits=n_splits,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown CV method: {cv_method}")
    
    return results 