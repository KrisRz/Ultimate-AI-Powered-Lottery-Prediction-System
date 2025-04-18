"""
Functions for predicting lottery numbers using various models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.analyze_data import analyze_lottery_data, load_data
from scripts.train_models import create_sequences, MODELS_DIR, DATA_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models() -> Dict[str, Any]:
    """Load trained models."""
    try:
        model_file = MODELS_DIR / "trained_models.joblib"
        if not model_file.exists():
            raise FileNotFoundError("Models not found. Please train models first.")
        return joblib.load(model_file)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def predict_lstm(model: tf.keras.Model, scaler: Any, data: np.ndarray) -> float:
    """Make prediction using LSTM model."""
    try:
        X = create_sequences(data, 10)[0][-1:]
        pred = model.predict(X, verbose=0)[0][0]
        return scaler.inverse_transform([[pred]])[0][0]
    except Exception as e:
        logger.error(f"Error in LSTM prediction: {e}")
        raise

def predict_arima(model: Any, data: np.ndarray) -> float:
    """Make prediction using ARIMA model."""
    try:
        forecast = model.forecast(steps=1)
        return forecast[0]
    except Exception as e:
        logger.error(f"Error in ARIMA prediction: {e}")
        raise

def predict_holtwinters(model: Any, data: np.ndarray) -> float:
    """Make prediction using Holt-Winters model."""
    try:
        forecast = model.forecast(1)
        return forecast[0]
    except Exception as e:
        logger.error(f"Error in Holt-Winters prediction: {e}")
        raise

def predict_ml_model(model: Any, scaler: Any, data: np.ndarray) -> float:
    """Make prediction using ML model."""
    try:
        X = create_sequences(data, 10)[0][-1:].reshape(1, -1)
        pred = model.predict(X)[0]
        return scaler.inverse_transform([[pred]])[0][0]
    except Exception as e:
        logger.error(f"Error in ML model prediction: {e}")
        raise

def generate_combinations(predictions: Dict[str, List[float]], n_combinations: int = 10) -> List[List[int]]:
    """Generate number combinations based on predictions."""
    try:
        all_numbers = set(range(1, 50))  # Adjust range based on lottery rules
        combinations = []
        
        # Convert predictions to probabilities
        prob_dict = {}
        for col, preds in predictions.items():
            mean_pred = np.mean(preds)
            std_pred = np.std(preds)
            for num in all_numbers:
                prob = 1 / (1 + np.abs(num - mean_pred) / (std_pred + 1e-6))
                prob_dict[num] = prob_dict.get(num, 0) + prob
        
        # Sort numbers by probability
        sorted_numbers = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_numbers[:15]]  # Take top 15 numbers
        
        # Generate combinations
        for _ in range(n_combinations):
            combination = sorted(np.random.choice(top_numbers, size=6, replace=False))
            combinations.append(combination)
        
        return combinations
    except Exception as e:
        logger.error(f"Error generating combinations: {e}")
        raise

def score_combination(combination: List[int], analysis_results: Dict) -> float:
    """Score a combination based on historical patterns."""
    try:
        score = 0.0
        
        # Check for common pairs
        for i, j in itertools.combinations(combination, 2):
            pair = tuple(sorted([i, j]))
            if pair in analysis_results['common_pairs']:
                score += 0.1 * analysis_results['common_pairs'][pair]
        
        # Check for hot/cold numbers balance
        hot_count = len(set(combination) & set(analysis_results['hot_numbers']))
        cold_count = len(set(combination) & set(analysis_results['cold_numbers']))
        score += 0.2 * (min(hot_count, cold_count) / max(hot_count, cold_count, 1))
        
        # Check number frequency distribution
        freqs = [analysis_results['number_frequencies'].get(num, 0) for num in combination]
        score += 0.3 * (1 - np.std(freqs) / (np.mean(freqs) + 1e-6))
        
        # Check gaps
        gaps = [analysis_results['max_gaps'].get(num, 0) for num in combination]
        score += 0.2 * (1 - np.std(gaps) / (np.mean(gaps) + 1e-6))
        
        return score
    except Exception as e:
        logger.error(f"Error scoring combination: {e}")
        raise

def monte_carlo_simulation(combinations: List[List[int]], analysis_results: Dict, n_simulations: int = 1000) -> List[List[int]]:
    """Run Monte Carlo simulation to optimize combinations."""
    try:
        scores = []
        for combination in combinations:
            total_score = 0
            for _ in range(n_simulations):
                # Randomly modify combination
                modified = combination.copy()
                if np.random.random() < 0.3:  # 30% chance to modify
                    idx = np.random.randint(0, len(modified))
                    new_num = np.random.randint(1, 50)  # Adjust range based on lottery rules
                    while new_num in modified:
                        new_num = np.random.randint(1, 50)
                    modified[idx] = new_num
                    modified.sort()
                
                score = score_combination(modified, analysis_results)
                total_score += score
            
            scores.append(total_score / n_simulations)
        
        # Sort combinations by score
        sorted_combinations = [x for _, x in sorted(zip(scores, combinations), reverse=True)]
        return sorted_combinations
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {e}")
        raise

def predict_next_draw(models_dict: Dict[str, Any], df: pd.DataFrame) -> List[List[int]]:
    """Predict numbers for the next draw."""
    try:
        predictions = {}
        number_cols = [col for col in df.columns if col.startswith('Number')]
        
        for col in number_cols:
            data = df[col].values
            scaler = models_dict['scalers'][col]
            scaled_data = scaler.transform(data.reshape(-1, 1)).ravel()
            
            col_predictions = []
            
            # LSTM prediction
            lstm_pred = predict_lstm(models_dict['models'][f'{col}_lstm'], 
                                  scaler, scaled_data)
            col_predictions.append(lstm_pred)
            
            # ARIMA prediction
            arima_pred = predict_arima(models_dict['models'][f'{col}_arima'], 
                                    data)
            col_predictions.append(arima_pred)
            
            # Holt-Winters prediction
            hw_pred = predict_holtwinters(models_dict['models'][f'{col}_hw'], 
                                       data)
            col_predictions.append(hw_pred)
            
            # ML models predictions
            for model_type in ['xgb', 'lgb', 'cat', 'gb', 'knn']:
                ml_pred = predict_ml_model(models_dict['models'][f'{col}_{model_type}'],
                                        scaler, scaled_data)
                col_predictions.append(ml_pred)
            
            predictions[col] = col_predictions
        
        # Generate and optimize combinations
        analysis_results = analyze_lottery_data(df)
        combinations = generate_combinations(predictions)
        optimized_combinations = monte_carlo_simulation(combinations, analysis_results)
        
        return optimized_combinations
    except Exception as e:
        logger.error(f"Error predicting next draw: {e}")
        raise

def backtest(df: pd.DataFrame, window_size: int = 52) -> Dict[str, float]:
    """Perform backtesting of the prediction system."""
    try:
        results = {
            'accuracy': [],
            'partial_matches': [],
            'mae': [],
            'rmse': []
        }
        
        for i in tqdm(range(window_size, len(df)), desc="Backtesting"):
            train_df = df.iloc[:i]
            test_row = df.iloc[i]
            
            # Train models on window
            from scripts.train_models import train_all_models
            models = train_all_models(train_df)
            
            # Make predictions
            predictions = predict_next_draw(models, train_df)
            actual_numbers = sorted([test_row[col] for col in df.columns if col.startswith('Number')])
            
            # Calculate metrics
            best_match = 0
            for pred in predictions:
                matches = len(set(pred) & set(actual_numbers))
                best_match = max(best_match, matches)
            
            results['accuracy'].append(1 if best_match == 6 else 0)
            results['partial_matches'].append(best_match / 6)
            
            # Calculate regression metrics
            pred_array = np.array(predictions[0])  # Use first prediction
            actual_array = np.array(actual_numbers)
            results['mae'].append(mean_absolute_error(actual_array, pred_array))
            results['rmse'].append(np.sqrt(mean_squared_error(actual_array, pred_array)))
        
        # Aggregate results
        return {
            'accuracy': np.mean(results['accuracy']),
            'partial_matches': np.mean(results['partial_matches']),
            'mae': np.mean(results['mae']),
            'rmse': np.mean(results['rmse'])
        }
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        raise

def rolling_window_cv(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, List[float]]:
    """Perform rolling window cross-validation."""
    try:
        cv_results = {
            'accuracy': [],
            'partial_matches': [],
            'mae': [],
            'rmse': []
        }
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tqdm(tscv.split(df), desc="Cross-validation", total=n_splits):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # Train models
            from scripts.train_models import train_all_models
            models = train_all_models(train_df)
            
            # Make predictions for each test row
            for _, test_row in test_df.iterrows():
                predictions = predict_next_draw(models, train_df)
                actual_numbers = sorted([test_row[col] for col in df.columns if col.startswith('Number')])
                
                # Calculate metrics
                best_match = 0
                for pred in predictions:
                    matches = len(set(pred) & set(actual_numbers))
                    best_match = max(best_match, matches)
                
                cv_results['accuracy'].append(1 if best_match == 6 else 0)
                cv_results['partial_matches'].append(best_match / 6)
                
                # Calculate regression metrics
                pred_array = np.array(predictions[0])
                actual_array = np.array(actual_numbers)
                cv_results['mae'].append(mean_absolute_error(actual_array, pred_array))
                cv_results['rmse'].append(np.sqrt(mean_squared_error(actual_array, pred_array)))
        
        return cv_results
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        raise

def test_randomness(df: pd.DataFrame) -> Dict[str, float]:
    """Test for randomness in predictions."""
    try:
        from scipy import stats
        
        # Get predictions
        models = load_models()
        predictions = predict_next_draw(models, df)
        flat_predictions = np.array(predictions).ravel()
        
        results = {}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(flat_predictions, 'uniform', 
                                      args=(1, max(flat_predictions)))
        results['ks_pvalue'] = ks_pval
        
        # Runs test
        median = np.median(flat_predictions)
        runs = np.where(flat_predictions > median, 1, 0)
        runs_stat, runs_pval = stats.runs_test(runs)
        results['runs_pvalue'] = runs_pval
        
        # Chi-square test
        observed = pd.Series(flat_predictions).value_counts()
        expected = np.ones_like(observed) * len(flat_predictions) / len(observed)
        chi2, chi2_pval = stats.chisquare(observed, expected)
        results['chi2_pvalue'] = chi2_pval
        
        return results
    except Exception as e:
        logger.error(f"Error testing randomness: {e}")
        raise

if __name__ == "__main__":
    # Load data and models
    df = load_data(Path(DATA_FILE))
    models = load_models()
    
    # Make predictions
    predictions = predict_next_draw(models, df)
    print("\nPredicted combinations for next draw:")
    for i, combination in enumerate(predictions[:5], 1):
        print(f"Combination {i}: {combination}")
    
    # Run backtesting
    print("\nRunning backtesting...")
    backtest_results = backtest(df)
    print("\nBacktesting results:")
    for metric, value in backtest_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Test randomness
    print("\nTesting randomness...")
    randomness_results = test_randomness(df)
    print("\nRandomness test results:")
    for test, pvalue in randomness_results.items():
        print(f"{test}: {pvalue:.4f}") 