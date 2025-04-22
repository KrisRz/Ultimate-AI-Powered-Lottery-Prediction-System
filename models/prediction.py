import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from .utils import ensure_valid_prediction, log_memory_usage, LOOK_BACK, N_PREDICTIONS
from .base import BaseModel, TimeSeriesModel, EnsembleModel, MetaModel

def predict_with_lstm(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using LSTM model."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            lstm_model, scaler = models['lstm']
            all_numbers = np.array([num for draw in df['Main_Numbers'] for num in draw]).reshape(-1, 1)
            X_scaled = scaler.transform(all_numbers[-LOOK_BACK:])
            X_reshaped = X_scaled.reshape((1, LOOK_BACK, 1))
            pred = lstm_model.predict(X_reshaped, verbose=0)
            pred = scaler.inverse_transform(pred)[0]
            return ensure_valid_prediction(pred)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_arima(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using ARIMA models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            arima_models = models['arima']
            predictions = []
            for model in arima_models:
                if model is not None:
                    pred = model.predict(n_periods=1)[0]
                    predictions.append(int(round(pred)))
                else:
                    predictions.append(np.random.randint(1, 60))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_holtwinters(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using Holt-Winters models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            holtwinters_models = models['holtwinters']
            predictions = []
            for model in holtwinters_models:
                pred = model.forecast(1)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_linear(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using Linear Regression models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            linear_models = models['linear']
            X = pd.DataFrame({
                'index': [len(df)],
                'dayofweek': pd.to_numeric(df['DayOfWeek'].iloc[-1], errors='coerce').fillna(0),
                'sum': df['Sum'].iloc[-1],
                'mean': df['Mean'].iloc[-1],
                'unique': df['Unique'].iloc[-1],
                'zscore_sum': df['ZScore_Sum'].iloc[-1],
                'primes': df['Primes'].iloc[-1],
                'odds': df['Odds'].iloc[-1],
                'gaps': df['Gaps'].iloc[-1],
                'freq_10': df['Freq_10'].iloc[-1],
                'freq_20': df['Freq_20'].iloc[-1],
                'freq_50': df['Freq_50'].iloc[-1],
                'pair_freq': df['Pair_Freq'].iloc[-1],
                'triplet_freq': df['Triplet_Freq'].iloc[-1]
            })
            predictions = []
            for model in linear_models:
                pred = model.predict(X)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_xgboost(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using XGBoost models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            xgboost_models = models['xgboost']
            X = pd.DataFrame({
                'index': [len(df)],
                'dayofweek': pd.to_numeric(df['DayOfWeek'].iloc[-1], errors='coerce').fillna(0),
                'sum': df['Sum'].iloc[-1],
                'mean': df['Mean'].iloc[-1],
                'unique': df['Unique'].iloc[-1],
                'zscore_sum': df['ZScore_Sum'].iloc[-1],
                'primes': df['Primes'].iloc[-1],
                'odds': df['Odds'].iloc[-1],
                'gaps': df['Gaps'].iloc[-1],
                'freq_10': df['Freq_10'].iloc[-1],
                'freq_20': df['Freq_20'].iloc[-1],
                'freq_50': df['Freq_50'].iloc[-1],
                'pair_freq': df['Pair_Freq'].iloc[-1],
                'triplet_freq': df['Triplet_Freq'].iloc[-1]
            })
            predictions = []
            for model in xgboost_models:
                pred = model.predict(X)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_lightgbm(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using LightGBM models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            lightgbm_models = models['lightgbm']
            X = pd.DataFrame({
                'index': [len(df)],
                'dayofweek': pd.to_numeric(df['DayOfWeek'].iloc[-1], errors='coerce').fillna(0),
                'sum': df['Sum'].iloc[-1],
                'mean': df['Mean'].iloc[-1],
                'unique': df['Unique'].iloc[-1],
                'zscore_sum': df['ZScore_Sum'].iloc[-1],
                'primes': df['Primes'].iloc[-1],
                'odds': df['Odds'].iloc[-1],
                'gaps': df['Gaps'].iloc[-1],
                'freq_10': df['Freq_10'].iloc[-1],
                'freq_20': df['Freq_20'].iloc[-1],
                'freq_50': df['Freq_50'].iloc[-1],
                'pair_freq': df['Pair_Freq'].iloc[-1],
                'triplet_freq': df['Triplet_Freq'].iloc[-1]
            })
            predictions = []
            for model in lightgbm_models:
                pred = model.predict(X)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_knn(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using KNN models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            knn_models = models['knn']
            X = pd.DataFrame({
                'index': [len(df)],
                'dayofweek': pd.to_numeric(df['DayOfWeek'].iloc[-1], errors='coerce').fillna(0),
                'sum': df['Sum'].iloc[-1],
                'mean': df['Mean'].iloc[-1],
                'unique': df['Unique'].iloc[-1],
                'zscore_sum': df['ZScore_Sum'].iloc[-1],
                'primes': df['Primes'].iloc[-1],
                'odds': df['Odds'].iloc[-1],
                'gaps': df['Gaps'].iloc[-1],
                'freq_10': df['Freq_10'].iloc[-1],
                'freq_20': df['Freq_20'].iloc[-1],
                'freq_50': df['Freq_50'].iloc[-1],
                'pair_freq': df['Pair_Freq'].iloc[-1],
                'triplet_freq': df['Triplet_Freq'].iloc[-1]
            })
            predictions = []
            for model in knn_models:
                pred = model.predict(X)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_gradientboosting(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using Gradient Boosting models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            gb_models = models['gradientboosting']
            X = pd.DataFrame({
                'index': [len(df)],
                'dayofweek': pd.to_numeric(df['DayOfWeek'].iloc[-1], errors='coerce').fillna(0),
                'sum': df['Sum'].iloc[-1],
                'mean': df['Mean'].iloc[-1],
                'unique': df['Unique'].iloc[-1],
                'zscore_sum': df['ZScore_Sum'].iloc[-1],
                'primes': df['Primes'].iloc[-1],
                'odds': df['Odds'].iloc[-1],
                'gaps': df['Gaps'].iloc[-1],
                'freq_10': df['Freq_10'].iloc[-1],
                'freq_20': df['Freq_20'].iloc[-1],
                'freq_50': df['Freq_50'].iloc[-1],
                'pair_freq': df['Pair_Freq'].iloc[-1],
                'triplet_freq': df['Triplet_Freq'].iloc[-1]
            })
            predictions = []
            for model in gb_models:
                pred = model.predict(X)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_catboost(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using CatBoost models."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            catboost_models = models['catboost']
            X = pd.DataFrame({
                'index': [len(df)],
                'dayofweek': pd.to_numeric(df['DayOfWeek'].iloc[-1], errors='coerce').fillna(0),
                'sum': df['Sum'].iloc[-1],
                'mean': df['Mean'].iloc[-1],
                'unique': df['Unique'].iloc[-1],
                'zscore_sum': df['ZScore_Sum'].iloc[-1],
                'primes': df['Primes'].iloc[-1],
                'odds': df['Odds'].iloc[-1],
                'gaps': df['Gaps'].iloc[-1],
                'freq_10': df['Freq_10'].iloc[-1],
                'freq_20': df['Freq_20'].iloc[-1],
                'freq_50': df['Freq_50'].iloc[-1],
                'pair_freq': df['Pair_Freq'].iloc[-1],
                'triplet_freq': df['Triplet_Freq'].iloc[-1]
            })
            predictions = []
            for model in catboost_models:
                pred = model.predict(X)[0]
                predictions.append(int(round(pred)))
            return ensure_valid_prediction(predictions)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_cnn_lstm(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using CNN-LSTM model."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            cnn_lstm_model, scaler = models['cnn_lstm']
            all_numbers = np.array([num for draw in df['Main_Numbers'] for num in draw]).reshape(-1, 1)
            X_scaled = scaler.transform(all_numbers[-LOOK_BACK:])
            X_reshaped = X_scaled.reshape((1, LOOK_BACK, 1))
            pred = cnn_lstm_model.predict(X_reshaped, verbose=0)
            pred = scaler.inverse_transform(pred)[0]
            return ensure_valid_prediction(pred)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_autoencoder(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using Autoencoder model."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            autoencoder_model, scaler = models['autoencoder']
            all_numbers = np.array([num for draw in df['Main_Numbers'] for num in draw]).reshape(-1, 1)
            X_scaled = scaler.transform(all_numbers[-LOOK_BACK:])
            X_reshaped = X_scaled.reshape((1, LOOK_BACK, 1))
            pred = autoencoder_model.predict(X_reshaped, verbose=0)
            pred = scaler.inverse_transform(pred)[0]
            return ensure_valid_prediction(pred)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_with_meta(df: pd.DataFrame, models: Dict) -> List[int]:
    """Predict next numbers using Meta model."""
    try:
        if not df.empty and 'Main_Numbers' in df.columns:
            meta_model = models['meta']
            predictions = []
            for model_name, model in models.items():
                if model_name != 'meta':
                    if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
                        pred = model[0].predict(df)
                    else:
                        pred = model.predict(df)
                    predictions.append(pred)
            X_meta = np.array(predictions)
            pred = meta_model.predict(X_meta)[0]
            return ensure_valid_prediction(pred)
        else:
            return ensure_valid_prediction([])
    except Exception as e:
        return ensure_valid_prediction([])

def predict_next_draw(df: pd.DataFrame, models: Dict) -> Dict[str, List[int]]:
    """Predict next draw numbers using all models."""
    predictions = {}
    prediction_functions = {
        'lstm': predict_with_lstm,
        'arima': predict_with_arima,
        'holtwinters': predict_with_holtwinters,
        'linear': predict_with_linear,
        'xgboost': predict_with_xgboost,
        'lightgbm': predict_with_lightgbm,
        'knn': predict_with_knn,
        'gradientboosting': predict_with_gradientboosting,
        'catboost': predict_with_catboost,
        'cnn_lstm': predict_with_cnn_lstm,
        'autoencoder': predict_with_autoencoder,
        'meta': predict_with_meta
    }
    
    for model_name, predict_func in prediction_functions.items():
        if model_name in models:
            predictions[model_name] = predict_func(df, models)
    
    return predictions

def ensemble_prediction(df: pd.DataFrame, models: Dict, n_predictions: int = 10) -> List[List[int]]:
    """Generate ensemble predictions using multiple models."""
    predictions = []
    for _ in range(n_predictions):
        model_predictions = []
        for model_name, model in models.items():
            if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
                pred = model[0].predict(df)
            else:
                pred = model.predict(df)
            model_predictions.append(pred)
        
        # Combine predictions using voting
        combined = []
        for i in range(6):
            votes = Counter([pred[i] for pred in model_predictions])
            combined.append(votes.most_common(1)[0][0])
        predictions.append(ensure_valid_prediction(combined))
    
    return predictions

def generate_optimized_predictions(df: pd.DataFrame, models: Dict, n_predictions: int = 10) -> List[List[int]]:
    """Generate optimized predictions using multiple models and scoring."""
    predictions = []
    for _ in range(n_predictions):
        model_predictions = []
        for model_name, model in models.items():
            if model_name in ['lstm', 'cnn_lstm', 'autoencoder']:
                pred = model[0].predict(df)
            else:
                pred = model.predict(df)
            model_predictions.append(pred)
        
        # Score and select best predictions
        scored = score_combinations(model_predictions, df)
        best_pred = max(scored, key=lambda x: x[1])[0]
        predictions.append(ensure_valid_prediction(best_pred))
    
    return predictions

def score_combinations(predictions: List[List[int]], df: pd.DataFrame) -> List[Tuple[List[int], float]]:
    """Score combinations of predictions based on historical patterns."""
    scored = []
    for pred in predictions:
        score = 0
        
        # Check number properties
        score += sum(1 for n in pred if n in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])  # Primes
        score += sum(1 for n in pred if n % 2 == 1)  # Odds
        score += sum(1 for n in pred if n <= 30)  # Low numbers
        
        # Check historical patterns
        all_numbers = df['Main_Numbers'].tolist()
        window_size = min(50, len(all_numbers))
        window_numbers = all_numbers[-window_size:]
        
        # Check pair frequency
        pairs = list(combinations(pred, 2))
        pair_freq = sum(1 for p in pairs for draw in window_numbers if p[0] in draw and p[1] in draw)
        score += pair_freq / len(window_numbers)
        
        # Check triplet frequency
        triplets = list(combinations(pred, 3))
        triplet_freq = sum(1 for t in triplets for draw in window_numbers if t[0] in draw and t[1] in draw and t[2] in draw)
        score += triplet_freq / len(window_numbers)
        
        scored.append((pred, score))
    
    return scored

def monte_carlo_simulation(df: pd.DataFrame, n_simulations: int = 1000) -> List[List[int]]:
    """Generate predictions using Monte Carlo simulation."""
    all_numbers = [num for draw in df['Main_Numbers'] for num in draw]
    freq = Counter(all_numbers)
    total = sum(freq.values())
    probs = {n: c/total for n, c in freq.items()}
    
    predictions = []
    for _ in range(n_simulations):
        pred = []
        while len(pred) < 6:
            n = np.random.choice(list(probs.keys()), p=list(probs.values()))
            if n not in pred:
                pred.append(n)
        predictions.append(sorted(pred))
    
    return predictions

def backtest(df: pd.DataFrame, models: Dict = None) -> Tuple[float, List[List[int]]]:
    """Backtest models on historical data."""
    if models is None:
        models = update_models(df)
    
    predictions = []
    actual = df['Main_Numbers'].tolist()
    
    for i in range(len(df) - 1):
        train_df = df.iloc[:i+1]
        test_df = df.iloc[i+1:i+2]
        
        if i % 10 == 0:  # Retrain models every 10 draws
            models = update_models(train_df)
        
        pred = predict_next_draw(train_df, models)
        predictions.append(list(pred.values())[0])  # Use first model's prediction
    
    accuracy = calculate_prediction_accuracy(predictions, actual[1:])
    return accuracy, predictions

def calculate_prediction_accuracy(predictions: List[List[int]], actual: List[int]) -> float:
    """Calculate prediction accuracy."""
    correct = 0
    total = 0
    
    for pred, act in zip(predictions, actual):
        matches = len(set(pred) & set(act))
        correct += matches
        total += 6
    
    return correct / total if total > 0 else 0 