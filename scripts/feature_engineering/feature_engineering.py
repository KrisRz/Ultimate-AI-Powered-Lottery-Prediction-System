import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import logging
from scipy import stats
import json
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
import gc
import time
import pickle
import math

# Setup logging
logger = logging.getLogger(__name__)

def load_feature_config(config_path: str = "config/data_config.json") -> Dict:
    """
    Load feature engineering configuration from config file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing feature engineering configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('feature_engineering', {})
    except Exception as e:
        logger.error(f"Error loading feature config: {e}")
        return {}

def create_temporal_features(df: pd.DataFrame, date_col: str = 'Draw Date') -> pd.DataFrame:
    """
    Create temporal features from date column.
    
    Args:
        df: DataFrame containing lottery data
        date_col: Name of the date column
        
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Ensure date column is datetime type
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract basic date components
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    df['DayOfWeek'] = df[date_col].dt.dayofweek
    df['Quarter'] = df[date_col].dt.quarter
    df['WeekOfYear'] = df[date_col].dt.isocalendar().week
    
    # Days since the start of the dataset
    min_date = df[date_col].min()
    df['DaysSinceStart'] = (df[date_col] - min_date).dt.days
    
    # Calculate gap between draws
    df['DaysSincePreviousDraw'] = df[date_col].diff().dt.days
    
    # Cyclical encoding of temporal features (sin/cos transformation)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    return df

def create_number_features(df: pd.DataFrame, numbers_col: str = 'Main_Numbers') -> pd.DataFrame:
    """
    Create features based on the lottery numbers themselves.
    
    Args:
        df: DataFrame containing lottery data
        numbers_col: Name of the column containing list of drawn numbers
        
    Returns:
        DataFrame with additional number-based features
    """
    df = df.copy()
    
    # Basic statistical features
    df['Sum'] = df[numbers_col].apply(sum)
    df['Mean'] = df[numbers_col].apply(np.mean)
    df['Std'] = df[numbers_col].apply(np.std)
    df['Median'] = df[numbers_col].apply(np.median)
    df['Range'] = df[numbers_col].apply(lambda x: max(x) - min(x))
    df['Unique'] = df[numbers_col].apply(lambda x: len(set(x)))
    
    # Number properties
    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    fibonacci_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    df['Primes'] = df[numbers_col].apply(lambda x: sum(1 for n in x if n in prime_numbers))
    df['Fibonacci'] = df[numbers_col].apply(lambda x: sum(1 for n in x if n in fibonacci_numbers))
    df['Odds'] = df[numbers_col].apply(lambda x: sum(1 for n in x if n % 2 == 1))
    df['Evens'] = df[numbers_col].apply(lambda x: sum(1 for n in x if n % 2 == 0))
    df['Low_Numbers'] = df[numbers_col].apply(lambda x: sum(1 for n in x if n <= 30))
    df['High_Numbers'] = df[numbers_col].apply(lambda x: sum(1 for n in x if n > 30))
    
    # Numerical patterns
    df['Consecutive_Pairs'] = df[numbers_col].apply(
        lambda x: sum(1 for i in range(len(x)-1) if x[i+1] - x[i] == 1)
    )
    df['Gaps'] = df[numbers_col].apply(
        lambda x: np.mean([x[i+1] - x[i] for i in range(len(x)-1)])
    )
    df['Max_Gap'] = df[numbers_col].apply(
        lambda x: max([x[i+1] - x[i] for i in range(len(x)-1)])
    )
    
    # Digit-based features
    df['Sum_Digits'] = df[numbers_col].apply(
        lambda x: sum(sum(int(digit) for digit in str(num)) for num in x)
    )
    df['Numbers_Ending_Same'] = df[numbers_col].apply(
        lambda x: max(sum(1 for num in x if num % 10 == digit) for digit in range(10))
    )
    
    # Number decades (e.g., 1-10, 11-20, etc.)
    for i in range(6):
        decade_start = i * 10 + 1
        decade_end = (i + 1) * 10
        df[f'Decade_{decade_start}_{decade_end}'] = df[numbers_col].apply(
            lambda x: sum(1 for n in x if decade_start <= n <= decade_end)
        )
    
    return df

def create_frequency_features(df: pd.DataFrame, numbers_col: str = 'Main_Numbers', 
                             windows: List[int] = [10, 20, 50, 100]) -> pd.DataFrame:
    """
    Create features based on frequency analysis of lottery numbers.
    
    Args:
        df: DataFrame containing lottery data
        numbers_col: Name of the column containing list of drawn numbers
        windows: List of window sizes for rolling statistics
        
    Returns:
        DataFrame with additional frequency-based features
    """
    df = df.copy()
    
    # Calculate number frequencies across the entire dataset
    all_numbers = np.concatenate(df[numbers_col].values)
    number_counts = pd.Series(all_numbers).value_counts()
    
    # Average frequency of chosen numbers
    df['Number_Frequency'] = df[numbers_col].apply(
        lambda x: sum(number_counts.get(num, 0) for num in x) / len(x)
    )
    
    # Standard deviation of number frequencies
    df['Number_Frequency_Std'] = df[numbers_col].apply(
        lambda x: np.std([number_counts.get(num, 0) for num in x])
    )
    
    # Days since each number last appeared
    for window in windows:
        if len(df) < window:
            continue
            
        # Rolling statistics
        df[f'Sum_MA_{window}'] = df['Sum'].rolling(window=window, min_periods=1).mean()
        df[f'Sum_STD_{window}'] = df['Sum'].rolling(window=window, min_periods=1).std()
        df[f'Mean_MA_{window}'] = df['Mean'].rolling(window=window, min_periods=1).mean()
        df[f'Mean_STD_{window}'] = df['Mean'].rolling(window=window, min_periods=1).std()
        
        # Calculate number hotness/coldness in window
        for i in range(len(df) - window + 1):
            window_numbers = np.concatenate(df[numbers_col].iloc[i:i+window].values)
            window_counts = pd.Series(window_numbers).value_counts()
            
            # Calculate average hotness for each draw in window
            for j in range(i, i+window):
                if j < len(df):
                    nums = df[numbers_col].iloc[j]
                    df.loc[df.index[j], f'Hotness_{window}'] = np.mean([window_counts.get(num, 0) for num in nums])
    
    return df

def create_lag_features(df: pd.DataFrame, target_cols: List[str], lag_periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Create lag features from target columns.
    
    Args:
        df: DataFrame containing lottery data
        target_cols: List of columns to create lag features from
        lag_periods: List of lag periods
        
    Returns:
        DataFrame with additional lag features
    """
    df = df.copy()
    
    for col in target_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping.")
            continue
            
        for lag in lag_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def create_combination_features(df: pd.DataFrame, numbers_col: str = 'Main_Numbers') -> pd.DataFrame:
    """
    Create features based on number combinations.
    
    Args:
        df: DataFrame containing lottery data
        numbers_col: Name of the column containing list of drawn numbers
        
    Returns:
        DataFrame with additional combination-based features
    """
    df = df.copy()
    
    # Extract all possible pairs
    all_pairs = []
    for numbers in df[numbers_col]:
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                all_pairs.append((min(numbers[i], numbers[j]), max(numbers[i], numbers[j])))
    
    # Count pair frequencies
    pair_counts = pd.Series(all_pairs).value_counts()
    top_pairs = pair_counts.head(30).index.tolist()
    
    # Create features for top pairs
    for i, pair in enumerate(top_pairs):
        df[f'Pair_{pair[0]}_{pair[1]}'] = df[numbers_col].apply(
            lambda x: 1 if pair[0] in x and pair[1] in x else 0
        )
    
    return df

def engineer_features(df: pd.DataFrame, config_path: str = "config/data_config.json") -> pd.DataFrame:
    """
    Main function to apply all feature engineering steps.
    
    Args:
        df: DataFrame containing lottery data
        config_path: Path to the configuration file
        
    Returns:
        DataFrame with all engineered features
    """
    try:
        df = df.copy()
        config = load_feature_config(config_path)
        
        # Apply temporal features
        if config.get('temporal_features', {}).get('enabled', True):
            df = create_temporal_features(df)
            logger.info("Added temporal features")
        
        # Apply number-based features
        df = create_number_features(df)
        logger.info("Added number-based features")
        
        # Apply frequency-based features
        if config.get('frequency_features', {}).get('enabled', True):
            windows = config.get('frequency_features', {}).get('rolling_windows', [10, 20, 50, 100])
            df = create_frequency_features(df, windows=windows)
            logger.info("Added frequency-based features")
        
        # Apply lag features
        lag_cols = ['Sum', 'Mean', 'Primes', 'Odds', 'Evens', 'Number_Frequency']
        lag_periods = [1, 2, 3, 5, 10]
        df = create_lag_features(df, lag_cols, lag_periods)
        logger.info("Added lag features")
        
        # Apply combination features
        if config.get('frequency_features', {}).get('number_combinations', {}).get('enabled', True):
            df = create_combination_features(df)
            logger.info("Added combination features")
        
        # Drop rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with NaN values")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise

def enhanced_time_series_features(df: pd.DataFrame, 
                                look_back: int = 200, 
                                use_tsfresh: bool = True,
                                cache_features: bool = True) -> pd.DataFrame:
    """
    Generate enhanced time series features for lottery prediction models
    
    Args:
        df: DataFrame with lottery data
        look_back: Number of previous draws to use in feature calculations
        use_tsfresh: Whether to use tsfresh for automatic feature extraction
        cache_features: Whether to cache computed features to disk
        
    Returns:
        DataFrame with enhanced features
    """
    start_time = time.time()
    logger.info(f"Generating enhanced time series features from {len(df)} draws with look_back={look_back}")
    
    cache_path = Path("outputs/outputs/results/enhanced_features_cache.pkl")
    
    # Check for cached features
    if cache_features and cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                if (cache.get('data_hash') == hash(str(df.head(10)) + str(df.tail(10))) and
                    cache.get('look_back') == look_back and
                    cache.get('use_tsfresh') == use_tsfresh):
                    logger.info(f"Using cached features from {cache_path}")
                    return cache['features']
        except Exception as e:
            logger.warning(f"Could not load cached features: {str(e)}")
    
    # Make a copy to avoid modifying original
    df_enhanced = df.copy()
    
    # Ensure Main_Numbers is present and properly formatted
    if 'Main_Numbers' not in df.columns:
        raise ValueError("DataFrame must contain 'Main_Numbers' column")
    
    # Check if Main_Numbers is a list in each row
    if not isinstance(df.iloc[0]['Main_Numbers'], (list, np.ndarray)):
        raise ValueError("Main_Numbers must be lists of integers")
    
    # Extract basic statistics from each draw
    df_enhanced['sum'] = df['Main_Numbers'].apply(sum)
    df_enhanced['mean'] = df['Main_Numbers'].apply(np.mean)
    df_enhanced['median'] = df['Main_Numbers'].apply(np.median)
    df_enhanced['std'] = df['Main_Numbers'].apply(np.std)
    df_enhanced['variance'] = df['Main_Numbers'].apply(np.var)
    df_enhanced['range'] = df['Main_Numbers'].apply(lambda x: max(x) - min(x))
    df_enhanced['evenness'] = df['Main_Numbers'].apply(lambda x: sum(1 for n in x if n % 2 == 0) / len(x))
    
    # Calculate advanced number properties
    df_enhanced['entropy'] = df['Main_Numbers'].apply(
        lambda x: -sum((n/sum(x)) * np.log(n/sum(x)) if n > 0 else 0 for n in x)
    )
    
    # Generate rolling statistics over multiple windows
    for window in [5, 10, 20, 50, look_back]:
        if len(df) > window:
            for col in ['sum', 'mean', 'std']:
                df_enhanced[f'rolling_{col}_{window}'] = df_enhanced[col].rolling(window=window, min_periods=1).mean()
                df_enhanced[f'rolling_{col}_{window}_std'] = df_enhanced[col].rolling(window=window, min_periods=1).std()
    
    # Create lag features
    for lag in [1, 2, 3, 5, 10]:
        if len(df) > lag:
            for col in ['sum', 'mean', 'std', 'range']:
                df_enhanced[f'{col}_lag_{lag}'] = df_enhanced[col].shift(lag)
    
    # Difference features
    for lag in [1, 2, 5]:
        if len(df) > lag:
            for col in ['sum', 'mean']:
                df_enhanced[f'{col}_diff_{lag}'] = df_enhanced[col].diff(lag)
    
    # Exponential weighted features
    for span in [5, 10, 20]:
        if len(df) > span:
            for col in ['sum', 'mean']:
                df_enhanced[f'{col}_ewm_{span}'] = df_enhanced[col].ewm(span=span).mean()
    
    # Create frequency-based features
    number_freq = {}
    
    # Number frequency over different windows
    for window in [20, 50, 100, look_back]:
        if len(df) > window:
            for i in range(max(0, len(df) - window), len(df)):
                for num in df.iloc[i]['Main_Numbers']:
                    number_freq.setdefault(window, {}).setdefault(num, 0)
                    number_freq[window][num] += 1
    
    # Apply frequency features
    for window in number_freq.keys():
        df_enhanced[f'freq_{window}'] = df['Main_Numbers'].apply(
            lambda x: sum(number_freq[window].get(num, 0) for num in x) / len(x) if window in number_freq else 0
        )
    
    # Pattern-based features
    df_enhanced['consecutive_pairs'] = df['Main_Numbers'].apply(
        lambda x: sum(1 for i in range(len(x)-1) if x[i+1] - x[i] == 1)
    )
    
    df_enhanced['has_triplet'] = df['Main_Numbers'].apply(
        lambda x: 1 if any(x[i+2] - x[i] == 2 and x[i+1] - x[i] == 1 for i in range(len(x)-2)) else 0
    )
    
    # Statistical distribution features
    df_enhanced['skewness'] = df['Main_Numbers'].apply(
        lambda x: pd.Series(x).skew() if len(x) > 2 else 0
    )
    
    df_enhanced['kurtosis'] = df['Main_Numbers'].apply(
        lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0
    )
    
    # Advanced statistical features
    df_enhanced['coefficient_variation'] = df_enhanced['std'] / df_enhanced['mean']
    
    # Number distribution features
    for decade in range(6):
        lower = decade * 10 + 1
        upper = (decade + 1) * 10
        df_enhanced[f'decade_{lower}_{upper}'] = df['Main_Numbers'].apply(
            lambda x: sum(1 for n in x if lower <= n <= upper) / len(x)
        )
    
    # Add TSFresh features if requested
    if use_tsfresh and len(df) > 50:  # Only if we have enough data
        try:
            logger.info("Extracting TSFresh features...")
            
            # Prepare data for tsfresh
            ts_data = []
            for idx, row in df.iterrows():
                for i, num in enumerate(row['Main_Numbers']):
                    ts_data.append({
                        'id': idx,
                        'time': i,
                        'value': num
                    })
            
            ts_df = pd.DataFrame(ts_data)
            
            # Extract features with efficient parameters
            fc_params = EfficientFCParameters()
            tsfresh_features = extract_features(ts_df, 
                                          column_id='id', 
                                          column_sort='time',
                                          column_value='value',
                                          default_fc_parameters=fc_params,
                                          n_jobs=0)  # Use all available cores
            
            # Keep only the most useful features
            tsfresh_features = tsfresh_features.dropna(axis=1, how='all')
            useful_cols = [col for col in tsfresh_features.columns 
                          if not (tsfresh_features[col].isna().sum() > 0.5 * len(tsfresh_features))]
            
            logger.info(f"Generated {len(useful_cols)} TSFresh features")
            
            # Merge with main features
            df_enhanced = pd.concat([df_enhanced, tsfresh_features[useful_cols]], axis=1)
            
        except Exception as e:
            logger.error(f"Error generating TSFresh features: {str(e)}")
    
    # Fill any missing values
    df_enhanced = df_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Cache the outputs/results if requested
    if cache_features:
        try:
            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, 'wb') as f:
                cache = {
                    'features': df_enhanced,
                    'data_hash': hash(str(df.head(10)) + str(df.tail(10))),
                    'look_back': look_back,
                    'use_tsfresh': use_tsfresh,
                    'timestamp': time.time()
                }
                pickle.dump(cache, f)
                logger.info(f"Cached enhanced features to {cache_path}")
        except Exception as e:
            logger.warning(f"Could not cache features: {str(e)}")
    
    # Log timing
    duration = time.time() - start_time
    logger.info(f"Generated {len(df_enhanced.columns)} enhanced features in {duration:.2f} seconds")
    
    return df_enhanced

def prepare_lstm_features(df: pd.DataFrame, 
                         look_back: int = 200, 
                         target_col: str = 'Main_Numbers',
                         scale_method: str = 'robust',
                         use_pca: bool = False,
                         pca_components: int = 20) -> Tuple[np.ndarray, np.ndarray, Union[StandardScaler, RobustScaler]]:
    """
    Prepare optimized features for LSTM models from lottery data
    
    Args:
        df: DataFrame with lottery data
        look_back: Number of previous timesteps to use
        target_col: Column containing lottery numbers
        scale_method: Scaling method ('standard', 'robust', or 'power')
        use_pca: Whether to apply PCA dimensionality reduction
        pca_components: Number of PCA components to keep if using PCA
        
    Returns:
        Tuple of (X_features, y_targets, scaler)
    """
    start_time = time.time()
    logger.info(f"Preparing LSTM features from {len(df)} draws with look_back={look_back}")
    
    # Generate enhanced features
    df_features = enhanced_time_series_features(df, look_back)
    
    # Select numerical columns for features
    feature_cols = df_features.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude target column and direct number columns if they exist
    exclude_patterns = [target_col, 'Number_', 'Bonus', 'Jackpot', 'Winners']
    feature_cols = [col for col in feature_cols if not any(pattern in col for pattern in exclude_patterns)]
    
    # Create the feature matrix
    X = df_features[feature_cols].values
    
    # Normalize/scale the features
    if scale_method == 'robust':
        scaler = RobustScaler()
    elif scale_method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:  # default to standard
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA if requested
    if use_pca and len(feature_cols) > pca_components:
        try:
            pca = PCA(n_components=pca_components)
            X_scaled = pca.fit_transform(X_scaled)
            explained_var = sum(pca.explained_variance_ratio_) * 100
            logger.info(f"Applied PCA: reduced from {len(feature_cols)} to {pca_components} features "
                      f"({explained_var:.2f}% variance explained)")
        except Exception as e:
            logger.error(f"Error applying PCA: {str(e)}")
    
    # Create sequences for LSTM
    X_sequences = []
    y_targets = []
    
    for i in range(len(X_scaled) - look_back):
        X_sequences.append(X_scaled[i:i + look_back])
        y_targets.append(df[target_col].iloc[i + look_back])
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    
    # Reshape if needed
    if len(X_sequences.shape) == 2:  # For single feature
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
    
    # Ensure y_targets is properly formatted
    y_arr = np.array([list(t) for t in y_targets])
    
    # Log the shapes and timing
    duration = time.time() - start_time
    logger.info(f"LSTM feature preparation complete in {duration:.2f} seconds")
    logger.info(f"Shapes: X={X_sequences.shape}, y={y_arr.shape}")
    
    # Clean up to save memory
    gc.collect()
    
    return X_sequences, y_arr, scaler

def expand_draw_sequences(df: pd.DataFrame, n_prev_draws: int = 5) -> pd.DataFrame:
    """
    Expand DataFrame to include previous draws as features
    
    Args:
        df: DataFrame with lottery data
        n_prev_draws: Number of previous draws to include
        
    Returns:
        Expanded DataFrame with previous draw features
    """
    logger.info(f"Expanding draw sequences with {n_prev_draws} previous draws")
    
    # Create a copy to avoid modifying original
    df_expanded = df.copy()
    
    # For each previous draw, add features
    for i in range(1, n_prev_draws + 1):
        if i < len(df):
            # Previous draw numbers
            df_expanded[f'prev_{i}_numbers'] = df['Main_Numbers'].shift(i)
            
            # Previous draw statistics
            for stat in ['sum', 'mean', 'std']:
                if stat in df.columns:
                    df_expanded[f'prev_{i}_{stat}'] = df[stat].shift(i)
    
    # Drop rows with missing data due to shifts
    df_expanded = df_expanded.dropna()
    
    return df_expanded

def calculate_winning_probabilities(df: pd.DataFrame) -> Dict:
    """
    Calculate probability weights for lottery numbers based on historical data
    
    Args:
        df: DataFrame with lottery draw history
        
    Returns:
        Dictionary of probability weights for each number
    """
    # Extract all drawn numbers
    all_numbers = [num for draw in df['Main_Numbers'] for num in draw]
    
    # Count frequency of each number
    number_counts = pd.Series(all_numbers).value_counts()
    
    # Calculate probability weights
    total_draws = len(df)
    total_numbers = len(all_numbers)
    
    weights = {}
    
    # Base frequency weights
    for num in range(1, 60):
        count = number_counts.get(num, 0)
        weights[num] = count / total_numbers
    
    # Apply Bayesian smoothing to avoid zero probabilities
    alpha = 1.0  # Pseudo-count
    for num in range(1, 60):
        weights[num] = (weights.get(num, 0) * total_numbers + alpha) / (total_numbers + 59 * alpha)
    
    return weights

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Check for data file
        data_path = Path("data/lottery_data_1995_2025.csv")
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            sys.exit(1)
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Convert string representation of lists to actual lists if needed
        if 'Main_Numbers' in df.columns and isinstance(df['Main_Numbers'].iloc[0], str):
            df['Main_Numbers'] = df['Main_Numbers'].apply(eval)
        
        # Test feature engineering
        enhanced_df = enhanced_time_series_features(df, look_back=100)
        print(f"Generated {len(enhanced_df.columns)} enhanced features")
        
        # Test LSTM feature preparation
        X, y, scaler = prepare_lstm_features(df, look_back=50)
        print(f"Prepared LSTM features with shapes: X={X.shape}, y={y.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 