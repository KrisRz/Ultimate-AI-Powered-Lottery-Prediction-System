import numpy as np
import pandas as pd
import os
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("data/lottery_data_1995_2025.csv")
MODELS_DIR = Path("models/checkpoints/simple")
MODELS_PATH = MODELS_DIR / "lightgbm_model.pkl"

def ensure_valid_prediction(prediction, min_val=1, max_val=59, required_length=6):
    """Ensure a prediction contains valid lottery numbers"""
    # Convert to numpy array if it's not already
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)
    
    # Round and clip values
    prediction = np.round(prediction).astype(int)
    prediction = np.clip(prediction, min_val, max_val)
    
    # Ensure unique values
    unique_nums = set(prediction)
    while len(unique_nums) < required_length:
        new_num = np.random.randint(min_val, max_val + 1)
        if new_num not in unique_nums:
            unique_nums.add(new_num)
    
    # Sort and return required number of values
    return sorted(list(unique_nums))[:required_length]

def load_data(data_path=DATA_PATH):
    """Load lottery data from CSV file"""
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        # Process the Main_Numbers columns - convert string to list
        if 'Main_Numbers' in df.columns:
            df['Main_Numbers'] = df['Main_Numbers'].apply(
                lambda x: [int(num) for num in x.strip('[]').split(',')]
            )
        logger.info(f"Loaded {len(df)} lottery draws")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_features(df):
    """Prepare features for model training"""
    # Extract the target variables (the lottery numbers)
    y = np.array([row for row in df['Main_Numbers']])
    
    # Create simple features
    features = []
    
    # Add basic time series features
    for i in range(1, min(11, len(df))):
        col_name = f'prev_{i}'
        df[col_name] = df['Main_Numbers'].shift(i)
        features.append(col_name)
    
    # Add some statistical features
    for i in range(1, 6):
        # Moving averages of past draws
        df[f'ma_{i}'] = df['Main_Numbers'].rolling(i).apply(
            lambda x: np.mean([num for sublist in x for num in sublist])
        )
        features.append(f'ma_{i}')
        
        # Moving standard deviations
        df[f'std_{i}'] = df['Main_Numbers'].rolling(i).apply(
            lambda x: np.std([num for sublist in x for num in sublist])
        )
        features.append(f'std_{i}')
    
    # Drop rows with NaN values (the first few rows due to lagging/rolling)
    df = df.dropna()
    
    # Create feature matrix X
    X = np.zeros((len(df), len(features), 6))
    
    # Fill in the feature matrix
    for i, feature in enumerate(features):
        for j in range(len(df)):
            if isinstance(df[feature].iloc[j], list):
                # Fill in each number position with the previous draw's number in the same position
                for k in range(min(6, len(df[feature].iloc[j]))):
                    X[j, i, k] = df[feature].iloc[j][k]
            else:
                # Fill in the same statistical value for all positions
                X[j, i, :] = df[feature].iloc[j]
    
    # Reshape the feature matrix to 2D
    X = X.reshape(len(df), -1)
    
    # Get the target values after dropping rows
    y = np.array([row for row in df['Main_Numbers']])
    
    logger.info(f"Prepared features: X shape {X.shape}, y shape {y.shape}")
    return X, y

def train_lightgbm_model(X, y, n_trials=10):
    """Train a LightGBM model for lottery prediction"""
    logger.info("Training LightGBM model...")
    
    # Initialize scaler and scale input data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a model for each number position
    models = []
    for i in range(6):
        logger.info(f"Training model for number position {i+1}/6")
        
        # Default parameters for quick training
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': -1
        }
        
        # Train the model
        model = lgb.LGBMRegressor(**params, random_state=42)
        model.fit(X_scaled, y[:, i])
        models.append(model)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-5:]  # Top 5 features
            logger.info(f"Top features for model {i+1}: {top_indices}")
    
    return models, scaler

def predict_lightgbm_model(models, scaler, X):
    """Generate predictions using trained LightGBM models"""
    # Scale input
    X_scaled = scaler.transform(X)
    
    # Generate predictions
    predictions = np.zeros((X.shape[0], 6))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_scaled)
    
    # Validate predictions
    valid_predictions = []
    for i in range(predictions.shape[0]):
        valid_predictions.append(ensure_valid_prediction(predictions[i]))
    
    return np.array(valid_predictions)

def save_model(models, scaler, path=MODELS_PATH):
    """Save the trained model to disk"""
    logger.info(f"Saving model to {path}")
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'models': models, 'scaler': scaler}, f)

def load_model(path=MODELS_PATH):
    """Load a trained model from disk"""
    logger.info(f"Loading model from {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['models'], data['scaler']

def main():
    """Main function to train and test the lottery prediction model"""
    # Load data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Check if model exists
    if MODELS_PATH.exists():
        # Load existing model
        logger.info("Loading existing model...")
        models, scaler = load_model()
    else:
        # Train a new model
        logger.info("Training new model...")
        models, scaler = train_lightgbm_model(X, y)
        save_model(models, scaler)
    
    # Make a prediction for the next draw
    logger.info("Generating prediction for next draw...")
    prediction = predict_lightgbm_model(models, scaler, X[-1:])
    
    print("\n" + "="*50)
    print("LOTTERY PREDICTION FOR NEXT DRAW")
    print("="*50)
    print(f"Predicted numbers: {prediction[0]}")
    print("="*50 + "\n")
    
    return prediction[0]

if __name__ == "__main__":
    main() 