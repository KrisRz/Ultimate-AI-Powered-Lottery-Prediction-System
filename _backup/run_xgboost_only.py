import logging
import sys
import time
import traceback
import numpy as np
import os
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lottery_xgboost.log')
    ]
)
logger = logging.getLogger(__name__)

# Progress bar function
def create_progress_bar(total, desc="Progress"):
    """Create a progress bar for tracking operations"""
    class SimpleProgressBar:
        def __init__(self, total, description):
            self.total = total
            self.current = 0
            self.description = description
            print(f"\n{self.description}: 0/{self.total}")
            
        def update(self, amount=1):
            self.current += amount
            print(f"\r{self.description}: {self.current}/{self.total}", end="")
            
        def close(self):
            print("\n")
    
    return SimpleProgressBar(total, desc)

def load_data(data_path):
    """Load lottery data from CSV file"""
    try:
        import pandas as pd
        print(f"Loading data from {data_path}...")
        logger.info(f"Loading data from {data_path}")
        
        # Read CSV data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} rows of data")
        
        # Check if we have Main_Numbers column
        if 'Main_Numbers' not in data.columns:
            # If we have Balls column, try to extract Main_Numbers
            if 'Balls' in data.columns:
                print("Extracting Main_Numbers from Balls column...")
                logger.info("Extracting Main_Numbers from Balls column")
                
                # Process rows
                main_numbers_list = []
                bonus_numbers = []
                valid_indices = []
                
                print(f"Sample Balls value: {data['Balls'].iloc[0]}")
                
                ball_data_pbar = create_progress_bar(len(data), "Processing ball data")
                for idx, ball_data in enumerate(data['Balls']):
                    try:
                        # Convert to string if not already
                        if not isinstance(ball_data, str):
                            ball_data = str(ball_data)
                            
                        # Skip NaN values
                        if ball_data.lower() == 'nan':
                            ball_data_pbar.update(1)
                            continue
                        
                        # Strip quotes if present
                        ball_data = ball_data.strip('"\'')
                        
                        if ' BONUS ' in ball_data:
                            # Standard format with BONUS
                            main_part, bonus_part = ball_data.split(' BONUS ')
                            main_nums = [int(num) for num in main_part.split()]
                            bonus = int(bonus_part.split(',')[0])  # Get the first part in case there's a comma
                        else:
                            # Try alternative format
                            parts = ball_data.split()
                            if len(parts) < 7:  # Need at least 6 main numbers + bonus
                                ball_data_pbar.update(1)
                                continue
                                
                            # Assume first 6 are main numbers, last is bonus
                            main_nums = [int(num) for num in parts[:6]]
                            bonus = int(parts[-1])
                            
                        # Validate we have exactly 6 main numbers
                        if len(main_nums) != 6:
                            ball_data_pbar.update(1)
                            continue
                            
                        # Sort main numbers
                        main_nums = sorted(main_nums)
                        
                        main_numbers_list.append(main_nums)
                        bonus_numbers.append(bonus)
                        valid_indices.append(idx)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing row {idx}: {str(e)}")
                        if idx < 5:  # Print first few errors for debugging
                            print(f"Error parsing row {idx}: {str(e)}, value: {ball_data}")
                    
                    ball_data_pbar.update(1)
                
                ball_data_pbar.close()
                
                # Keep only valid rows
                if valid_indices:
                    print(f"Keeping {len(valid_indices)} valid rows out of {len(data)}")
                    data = data.iloc[valid_indices].copy()
                    data['Main_Numbers'] = main_numbers_list
                    data['Bonus'] = bonus_numbers
                    data = data.reset_index(drop=True)
                    
                    if len(data) < 10:
                        raise ValueError(f"Not enough valid data rows: only {len(data)} valid entries found")
                else:
                    raise ValueError("No valid lottery data found")
            else:
                raise ValueError("No Main_Numbers or Balls column found in data")
        
        print(f"Final dataset has {len(data)} rows")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def prepare_features(data):
    """Prepare features for training"""
    try:
        print("Preparing features...")
        
        # Convert Main_Numbers from string to array if needed
        if isinstance(data['Main_Numbers'].iloc[0], str):
            data['Main_Numbers'] = data['Main_Numbers'].apply(
                lambda x: np.array(eval(x)) if isinstance(x, str) else x
            )
        
        # Basic features
        features = []
        labels = []
        
        # Number of rows to use for training
        n_rows = len(data)
        history_size = 5  # Number of previous draws to use as features
        
        print(f"Creating features from {n_rows} draws with history size {history_size}")
        logger.info(f"Creating features from {n_rows} draws with history size {history_size}")
        
        # Create progress bar
        pbar = create_progress_bar(n_rows - history_size, "Creating features")
        
        for i in range(history_size, n_rows):
            # Get the current and previous draws
            current_numbers = np.array(data['Main_Numbers'].iloc[i])
            
            # Features from previous draws
            feature_vector = []
            for j in range(1, history_size + 1):
                prev_numbers = np.array(data['Main_Numbers'].iloc[i-j])
                feature_vector.extend(prev_numbers)
                
                # Add simple derived features
                feature_vector.append(np.mean(prev_numbers))
                feature_vector.append(np.std(prev_numbers))
                feature_vector.append(np.max(prev_numbers) - np.min(prev_numbers))
            
            features.append(feature_vector)
            labels.append(current_numbers)
            pbar.update(1)
        
        pbar.close()
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Created feature matrix with shape {X.shape} and labels with shape {y.shape}")
        logger.info(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def train_xgboost_model(X_train, y_train, tune_hyperparams=True, n_trials=10):
    """Train XGBoost models for lottery prediction"""
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler
        import optuna
        
        print("Training XGBoost models...")
        logger.info("Starting XGBoost model training")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Train a separate model for each number position
        models = []
        
        # Default parameters if tuning fails
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        # Create progress bar
        pbar = create_progress_bar(6, "Training models")
        
        for i in range(6):
            print(f"\nTraining model for number position {i+1}/6")
            logger.info(f"Training XGBoost model for position {i+1}/6")
            
            # Hyperparameter tuning
            if tune_hyperparams:
                try:
                    print(f"Tuning hyperparameters for model {i+1}/6...")
                    
                    # Define objective function for Optuna
                    def objective(trial):
                        params = {
                            'objective': 'reg:squarederror',
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                            'max_depth': trial.suggest_int('max_depth', 3, 8),
                            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0),
                            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0),
                            'random_state': 42
                        }
                        
                        # Train with cross-validation
                        from sklearn.model_selection import cross_val_score
                        model = xgb.XGBRegressor(**params)
                        scores = cross_val_score(model, X_scaled, y_train[:, i], cv=3)
                        return scores.mean()
                    
                    # Create and run the study
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials)
                    
                    # Get best parameters
                    best_params = study.best_params
                    best_params['objective'] = 'reg:squarederror'
                    best_params['random_state'] = 42
                    
                    print(f"Best parameters: {best_params}")
                    logger.info(f"Best parameters for model {i+1}: {best_params}")
                    
                except Exception as e:
                    print(f"Error during tuning: {str(e)}, using default parameters")
                    logger.error(f"Tuning error: {str(e)}")
                    best_params = default_params
            else:
                best_params = default_params
            
            # Train the model
            try:
                model = xgb.XGBRegressor(**best_params)
                model.fit(X_scaled, y_train[:, i])
                models.append(model)
                print(f"Successfully trained model {i+1}/6")
                logger.info(f"Successfully trained model {i+1}/6")
                
                # Log feature importance
                if hasattr(model, 'feature_importances_'):
                    top_features = np.argsort(model.feature_importances_)[-5:]
                    logger.info(f"Top features for model {i+1}: {top_features}")
                
            except Exception as e:
                logger.error(f"Error training model {i+1}: {str(e)}")
                print(f"Error training model: {str(e)}")
                
                # Fallback to a simpler model
                try:
                    print("Trying fallback with simpler parameters...")
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        max_depth=3,
                        learning_rate=0.1,
                        n_estimators=50,
                        random_state=42
                    )
                    model.fit(X_scaled, y_train[:, i])
                    models.append(model)
                except Exception as e2:
                    logger.error(f"Fallback training also failed: {str(e2)}")
                    
                    # Create a dummy model that returns the mean
                    mean_value = np.mean(y_train[:, i])
                    class DummyModel:
                        def predict(self, X):
                            return np.full(X.shape[0], mean_value)
                    
                    models.append(DummyModel())
                    print(f"Using dummy model for position {i+1}")
            
            pbar.update(1)
        
        pbar.close()
        return models, scaler
        
    except Exception as e:
        logger.error(f"Error in XGBoost training: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def ensure_valid_prediction(pred_numbers):
    """Ensure predictions are valid lottery numbers"""
    try:
        # Round to nearest integer
        rounded = np.round(pred_numbers).astype(int)
        
        # Ensure numbers are within valid range (1-49 typically)
        valid_range = np.clip(rounded, 1, 49)
        
        # Ensure no duplicates
        unique_numbers = []
        for num in valid_range:
            if num not in unique_numbers:
                unique_numbers.append(num)
        
        # If we don't have enough unique numbers, add more
        while len(unique_numbers) < 6:
            new_num = np.random.randint(1, 50)
            if new_num not in unique_numbers:
                unique_numbers.append(new_num)
        
        # Return sorted numbers
        return np.array(sorted(unique_numbers[:6]))
        
    except Exception as e:
        logger.error(f"Error ensuring valid prediction: {str(e)}")
        # Return random valid numbers as fallback
        return np.array(sorted(np.random.choice(range(1, 50), 6, replace=False)))

def predict_next_draw(models, scaler, X_test, n_predictions=5):
    """Generate predictions for the next draw"""
    try:
        print(f"Generating {n_predictions} predictions...")
        
        # Ensure X_test is a 2D array
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)
        
        # Scale the input
        X_scaled = scaler.transform(X_test)
        
        # Make predictions
        all_predictions = []
        
        for _ in range(n_predictions):
            # Add some randomness for diverse predictions
            noise = np.random.normal(0, 0.1, X_scaled.shape)
            X_noisy = X_scaled + noise
            
            # Generate prediction
            prediction = np.zeros(6)
            for i, model in enumerate(models):
                prediction[i] = model.predict(X_noisy)[0]
            
            # Ensure valid prediction
            valid_prediction = ensure_valid_prediction(prediction)
            all_predictions.append(valid_prediction)
        
        return all_predictions
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Return random predictions as fallback
        return [np.array(sorted(np.random.choice(range(1, 50), 6, replace=False))) 
                for _ in range(n_predictions)]

def main():
    """Main function to run the lottery prediction system"""
    try:
        start_time = time.time()
        print("Starting XGBoost-only Lottery Prediction System")
        logger.info("Starting XGBoost-only Lottery Prediction System")
        
        # 1. Load data
        data_path = "data/fixed_lottery_data.csv"  # Using the fixed data file
        data = load_data(data_path)
        
        # 2. Prepare features
        X, y = prepare_features(data)
        
        # 3. Train models
        models, scaler = train_xgboost_model(X, y, tune_hyperparams=True, n_trials=10)
        
        # 4. Generate predictions for the next draw
        # Use the most recent data point as test input
        X_test = X[-1:] 
        predictions = predict_next_draw(models, scaler, X_test, n_predictions=10)
        
        # 5. Display predictions
        print("\nPredictions for the next draw:")
        for i, pred in enumerate(predictions):
            print(f"Prediction {i+1}: {pred}")
        
        # Save predictions to file
        try:
            import json
            with open("results/xgboost_predictions.json", "w") as f:
                json.dump({f"prediction_{i+1}": pred.tolist() for i, pred in enumerate(predictions)}, 
                         f, indent=2)
            print("\nPredictions saved to results/xgboost_predictions.json")
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
        
        duration = time.time() - start_time
        print(f"\nCompleted in {duration:.2f} seconds")
        logger.info(f"Prediction process completed in {duration:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"\nError: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 