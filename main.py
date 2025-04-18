"""
Main execution script for lottery prediction system.
"""
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

from scripts.predict_numbers import (
    predict_next_draw,
    backtest,
    rolling_window_cv,
    test_randomness,
    load_models
)
from scripts.train_models import update_models
from scripts.analyze_data import load_data, analyze_lottery_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lottery_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DATA_FILE = Path("data/lottery_data_1995_2025.csv")
RESULTS_DIR = Path("results")

def run_predictions(force_retrain: bool = False) -> None:
    """Run the complete prediction pipeline."""
    try:
        logger.info("Starting prediction pipeline...")
        
        # Create results directory
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Load data
        logger.info("Loading data...")
        df = load_data(DATA_FILE)
        logger.info(f"Loaded {len(df)} draws from {df['Date'].min()} to {df['Date'].max()}")
        
        # Update models if necessary
        logger.info("Updating models...")
        update_models(force_retrain)
        
        # Load models
        logger.info("Loading models...")
        models = load_models()
        
        # Test randomness
        logger.info("Testing for randomness...")
        randomness_results = test_randomness(df)
        logger.info("Randomness test results:")
        for test, pvalue in randomness_results.items():
            logger.info(f"{test}: {pvalue:.4f}")
        
        # Perform rolling window cross-validation
        logger.info("Running cross-validation...")
        cv_results = rolling_window_cv(df)
        logger.info("Cross-validation results:")
        for metric, values in cv_results.items():
            logger.info(f"{metric}: mean={pd.np.mean(values):.4f}, std={pd.np.std(values):.4f}")
        
        # Run backtesting
        logger.info("Running backtesting...")
        backtest_results = backtest(df)
        logger.info("Backtesting results:")
        for metric, value in backtest_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Generate predictions for next draw
        logger.info("Generating predictions for next draw...")
        predictions = predict_next_draw(models, df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"predictions_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("Lottery Number Predictions\n")
            f.write("=========================\n\n")
            f.write(f"Generated on: {datetime.now()}\n")
            f.write(f"Data range: {df['Date'].min()} to {df['Date'].max()}\n\n")
            
            f.write("Predicted Combinations:\n")
            f.write("----------------------\n")
            for i, combination in enumerate(predictions[:10], 1):
                f.write(f"Combination {i}: {combination}\n")
            
            f.write("\nModel Performance:\n")
            f.write("----------------\n")
            f.write("Backtesting Results:\n")
            for metric, value in backtest_results.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nCross-validation Results:\n")
            for metric, values in cv_results.items():
                f.write(f"{metric}: mean={pd.np.mean(values):.4f}, std={pd.np.std(values):.4f}\n")
            
            f.write("\nRandomness Tests:\n")
            for test, pvalue in randomness_results.items():
                f.write(f"{test}: {pvalue:.4f}\n")
        
        logger.info(f"Results saved to {results_file}")
        logger.info("Prediction pipeline completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise

if __name__ == "__main__":
    run_predictions() 