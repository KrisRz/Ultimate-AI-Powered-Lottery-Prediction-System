import logging
import numpy as np
from pathlib import Path
import pickle
import os
import pandas as pd
from typing import Dict, List, Tuple

from .fetch_data import load_data
from .data_validation import DataValidator
from .predict_next_draw import generate_next_draw_predictions, format_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lottery_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODELS_PATH = "models/trained_models.pkl"

def load_trained_models():
    """Load pre-trained models from file"""
    if not os.path.exists(MODELS_PATH):
        logger.error("No pre-trained models found. Please train models first.")
        return None
    
    logger.info("Loading pre-trained models...")
    with open(MODELS_PATH, 'rb') as f:
        return pickle.load(f)

def predict_next_draw(
    data_path: str = 'data/lottery_data_1995_2025.csv',
    n_predictions: int = 10
) -> Dict:
    """Generate predictions using pre-trained models"""
    # Load models
    models = load_trained_models()
    if models is None:
        return {'error': 'No pre-trained models found'}
    
    # Load and validate data
    data = load_data(data_path)
    validator = DataValidator()
    validation_results = validator.validate_data(data)
    
    if validation_results['errors']:
        logging.error(f"Data validation failed: {validation_results['errors']}")
        return {'error': 'Data validation failed'}
    
    # Generate predictions
    next_draw_predictions = generate_next_draw_predictions(
        models=models,
        last_data=data.iloc[-1:],
        n_predictions=n_predictions,
        min_number=1,
        max_number=59,
        n_numbers=6
    )
    
    # Format predictions for display
    predictions_text = format_predictions(next_draw_predictions)
    logging.info("\n" + predictions_text)
    
    return {
        'next_draw_predictions': next_draw_predictions,
        'validation_results': validation_results
    }

if __name__ == "__main__":
    results = predict_next_draw()
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("\nNext Draw Predictions:")
        for i, pred in enumerate(results['next_draw_predictions'], 1):
            print(f"Prediction {i}: {', '.join(map(str, pred))}") 