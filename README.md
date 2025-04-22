# Advanced Lottery Prediction System

A sophisticated machine learning system for predicting lottery numbers using various predictive models and ensemble techniques.

## 🎯 Overview

This project uses a combination of machine learning models, statistical analysis, and ensemble techniques to predict potential lottery numbers. The system leverages:

- Classical machine learning (XGBoost, LightGBM, CatBoost, etc.)
- Deep learning (LSTM, CNN-LSTM, Autoencoders)
- Statistical models (ARIMA, Holt-Winters)
- Pattern analysis and frequency mining
- Ensemble techniques to combine predictions

**Important Disclaimer**: While this system uses advanced modeling techniques, lottery draws are fundamentally random events. No prediction system can guarantee winning numbers. This project is for educational and entertainment purposes only.

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lottery_prediction.git
   cd lottery_prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Quick Start Demo

Run the demo script to see the system in action:

```bash
python demo.py
```

This will:
1. Load sample data or generate it if none is available
2. Train models or load pre-trained models
3. Generate predictions using various models
4. Create ensemble predictions
5. Save results to `results/demo_predictions.json`

### Using Individual Components

You can use specific components of the system in your own code:

```python
from models.compatibility import (
    predict_next_draw,
    ensemble_prediction,
    monte_carlo_simulation
)
import pandas as pd

# Load your lottery data
df = pd.read_csv('path/to/lottery_data.csv')

# Train or load models
models = load_trained_models()  # Or train your own

# Generate predictions
predictions = predict_next_draw(models, df, n_predictions=10)
```

### Training Models

Train all models on your data:

```python
from scripts.train_models import train_all_models
import pandas as pd

# Load your data
df = pd.read_csv('path/to/lottery_data.csv')

# Train all models (will save to models/trained_models.pkl)
models = train_all_models(df, force_retrain=True)
```

## 🧠 Model Architecture

The system includes the following models:

1. **LSTM Model**: Long Short-Term Memory neural network for sequence prediction
2. **CNN-LSTM Model**: Convolutional Neural Network with LSTM layers
3. **Autoencoder Model**: Deep learning model for pattern detection
4. **XGBoost/LightGBM/CatBoost Models**: Gradient boosting frameworks
5. **Linear Models**: Basic regression models
6. **Statistical Models**: ARIMA, Holt-Winters for time series forecasting
7. **Meta-Model**: Ensemble learning to combine predictions from all models

## 📁 Project Structure

```
lottery_prediction/
├── data/                    # Directory for lottery data
├── models/                  # Model implementations
│   ├── lstm_model.py        # LSTM implementation
│   ├── xgboost_model.py     # XGBoost implementation
│   ├── ...                  # Other models
│   └── meta_model.py        # Meta-model for ensemble learning
├── scripts/                 # Utility scripts
│   ├── fetch_data.py        # Data loading utilities
│   ├── train_models.py      # Model training script
│   └── predict_numbers.py   # Prediction utilities
├── tests/                   # Test files
├── results/                 # Directory for output files
├── demo.py                  # Demo script
└── README.md                # This file
```

## 🔍 How It Works

1. **Data Preparation**: Historical lottery draws are loaded and transformed into features
2. **Feature Engineering**: Various features are extracted (frequency patterns, statistical properties)
3. **Model Training**: Each model is trained on the prepared data
4. **Prediction Generation**: Models generate individual predictions
5. **Ensemble Learning**: Individual predictions are combined using weighted voting
6. **Monte Carlo Simulation**: Additional predictions using statistical sampling

## 📈 Performance Tracking

The system tracks the performance of each model over time and uses this information to adjust the weights in the ensemble. Better performing models receive higher weights in the final prediction.

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
python -m pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🚀 Optimized Model Training

The system includes optimized model training capabilities for better performance and faster training times:

```bash
# Run optimized training
./optimize_training.py --workers 4
```

### Optimization Features

1. **Parallel Model Training**: Trains multiple models simultaneously using multiple CPU cores
2. **Model Caching**: Caches trained models to avoid retraining when data hasn't changed
3. **Memory Profiling**: Tracks memory usage during training to help identify bottlenecks
4. **Early Stopping**: Automatically stops training when performance stops improving
5. **Data Preparation Optimization**: Prepares data once for all models rather than repeatedly
6. **Fallbacks**: Automatically falls back to simpler models if advanced models fail to train

### Performance Comparison

| Feature | Standard Training | Optimized Training |
|---------|------------------|-------------------|
| Training Time | Baseline | 2-3x faster |
| Memory Usage | High | Optimized |
| Multiple Models | Sequential | Parallel |
| Caching | None | Yes (7-day expiry) |
| Validation | Limited | Comprehensive |

### Advanced Usage

For more control over the optimization process:

```python
from models.optimized_training import train_all_models_parallel

# Custom number of worker processes
models = train_all_models_parallel(df, max_workers=6)

# Load optimized models
from models.optimized_training import load_optimized_models
models = load_optimized_models()
```

## Optimized LSTM Model for Lottery Prediction

### Optimizations Overview

The LSTM model implementation has been optimized with the following enhancements:

1. **Advanced Model Architecture**
   - Bidirectional LSTM layers for better pattern recognition
   - Additional dense layer for improved feature abstraction
   - L2 regularization for weights to prevent overfitting
   - Batch normalization for better training stability
   - Higher capacity (more units) to handle complex patterns

2. **GPU Acceleration**
   - Mixed precision training support for faster computation
   - Memory growth configuration for efficient GPU utilization
   - TensorFlow optimizations for parallel processing

3. **Enhanced Data Preparation**
   - Vectorized sequence creation for faster preprocessing
   - RobustScaler for better handling of outliers
   - Optional PCA for dimensionality reduction
   - Advanced feature engineering with rolling statistics and time-series features

4. **Training Improvements**
   - Adaptive learning rate with ReduceLROnPlateau
   - Increased batch size for better parallelization
   - Model checkpointing to save best models
   - Early stopping with longer patience for large datasets
   - Configurable hyperparameters via centralized configuration

5. **CNN-LSTM Hybrid Model**
   - Convolutional layers for spatial pattern extraction
   - Deeper network with multiple CNN layers
   - Regularization and pooling for better generalization

6. **Memory and Performance Optimizations**
   - Garbage collection to prevent memory leaks
   - GPU memory clearing between training sessions
   - Efficient sequence generation algorithms
   - Caching of computed features for faster retraining

### Usage

To train the models with optimized configuration:

```bash
python scripts/train_models.py --data data/lottery_data_1995_2025.csv --config default
```

Available configurations:
- `default`: Balanced configuration for good performance and training time
- `quick`: Faster training with smaller model and fewer epochs
- `deep`: Extensive training with larger model for potentially better results

### Model Parameters

Key parameters for the optimized LSTM model:

```python
LSTM_CONFIG = {
    "look_back": 200,                 # Number of previous draws to use as context
    "lstm_units_1": 256,              # Units in first LSTM layer
    "lstm_units_2": 128,              # Units in second LSTM layer
    "dropout_rate": 0.3,              # Dropout rate for regularization
    "l2_reg": 0.001,                  # L2 regularization factor
    "batch_size": 64,                 # Batch size for training
    "epochs": 300,                    # Maximum number of training epochs
    "learning_rate": 0.001,           # Initial learning rate
}
```

## Project Structure

```
/
├── data/                        # Data directory
│   └── lottery_data_1995_2025.csv  # Historical lottery data
├── logs/                        # Training and operation logs
├── models/                      # Model implementations
│   ├── lstm_model.py            # Optimized LSTM implementation
│   ├── cnn_lstm_model.py        # Optimized CNN-LSTM implementation
│   ├── feature_engineering.py   # Advanced feature engineering
│   ├── training_config.py       # Centralized configuration
│   └── checkpoints/             # Saved model checkpoints
├── results/                     # Prediction results
├── scripts/                     # Main scripts
│   ├── train_models.py          # Model training script
│   ├── fetch_data.py            # Data loading and preparation
│   └── predict_numbers.py       # Prediction generation
└── README.md                    # This file
```

## Dependencies

- **Python 3.11+**
- **Core Libraries**: `numpy`, `pandas`, `scikit-learn`, `scipy`
- **Deep Learning**: `tensorflow` 2.10+
- **Time-Series**: `tsfresh` for automated feature extraction
- **Boosting Models**: `xgboost`, `lightgbm`, `catboost`
- **Performance**: `psutil` for memory monitoring

## Performance Considerations

- The optimized LSTM model requires significantly more memory than the original implementation
- Training on GPU is highly recommended for large datasets
- Feature extraction with `tsfresh` can be computationally intensive
- Consider using the `quick` configuration for initial testing

## Examples

```python
# Import necessary modules
from scripts.fetch_data import load_data
from scripts.train_models import train_all_models
from scripts.predict_numbers import predict_next_draw

# Load data
df = load_data("data/lottery_data_1995_2025.csv")

# Train models (can take significant time)
models = train_all_models(df, force_retrain=True)

# Generate predictions
predictions = predict_next_draw(models, df, n_predictions=10)
print(f"Top 10 predicted combinations: {predictions}")
```
