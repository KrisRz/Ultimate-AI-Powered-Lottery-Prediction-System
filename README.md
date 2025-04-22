# Lottery Number Prediction System

A comprehensive machine learning system for analyzing lottery data and predicting potential winning combinations.

## Features

- Multiple prediction models including LSTM, ARIMA, Holt-Winters, and various ML models
- Historical data analysis and pattern recognition
- Monte Carlo simulation for combination optimization
- Backtesting and cross-validation
- Randomness testing
- Comprehensive logging and result tracking

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your lottery data in `data/lottery_data_1995_2025.csv` with the format:
```
Draw Date,Day,Balls,Jackpot,Winners,Draw Details
DD MMM YYYY,Day,"N1 N2 N3 N4 N5 N6 BONUS N7",£Amount,Count,details
```

Example:
```
30 Dec 1995,Sat,"06 32 39 42 43 45 BONUS 36",£23,966,033,0,draw details
```

2. Run the prediction system:
```bash
python main.py
```

3. Results will be saved in the `results` directory with timestamp.

## Project Structure

```
.
├── data/                   # Data directory
│   └── lottery_data_*.csv # Lottery data files
├── models/                # Trained model files
├── results/               # Prediction results
├── scripts/              # Core functionality
│   ├── analyze_data.py   # Data analysis functions
│   ├── predict_numbers.py # Prediction functions
│   └── train_models.py   # Model training functions
├── main.py               # Main execution script
└── requirements.txt      # Python dependencies
```

## Model Types

- LSTM (Long Short-Term Memory)
- ARIMA (Autoregressive Integrated Moving Average)
- Holt-Winters Exponential Smoothing
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- K-Nearest Neighbors

## Performance Metrics

- Accuracy (exact matches)
- Partial matches
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Randomness tests (KS test, Runs test, Chi-square test)

## License

MIT License
