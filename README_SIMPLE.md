# Simplified Lottery Prediction System

This is a simplified version of the full lottery prediction system, using only XGBoost to generate predictions.

## Features

- Data loading from CSV with proper parsing of lottery numbers
- XGBoost model training for number prediction
- Randomized predictions for better diversity
- Visualization of number frequencies and predictions
- Easy to run with minimal dependencies

## How to Run

1. Make sure you have the required dependencies installed:

```bash
pip install numpy pandas xgboost matplotlib
```

2. Run the simplified script:

```bash
python run_simple.py
```

This will:
- Load the lottery data from `data/fixed_lottery_data.csv`
- Train an XGBoost model
- Generate 5 lottery number predictions
- Create a visualization of number frequencies and predictions
- Save the visualization to `results/number_frequencies.png`

## Understanding the Output

The script will print the 5 predictions to the console, and it will also generate a visualization with two charts:

1. **Number Frequencies Chart**: Shows how often each number 1-59 appears in the historical data
   - Blue bars: Regular numbers
   - Orange bars: Numbers that appear in the predictions
   - Red line: Average frequency

2. **Prediction Heatmap**: Shows the predictions in a heatmap format
   - Each row corresponds to one prediction
   - Numbers are highlighted and labeled

## Customizing

You can easily modify the script to:

- Change the number of predictions by changing the `n_predictions` parameter
- Adjust the randomness factor by modifying the standard deviation in the `noise` variable
- Train with different XGBoost parameters
- Use a different dataset by changing the `data_path`

## Troubleshooting

If you encounter any issues:

1. Make sure you have the correct data file in the `data` directory
2. Check that your Python environment has all the required dependencies
3. Make sure the `results` directory exists or that you have write permissions

## License

This project is licensed under the same terms as the main lottery prediction system. 