This project is for demonstration purposes only. For code access or questions, please contact me.

# AI-Powered Lottery Prediction System

## 🚀 Overview

An advanced machine learning system designed to analyze historical lottery data and generate predictions using ensemble learning techniques. This project utilizes multiple AI models including LSTM neural networks, CNN-LSTM hybrid models, and custom feature engineering to identify patterns in lottery drawings.

## ✨ Key Features

- **Multi-Model Ensemble**: Combines predictions from LSTM and CNN-LSTM models with optimized weights
- **Interactive Progress Tracking**: Real-time progress bars during data download, model training, and prediction
- **Advanced Cross-Validation**: Time series validation with rolling windows for robust model evaluation
- **Enhanced Feature Engineering**: Extracts complex patterns from historical lottery data
- **Prediction Diversity**: Generates varied predictions through controlled perturbation techniques
- **Performance Monitoring**: Detects model drift and tracks prediction quality over time
- **Visualization Tools**: Creates intuitive graphics of model performance and predictions
- **Comprehensive Logging**: All operations are logged to `logs/lottery.log` for traceability

## 📋 System Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- CUDA-compatible GPU (optional, for faster training)
- 1GB+ free disk space

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd lottery-prediction-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Get Started

To get up and running quickly:

1. Create the local environment (predict-only stack)

```bash
conda env create -f environment.yml
conda activate lotto-predict
```

2. Run predictions (10 lines, optimized coverage):

```bash
./predict_tonight.sh
```

The system will automatically:
- Load cached lottery data (or download if not available)
- Load pre-trained models (or train if not available)
- Generate and display predictions with visualizations
- Save results to the outputs directory

## 📊 Usage

The system offers several command-line options to customize its behavior:

```bash
# One-liners
make predict      # run predictions
make backtest     # quick backtest with baselines
make nightly      # nightly backtest runner
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--retrain {yes,no}` | Retrain models from scratch (yes) or use existing trained models (no) |
| `--force` | Force download of fresh lottery data from the web |
| `--count COUNT` | Number of predictions to generate (default: 10) |
| `--diversity DIVERSITY` | Diversity level for predictions (0-1, default: 0.5) |
| `--sequence-length LENGTH` | Sequence length for model training (default: 30) |
| `--seed SEED` | Random seed for reproducibility |

## 🧠 How It Works

### 1. Data Processing

- Historical lottery data is automatically downloaded and processed
- Advanced feature engineering extracts patterns and statistical properties
- Time series sequences are created for deep learning models
- Data is normalized and split into training/validation/test sets

### 2. Model Training

The system trains multiple models:

- **LSTM Neural Network**: Learns temporal patterns in the sequence of draws
- **CNN-LSTM**: Hybrid model combining convolutional and recurrent layers for feature extraction
- **Ensemble Model**: Combines predictions from both models with optimized weights based on validation performance

### 3. Validation & Monitoring

- Time series validation with proper train/validation splits
- Custom lottery-specific metrics (exact matches, partial matches)
- Performance monitoring with detailed metrics tracking
- Model interpretation with feature importance analysis

### 4. Prediction Generation

- Ensembles multiple model predictions with optimized weights
- Uses perturbation techniques to generate diverse predictions
- Visualizes predictions for better understanding
- Saves predictions in both JSON and human-readable formats

## 📁 Project Structure

```
lottery-prediction-system/
├── data/                    # Lottery data files
├── logs/                    # Training and operation logs
├── models/                  # Model implementations
│   └── checkpoints/         # Saved model weights and ensembles
├── outputs/                 # Generated outputs
│   ├── predictions/         # Prediction results (JSON and text)
│   ├── training/            # Training checkpoints and metadata
│   ├── validation/          # Validation results and visualizations
│   ├── monitoring/          # Performance monitoring data
│   ├── interpretations/     # Feature importance data
│   └── visualizations/      # Generated plots and charts
├── scripts/                 # Core implementation
│   ├── analyze_data.py      # Data analysis utilities
│   ├── feature_engineering/ # Feature engineering modules
│   ├── fetch_data.py        # Data loading and preprocessing
│   ├── improved_training.py # Enhanced model training
│   ├── main.py              # Main execution script
│   ├── model_bridge.py      # Interface between scripts and models
│   ├── new_predict.py       # Prediction generation
│   ├── performance_tracking.py # Performance monitoring
│   ├── train_models.py      # Model training implementations
│   ├── utils/               # Utility functions
│   └── validations/         # Validation modules
└── README.md                # This file
```

## 🔮 Results

When you run the system, it will:

1. **Load or Train Models**: Either load pre-trained models or train new ones
2. **Validate Models**: Evaluate performance on validation data
3. **Generate Predictions**: Create diverse sets of lottery number predictions
4. **Visualize Results**: Create visual representations of the predictions
5. **Save Outputs**: Store all results in the outputs directory

The predictions are:
- Displayed in the terminal with detailed output
- Saved as JSON at `outputs/predictions/predictions_YYYY-MM-DD.json`
- Saved as readable text at `outputs/predictions/formatted_predictions_YYYY-MM-DD.txt`
- Visualized and saved at `outputs/visualizations/prediction_viz_YYYYMMDD_HHMMSS.png`

## 📊 Output Directory Structure

The system maintains an organized output structure:

- **predictions/**: Contains JSON and text format predictions
- **training/**: Stores model checkpoints (.h5 files) and training metadata
- **validation/**: Holds validation metrics and comparison visualizations
- **monitoring/**: Contains performance tracking over time
- **interpretations/**: Stores feature importance data
- **visualizations/**: Contains prediction visualizations and plots

## ⚠️ Disclaimer

This software is provided for educational and entertainment purposes only. Lottery games are based on random chance, and while statistical analysis and machine learning can identify patterns in historical data, there is no guarantee of predicting future results.

## 📄 License

This project is licensed under the MIT License.
