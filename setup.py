from setuptools import setup, find_packages

setup(
    name="lottery_prediction",
    version="0.2.0",
    description="Advanced Lottery Number Prediction System",
    author="Lottery Prediction Team",
    author_email="info@lotteryprediction.com",
    packages=find_packages(),
    install_requires=[
        # Core packages
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        
        # Model libraries
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "keras>=2.10.0",
        
        # Optimization
        "optuna>=3.0.0",
        "hyperopt>=0.2.7",
        
        # Utility packages
        "tqdm>=4.62.0",
        "psutil>=5.9.0",  # For memory monitoring
        "gputil>=1.4.0",  # For GPU monitoring (optional)
        "joblib>=1.1.0",
        
        # Visualization
        "seaborn>=0.11.0", 
        "plotly>=5.3.0",
        
        # Data processing
        "python-dateutil>=2.8.2",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "gpu": [
            "tensorflow-metal>=1.0.0",  # For macOS Metal GPU acceleration
            "torch>=2.0.0",  # Optional PyTorch support
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "lottery-predict=scripts.main:main",
            "lottery-train=scripts.train_models:main",
            "lottery-data=scripts.fetch_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 