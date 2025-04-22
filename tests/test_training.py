import pandas as pd
import numpy as np

# Create synthetic data
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
days = np.array([d.day for d in dates])
months = np.array([d.month for d in dates])
years = np.array([d.year for d in dates]) 