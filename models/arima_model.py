import numpy as np
from pmdarima import auto_arima

def train_arima_model(X_train, y_train):
    models = []
    for i in range(6):
        series = y_train[:, i]
        model = auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, d=1,
                           seasonal=False, trace=False, error_action='warn',
                           suppress_warnings=True, stepwise=True)
        models.append(model)
    return models

def predict_arima_model(models, X):
    predictions = np.zeros((X.shape[0], 6))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(n_periods=X.shape[0])
    return predictions