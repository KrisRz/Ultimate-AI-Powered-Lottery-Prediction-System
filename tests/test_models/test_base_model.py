import unittest
import pandas as pd
import numpy as np
from models.base import BaseModel, TimeSeriesModel, EnsembleModel, MetaModel

class MockBaseModel(BaseModel):
    """Mock implementation of BaseModel for testing"""
    def train(self, df: pd.DataFrame) -> None:
        self.is_trained = True
        self.model = "trained_model"
    
    def predict(self, df: pd.DataFrame) -> list:
        return [1, 2, 3, 4, 5, 6]

class MockTimeSeriesModel(TimeSeriesModel):
    """Mock implementation of TimeSeriesModel for testing"""
    def train(self, df: pd.DataFrame) -> None:
        self.is_trained = True
        self.model = "trained_ts_model"
    
    def predict(self, df: pd.DataFrame) -> list:
        return [10, 20, 30, 40, 50, 60]

class MockEnsembleModel(EnsembleModel):
    """Mock implementation of EnsembleModel for testing"""
    def combine_predictions(self, predictions):
        # Simple implementation that takes the first prediction
        return predictions[0] if predictions else [1, 2, 3, 4, 5, 6]

class MockMetaModel(MetaModel):
    """Mock implementation of MetaModel for testing"""
    def train_meta_model(self, df: pd.DataFrame) -> None:
        self.meta_model = "trained_meta_model"
    
    def combine_predictions(self, predictions):
        # Simple implementation that takes the first prediction
        return predictions[0] if predictions else [1, 2, 3, 4, 5, 6]

class TestBaseModel(unittest.TestCase):
    def setUp(self):
        # Create synthetic test data
        n_samples = 100
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        self.df = pd.DataFrame({
            'Draw Date': dates,
            'Feature1': np.random.rand(n_samples),
            'Feature2': np.random.rand(n_samples),
            'Feature3': np.random.rand(n_samples)
        })
        
        # Create test models
        self.base_model = MockBaseModel("base_test")
        self.ts_model = MockTimeSeriesModel("ts_test", look_back=5)
        self.model1 = MockBaseModel("model1")
        self.model2 = MockBaseModel("model2")
        self.ensemble_model = MockEnsembleModel("ensemble_test", 
                                                {"model1": self.model1, "model2": self.model2})
        self.meta_model = MockMetaModel("meta_test", 
                                        {"model1": self.model1, "model2": self.model2})
    
    def test_base_model_initialization(self):
        self.assertEqual(self.base_model.name, "base_test")
        self.assertFalse(self.base_model.is_trained)
        self.assertIsNone(self.base_model.model)
    
    def test_base_model_train_predict(self):
        self.base_model.train(self.df)
        self.assertTrue(self.base_model.is_trained)
        self.assertEqual(self.base_model.model, "trained_model")
        
        prediction = self.base_model.predict(self.df)
        self.assertEqual(prediction, [1, 2, 3, 4, 5, 6])
    
    def test_time_series_model(self):
        self.assertEqual(self.ts_model.look_back, 5)
        
        # Test prepare_sequence
        data = np.array([i for i in range(20)])
        X, y = self.ts_model.prepare_sequence(data)
        self.assertEqual(X.shape[0], 9)  # 20 - 5 - 6 = 9 samples
        self.assertEqual(X.shape[1], 5)  # look_back = 5
        self.assertEqual(y.shape[0], 9)  # 9 samples
        self.assertEqual(y.shape[1], 6)  # 6 targets
        
        # Check first sequence
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(y[0], [5, 6, 7, 8, 9, 10])
    
    def test_ensemble_model(self):
        # Test initialization
        self.assertEqual(len(self.ensemble_model.models), 2)
        self.assertIn("model1", self.ensemble_model.models)
        self.assertIn("model2", self.ensemble_model.models)
        
        # Test train method trains all models
        self.ensemble_model.train(self.df)
        self.assertTrue(all(model.is_trained for model in self.ensemble_model.models.values()))
        
        # Test predict method combines predictions
        prediction = self.ensemble_model.predict(self.df)
        self.assertEqual(prediction, [1, 2, 3, 4, 5, 6])  # Should match model1's prediction
    
    def test_meta_model(self):
        # Test meta_model training
        self.meta_model.train(self.df)
        self.assertEqual(self.meta_model.meta_model, "trained_meta_model")
        
        # Test that base models are also trained
        self.assertTrue(all(model.is_trained for model in self.meta_model.models.values()))
        
        # Test prediction using meta model
        prediction = self.meta_model.predict(self.df)
        self.assertEqual(prediction, [1, 2, 3, 4, 5, 6])

if __name__ == '__main__':
    unittest.main() 