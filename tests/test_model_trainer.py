import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.model_trainer import ModelTrainer
from src.data_preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class TestModelTrainer(unittest.TestCase):
    """Test suite for ModelTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 13)
        self.X_test = np.random.rand(30, 13)
        self.y_train = np.random.randint(0, 2, 100)
        self.y_test = np.random.randint(0, 2, 30)
        self.trainer = ModelTrainer()

    def test_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.models, {})

    def test_train_logistic_regression(self):
        """Test Logistic Regression training."""
        model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, LogisticRegression)

    def test_train_random_forest(self):
        """Test Random Forest training."""
        model = self.trainer.train_random_forest(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_train_xgboost(self):
        """Test XGBoost training."""
        model = self.trainer.train_xgboost(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        # XGBoost returns a Booster object
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_model_probability_predictions(self):
        """Test probability predictions."""
        model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        probabilities = model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape, (len(self.y_test), 2))
        # Check that probabilities sum to 1
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(len(self.y_test)))

    def test_ensemble_training(self):
        """Test ensemble model training with multiple algorithms."""
        models = {
            'logistic_regression': self.trainer.train_logistic_regression(self.X_train, self.y_train),
            'random_forest': self.trainer.train_random_forest(self.X_train, self.y_train),
        }
        self.assertEqual(len(models), 2)
        for model in models.values():
            self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
