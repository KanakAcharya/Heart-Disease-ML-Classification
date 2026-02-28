"""Unit tests for data preprocessing module.

This module tests the DataPreprocessor class to ensure data handling,
imputation, splitting, and scaling work correctly.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(25, 80, 100),
            'sex': np.random.randint(0, 2, 100),
            'chol': np.random.randint(150, 400, 100),
            'trestbps': np.random.randint(90, 180, 100),
            'target': np.random.randint(0, 2, 100)
        })

    @pytest.fixture
    def preprocessor(self) -> DataPreprocessor:
        """Create a DataPreprocessor instance."""
        return DataPreprocessor(random_state=42)

    def test_preprocessor_initialization(self, preprocessor):
        """Test DataPreprocessor initialization."""
        assert preprocessor.random_state == 42
        assert hasattr(preprocessor, 'scaler')
        assert hasattr(preprocessor, 'imputer')

    def test_check_missing_values(self, preprocessor, sample_data):
        """Test missing values detection."""
        # Test with no missing values
        missing = preprocessor.check_missing_values(sample_data)
        assert missing.sum() == 0

        # Test with missing values
        sample_with_missing = sample_data.copy()
        sample_with_missing.loc[0, 'age'] = np.nan
        missing = preprocessor.check_missing_values(sample_with_missing)
        assert missing['age'] == 1

    def test_get_basic_stats(self, preprocessor, sample_data):
        """Test basic statistics extraction."""
        stats = preprocessor.get_basic_stats(sample_data)
        assert 'shape' in stats
        assert 'columns' in stats
        assert stats['shape'] == (100, 5)
        assert len(stats['columns']) == 5

    def test_split_data_stratified(self, preprocessor, sample_data):
        """Test stratified data splitting."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X, y, test_size=0.2, stratify=True
        )
        
        assert len(X_train) == int(100 * 0.8)
        assert len(X_test) == int(100 * 0.2)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling."""
        X_train = sample_data[['age', 'chol']].head(80)
        X_test = sample_data[['age', 'chol']].tail(20)
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Check shape is preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check scaling applied (mean ~0, std ~1)
        assert abs(X_train_scaled.mean().mean()) < 1e-10
        assert abs(X_train_scaled.std().mean() - 1) < 0.1

    def test_get_cross_validation_splits(self, preprocessor, sample_data):
        """Test cross-validation split generation."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        splits = preprocessor.get_cross_validation_splits(X, y, n_splits=5)
        split_list = list(splits)
        
        assert len(split_list) == 5
        for train_idx, test_idx in split_list:
            assert len(train_idx) + len(test_idx) == len(X)

    def test_preprocess_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline."""
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            sample_data, target_col='target', test_size=0.2
        )
        
        # Check shapes
        assert len(X_train) == int(100 * 0.8)
        assert len(X_test) == int(100 * 0.2)
        
        # Check data types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        
        # Check no NaN values
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
