"""Pytest configuration and fixtures for Heart Disease ML Classification tests."""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture(scope="session")
def sample_data():
    """
    Create a sample dataset for testing.
    
    Returns:
        pd.DataFrame: Sample heart disease dataset with 100 rows and 13 features.
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(30, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(100, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 7, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 5, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def test_features():
    """
    Get the list of feature columns for testing.
    
    Returns:
        list: List of feature column names.
    """
    return [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]


@pytest.fixture(scope="session")
def test_target():
    """
    Get the target column name for testing.
    
    Returns:
        str: Target column name.
    """
    return 'target'


@pytest.fixture
def train_test_split_data(sample_data, test_features, test_target):
    """
    Create a train-test split of the sample data.
    
    Args:
        sample_data: Sample dataset fixture.
        test_features: Features list fixture.
        test_target: Target column fixture.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) split data.
    """
    from sklearn.model_selection import train_test_split
    
    X = sample_data[test_features]
    y = sample_data[test_target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def scaled_data(train_test_split_data):
    """
    Create scaled train-test split data.
    
    Args:
        train_test_split_data: Train-test split fixture.
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test).
    """
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split_data
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """
    Reset random seeds before each test to ensure reproducibility.
    """
    np.random.seed(42)
    import random
    random.seed(42)
    yield
