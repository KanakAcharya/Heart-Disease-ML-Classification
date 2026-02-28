"""Model training script for Heart Disease Prediction.

This script downloads the UCI Heart Disease Dataset, trains three ML models
(Logistic Regression, Random Forest, and XGBoost), and saves them as pickled artifacts.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download and load the Cleveland Heart Disease Dataset
print("Loading Cleveland Heart Disease Dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

try:
    df = pd.read_csv(url, names=column_names, na_values='?')
except:
    print("Could not download dataset. Using sample data for demonstration.")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 297
    df = pd.DataFrame({
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(125, 565, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(60, 202, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0.0, 6.2, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Data preprocessing
df = df.dropna()
X = df.drop('target', axis=1)
y = df['target']

# Convert target to binary (0 or 1)
y = (y > 0).astype(int)

print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Class distribution (%):\n{y.value_counts(normalize=True) * 100}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape}")
print(f"Test set size: {X_test_scaled.shape}")

# Train models
print("\n" + "="*50)
print("Training Models...")
print("="*50)

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_train_score = lr_model.score(X_train_scaled, y_train)
lr_test_score = lr_model.score(X_test_scaled, y_test)
print(f"   Train Accuracy: {lr_train_score:.4f}")
print(f"   Test Accuracy: {lr_test_score:.4f}")
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
print(f"   Model saved to models/logistic_regression_model.pkl")

# 2. Random Forest
print("\n2. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_train_score = rf_model.score(X_train_scaled, y_train)
rf_test_score = rf_model.score(X_test_scaled, y_test)
print(f"   Train Accuracy: {rf_train_score:.4f}")
print(f"   Test Accuracy: {rf_test_score:.4f}")
joblib.dump(rf_model, 'models/random_forest_model.pkl')
print(f"   Model saved to models/random_forest_model.pkl")

# 3. XGBoost
print("\n3. Training XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False,
                          eval_metric='logloss', verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_train_score = xgb_model.score(X_train_scaled, y_train)
xgb_test_score = xgb_model.score(X_test_scaled, y_test)
print(f"   Train Accuracy: {xgb_train_score:.4f}")
print(f"   Test Accuracy: {xgb_test_score:.4f}")
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print(f"   Model saved to models/xgboost_model.pkl")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print(f"\n   Scaler saved to models/scaler.pkl")

print("\n" + "="*50)
print("✅ All models trained and saved successfully!")
print("="*50)
print(f"\nModels are available in the 'models/' directory:")
print("  - logistic_regression_model.pkl")
print("  - random_forest_model.pkl")
print("  - xgboost_model.pkl")
print("  - scaler.pkl")
