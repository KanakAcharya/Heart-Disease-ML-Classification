import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing for the Heart Disease dataset."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self, filepath):
        """Load CSV data from filepath."""
        return pd.read_csv(filepath)
    
    def check_missing_values(self, df):
        """Check for missing values in dataset."""
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("Missing values found:")
            print(missing[missing > 0])
        else:
            print("No missing values found.")
        return missing
    
    def get_basic_stats(self, df):
        """Get basic statistics of the dataset."""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'describe': df.describe()
        }
    
    def impute_missing_values(self, X):
        """Impute missing values using mean strategy."""
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        return X_imputed
    
    def split_data(self, X, y, test_size=0.2, stratify=True):
        """Split data into train and test sets."""
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state
            )
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, df, target_col='target', test_size=0.2):
        """Complete preprocessing pipeline."""
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Impute missing values
        X = self.impute_missing_values(X)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_cross_validation_splits(self, X, y, n_splits=5):
        """Get stratified K-fold cross-validation splits."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return skf.split(X, y)
