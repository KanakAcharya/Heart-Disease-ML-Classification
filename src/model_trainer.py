import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and evaluation."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.grid_searches = {}
        
    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """Train Logistic Regression model."""
        lr_params = {
            'max_iter': kwargs.get('max_iter', 1000),
            'random_state': self.random_state,
            'solver': kwargs.get('solver', 'lbfgs')
        }
        lr = LogisticRegression(**lr_params)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        return lr
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model."""
        rf_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 10),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        return rf
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost model."""
        xgb_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'random_state': self.random_state,
            'verbosity': 0,
            'eval_metric': 'logloss'
        }
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = xgb
        return xgb
    
    def hyperparameter_tuning_rf(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning for Random Forest using GridSearchCV."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.grid_searches['Random Forest'] = grid_search
        self.best_models['Random Forest'] = grid_search.best_estimator_
        
        print(f"Best RF parameters: {grid_search.best_params_}")
        print(f"Best RF CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def hyperparameter_tuning_xgb(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning for XGBoost using GridSearchCV."""
        param_grid = {
            'max_depth': [4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 150],
            'subsample': [0.7, 0.9]
        }
        
        xgb = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        grid_search = GridSearchCV(xgb, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.grid_searches['XGBoost'] = grid_search
        self.best_models['XGBoost'] = grid_search.best_estimator_
        
        print(f"Best XGB parameters: {grid_search.best_params_}")
        print(f"Best XGB CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def cross_validate_model(self, model, X_train, y_train, cv=5, scoring='roc_auc'):
        """Perform cross-validation for a model."""
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
    
    def save_model(self, model_name, filepath):
        """Save a trained model to disk."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        elif model_name in self.best_models:
            joblib.dump(self.best_models[model_name], filepath)
            print(f"Best model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found.")
    
    def load_model(self, filepath):
        """Load a saved model from disk."""
        model = joblib.load(filepath)
        return model
