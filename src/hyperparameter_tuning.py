"""Hyperparameter Tuning Module for Heart Disease Classification.

This module implements GridSearchCV for optimal hyperparameter selection
across Logistic Regression, Random Forest, and XGBoost classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Performs hyperparameter tuning for ML models."""

    def __init__(self, cv_folds: int = 5, scoring: str = 'roc_auc', n_jobs: int = -1):
        """
        Initialize the tuner.

        Parameters
        ----------
        cv_folds : int, default=5
            Number of cross-validation folds
        scoring : str, default='roc_auc'
            Scoring metric for evaluation
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = all processors)
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.tuned_models = {}
        self.cv_results = {}

    def tune_logistic_regression(self, X_train, y_train, verbose: int = 1) -> LogisticRegression:
        """Tune Logistic Regression hyperparameters.

        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        verbose : int, default=1
            Verbosity level

        Returns
        -------
        LogisticRegression
            Best tuned model
        """
        logger.info("Starting Logistic Regression tuning...")

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200, 500]
        }

        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            verbose=verbose,
            n_jobs=self.n_jobs
        )

        lr_grid.fit(X_train, y_train)
        self.tuned_models['LogisticRegression'] = lr_grid.best_estimator_
        self.cv_results['LogisticRegression'] = {
            'best_params': lr_grid.best_params_,
            'best_score': lr_grid.best_score_,
            'cv_results_df': pd.DataFrame(lr_grid.cv_results_)
        }

        logger.info(f"Best LR params: {lr_grid.best_params_}")
        logger.info(f"Best CV Score: {lr_grid.best_score_:.4f}")

        return lr_grid.best_estimator_

    def tune_random_forest(self, X_train, y_train, verbose: int = 1) -> RandomForestClassifier:
        """Tune Random Forest hyperparameters."""
        logger.info("Starting Random Forest tuning...")

        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            verbose=verbose,
            n_jobs=self.n_jobs
        )

        rf_grid.fit(X_train, y_train)
        self.tuned_models['RandomForest'] = rf_grid.best_estimator_
        self.cv_results['RandomForest'] = {
            'best_params': rf_grid.best_params_,
            'best_score': rf_grid.best_score_,
            'cv_results_df': pd.DataFrame(rf_grid.cv_results_)
        }

        logger.info(f"Best RF params: {rf_grid.best_params_}")
        logger.info(f"Best CV Score: {rf_grid.best_score_:.4f}")

        return rf_grid.best_estimator_

    def tune_xgboost(self, X_train, y_train, verbose: int = 1) -> XGBClassifier:
        """Tune XGBoost hyperparameters."""
        logger.info("Starting XGBoost tuning...")

        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            verbose=verbose,
            n_jobs=self.n_jobs
        )

        xgb_grid.fit(X_train, y_train)
        self.tuned_models['XGBoost'] = xgb_grid.best_estimator_
        self.cv_results['XGBoost'] = {
            'best_params': xgb_grid.best_params_,
            'best_score': xgb_grid.best_score_,
            'cv_results_df': pd.DataFrame(xgb_grid.cv_results_)
        }

        logger.info(f"Best XGB params: {xgb_grid.best_params_}")
        logger.info(f"Best CV Score: {xgb_grid.best_score_:.4f}")

        return xgb_grid.best_estimator_

    def get_tuned_models(self) -> dict:
        """Return all tuned models."""
        return self.tuned_models

    def get_cv_results(self) -> dict:
        """Return cross-validation results."""
        return self.cv_results

    def print_summary(self):
        """Print tuning summary."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*60)

        for model_name, results in self.cv_results.items():
            print(f"\n{model_name}:")
            print(f"  Best Parameters: {results['best_params']}")
            print(f"  Best CV Score: {results['best_score']:.4f}")
