"""SHAP Explainability Analysis Module for Model Interpretability.

This module provides tools for SHAP-based feature importance and
model prediction explanations.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Performs SHAP-based explainability analysis."""

    def __init__(self, feature_names: list = None):
        """
        Initialize SHAP analyzer.

        Parameters
        ----------
        feature_names : list, optional
            Names of features for visualization
        """
        self.feature_names = feature_names
        self.explainers = {}
        self.shap_values = {}

    def analyze_logistic_regression(self, model, X_test, X_train=None):
        """Analyze Logistic Regression with SHAP.

        Parameters
        ----------
        model : LogisticRegression
            Fitted logistic regression model
        X_test : array-like
            Test features for explanation
        X_train : array-like, optional
            Training data for background

        Returns
        -------
        dict
            SHAP values and explainer
        """
        logger.info("Analyzing Logistic Regression with SHAP...")

        background_data = X_train if X_train is not None else X_test[:100]
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
        shap_values = explainer.shap_values(X_test)

        self.explainers['LogisticRegression'] = explainer
        self.shap_values['LogisticRegression'] = shap_values

        return {'explainer': explainer, 'shap_values': shap_values}

    def analyze_tree_model(self, model, X_test, model_name: str = 'TreeModel'):
        """Analyze tree-based model with SHAP (faster).

        Parameters
        ----------
        model : RandomForestClassifier or XGBClassifier
            Fitted tree-based model
        X_test : array-like
            Test features for explanation
        model_name : str
            Name of the model

        Returns
        -------
        dict
            SHAP values and explainer
        """
        logger.info(f"Analyzing {model_name} with SHAP...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        self.explainers[model_name] = explainer
        self.shap_values[model_name] = shap_values

        return {'explainer': explainer, 'shap_values': shap_values}

    def plot_summary(self, model_name: str, save_path: str = None):
        """Plot SHAP summary plot (bar).

        Parameters
        ----------
        model_name : str
            Name of the model
        save_path : str, optional
            Path to save the figure
        """
        if model_name not in self.shap_values:
            logger.warning(f"No SHAP values found for {model_name}")
            return

        logger.info(f"Creating summary plot for {model_name}...")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values[model_name],
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        plt.show()

    def plot_dependence(self, model_name: str, feature_idx: int, save_path: str = None):
        """Plot SHAP dependence plot for a feature.

        Parameters
        ----------
        model_name : str
            Name of the model
        feature_idx : int
            Index of the feature
        save_path : str, optional
            Path to save the figure
        """
        if model_name not in self.shap_values:
            logger.warning(f"No SHAP values found for {model_name}")
            return

        logger.info(f"Creating dependence plot for feature {feature_idx}...")

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            self.shap_values[model_name],
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")

        plt.show()

    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Get feature importance from SHAP values.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        pd.DataFrame
            Feature importance DataFrame
        """
        if model_name not in self.shap_values:
            logger.warning(f"No SHAP values found for {model_name}")
            return None

        shap_vals = self.shap_values[model_name]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # For binary classification

        importance = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names or range(len(importance)),
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        return importance_df
