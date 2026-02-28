import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics and visualizations."""
    
    def __init__(self):
        self.results = {}
        self.confusion_matrices = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        self.results[model_name] = metrics
        self.confusion_matrices[model_name] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def get_classification_report(self, y_true, y_pred, model_name="Model"):
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, output_dict=True)
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """Plot confusion matrix for a model."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="Model"):
        """Plot ROC curve for a model."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        return plt
    
    def compare_models(self):
        """Compare performance across all evaluated models."""
        if not self.results:
            print("No models evaluated yet.")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        return comparison_df
    
    def print_results(self):
        """Print comprehensive evaluation results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        print("\n" + "="*60)
    
    def get_feature_importance(self, model, feature_names, top_n=10):
        """Extract and return top feature importances from tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            return importance_df
        else:
            print(f"Model {type(model).__name__} does not have feature_importances_.")
            return None
    
    def plot_feature_importance(self, model, feature_names, top_n=10, model_name="Model"):
        """Plot feature importance for tree-based models."""
        importance_df = self.get_feature_importance(model, feature_names, top_n)
        
        if importance_df is not None:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            return plt
        return None
