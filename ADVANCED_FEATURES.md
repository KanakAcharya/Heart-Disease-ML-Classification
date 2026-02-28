# Advanced ML Features & Techniques

This document outlines advanced machine learning techniques and improvements implemented in the Heart Disease ML Classification project to address performance optimization and production readiness.

## 1. Hyperparameter Tuning

### GridSearchCV Implementation
```python
from sklearn.model_selection import GridSearchCV

# Logistic Regression Tuning
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'solver': ['lbfgs', 'liblinear'],
             'max_iter': [100, 200, 500]}
lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=5, scoring='roc_auc')
lr_grid.fit(X_train_scaled, y_train)
print(f"Best LR params: {lr_grid.best_params_}")
print(f"Best CV Score: {lr_grid.best_score_:.4f}")

# Random Forest Tuning
rf_params = {'n_estimators': [50, 100, 150, 200],
             'max_depth': [5, 10, 15, 20],
             'min_samples_leaf': [1, 2, 4],
             'max_features': ['sqrt', 'log2']}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)

# XGBoost Tuning
xgb_params = {'learning_rate': [0.01, 0.05, 0.1],
              'n_estimators': [100, 200, 300],
              'max_depth': [3, 5, 7],
              'subsample': [0.8, 0.9, 1.0],
              'colsample_bytree': [0.8, 0.9, 1.0]}
xgb_grid = GridSearchCV(XGBClassifier(), xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)
```

**Expected Improvements**: +5-10% accuracy boost from baseline models.

## 2. K-Fold Cross-Validation

### Robust Model Evaluation
```python
from sklearn.model_selection import cross_val_score, cross_validate

# Comprehensive cross-validation metrics
scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall',
           'f1': 'f1',
           'roc_auc': 'roc_auc'}

for model_name, model in [('LogisticRegression', lr_best),
                          ('RandomForest', rf_best),
                          ('XGBoost', xgb_best)]:
    cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, scoring=scoring)
    print(f"\n{model_name} Cross-Validation Results:")
    for metric, scores in cv_results.items():
        if 'test_' in metric:
            metric_name = metric.replace('test_', '')
            print(f"  {metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## 3. SMOTE for Class Imbalance

### Handling Imbalanced Data
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Original class distribution
print(f"Class distribution before SMOTE:")
print(y_train.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Train models on balanced data
for model, name in [(lr_best, 'LogisticRegression'),
                    (rf_best, 'RandomForest'),
                    (xgb_best, 'XGBoost')]:
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} with SMOTE - Recall: {recall_score(y_test, y_pred):.4f}")
```

**Benefits**: Improved recall for minority class (disease cases).

## 4. SHAP Explainability Analysis

### Feature Importance & Model Interpretability
```python
import shap
import matplotlib.pyplot as plt

# SHAP for Logistic Regression
explainer_lr = shap.Explainer(lr_best, X_train_scaled, feature_names=X.columns)
shap_values_lr = explainer_lr(X_test_scaled)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_lr, X_test_scaled, feature_names=X.columns, plot_type='bar')
plt.title('SHAP Summary Plot - Feature Importance (Logistic Regression)')
plt.tight_layout()
plt.savefig('shap_summary_lr.png', dpi=300, bbox_inches='tight')
plt.show()

# Force plot for individual prediction
shap.initjs()
for idx in [0, 10, 20]:  # Sample predictions
    shap.force_plot(explainer_lr.expected_value, shap_values_lr[idx], X_test_scaled[idx], feature_names=X.columns)

# SHAP for Tree-based models (faster)
explainer_rf = shap.TreeExplainer(rf_best)
shap_values_rf = explainer_rf.shap_values(X_test)

# Dependence plot
plt.figure(figsize=(12, 4))
for i, feature in enumerate(X.columns[:3]):
    plt.subplot(1, 3, i+1)
    shap.dependence_plot(feature, shap_values_rf[1], X_test, feature_names=X.columns)
plt.tight_layout()
plt.savefig('shap_dependence_rf.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 5. ROC Curves & Advanced Visualizations

### Model Performance Comparison
```python
from sklearn.metrics import RocCurveDisplay, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

for model, name in [(lr_best, 'Logistic Regression'),
                    (rf_best, 'Random Forest'),
                    (xgb_best, 'XGBoost')]:
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance Level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 6. Feature Importance Analysis

### Identifying Key Predictors
```python
# Logistic Regression Coefficients
lr_coef = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_best.coef_[0]})
lr_coef['Abs_Coef'] = lr_coef['Coefficient'].abs()
lr_coef_sorted = lr_coef.sort_values('Abs_Coef', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(lr_coef_sorted['Feature'], lr_coef_sorted['Coefficient'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Logistic Regression Feature Coefficients', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# Random Forest Feature Importance
rf_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_best.feature_importances_})
rf_importance_sorted = rf_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(rf_importance_sorted['Feature'], rf_importance_sorted['Importance'], color='forestgreen')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# XGBoost Feature Importance
xgb_importance = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_best.feature_importances_})
xgb_importance_sorted = xgb_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(xgb_importance_sorted['Feature'], xgb_importance_sorted['Importance'], color='darkorange')
plt.xlabel('Importance Score', fontsize=12)
plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('xgb_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 7. Model Ensembling

### Combining Multiple Models
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression as LR

# Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[('lr', lr_best), ('rf', rf_best), ('xgb', xgb_best)],
    voting='soft'  # Use soft voting (probability averaging)
)
voting_clf.fit(X_train_smote, y_train_smote)
y_pred_voting = voting_clf.predict(X_test_scaled)

print(f"Voting Ensemble - Accuracy: {accuracy_score(y_test, y_pred_voting):.4f}")
print(f"Voting Ensemble - ROC-AUC: {roc_auc_score(y_test, voting_clf.predict_proba(X_test_scaled)[:, 1]):.4f}")

# Stacking Classifier (Meta-learner approach)
stacking_clf = StackingClassifier(
    estimators=[('lr', lr_best), ('rf', rf_best), ('xgb', xgb_best)],
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train_smote, y_train_smote)
y_pred_stacking = stacking_clf.predict(X_test_scaled)

print(f"\nStacking Ensemble - Accuracy: {accuracy_score(y_test, y_pred_stacking):.4f}")
print(f"Stacking Ensemble - ROC-AUC: {roc_auc_score(y_test, stacking_clf.predict_proba(X_test_scaled)[:, 1]):.4f}")
```

## 8. Calibration & Uncertainty Quantification

```python
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# Model Calibration
calibrated_lr = CalibratedClassifierCV(lr_best, method='sigmoid', cv=5)
calibrated_lr.fit(X_train_smote, y_train_smote)
y_proba_calibrated = calibrated_lr.predict_proba(X_test_scaled)[:, 1]

print(f"Calibrated Model Probability Range: [{y_proba_calibrated.min():.4f}, {y_proba_calibrated.max():.4f}]")
```

## Expected Performance Improvements

| Technique | Expected Accuracy Gain | Recall Improvement | Notes |
|-----------|----------------------|-------------------|-------|
| GridSearchCV | +3-8% | +2-5% | Significant for tree models |
| K-Fold CV | +1-2% | Stability | Better generalization estimate |
| SMOTE | +0-3% | +5-10% | Critical for recall in medical |
| Ensembling | +2-5% | +2-4% | Combines strengths of models |
| SHAP | N/A | N/A | Explainability, not accuracy |

## Integration Notes

- All advanced techniques are compatible with the existing pipeline
- Ensure dependency versions from `requirements.txt` are installed
- GPU acceleration available for XGBoost with `gpu_hist` tree_method
- Parallelize GridSearchCV with `n_jobs=-1` for faster computation

## References

- SHAP Documentation: https://shap.readthedocs.io/
- imbalanced-learn: https://imbalanced-learn.org/
- scikit-optimize: https://scikit-optimize.github.io/
- Scikit-learn Ensemble Methods: https://scikit-learn.org/stable/modules/ensemble.html
