# Heart Disease ML Classification - End-to-End Machine Learning Project

## Overview
This is a comprehensive end-to-end machine learning classification project that predicts the presence of heart disease in patients. The project demonstrates a complete ML pipeline including data exploration, preprocessing, feature engineering, model training, and evaluation.

## Dataset
- **Source**: UCI Machine Learning Repository (Cleveland Heart Disease Dataset)
- **Samples**: 297 records
- **Features**: 13 clinical and demographic attributes
- **Target Variable**: Binary (0 = No Heart Disease, 1 = Heart Disease)
- **Class Distribution**: 53.87% No Disease, 46.13% Disease

## Project Pipeline

### 1. Data Exploration & Analysis
- Dataset shape and structure analysis
- Statistical summary and distribution analysis
- Missing values identification
- Target variable distribution visualization
- Correlation matrix visualization

### 2. Data Preprocessing
- Handling missing values
- Data cleaning and normalization
- Train-test split (80-20 stratified split)
- Feature scaling using StandardScaler

### 3. Model Training
Three machine learning algorithms were trained and evaluated:

1. **Logistic Regression**
   - Train Accuracy: 85.23%
   - Test Accuracy: 83.33%
   - Precision: 84.62%
   - Recall: 78.57%
   - F1-Score: 81.48%
   - ROC-AUC: 0.9498

2. **Random Forest (100 trees, max_depth=10)**
   - Train Accuracy: 100.00% (slight overfitting)
   - Test Accuracy: 85.00%
   - Precision: 88.00%
   - Recall: 78.57%
   - F1-Score: 83.02%
   - ROC-AUC: 0.9397

3. **XGBoost (100 estimators, max_depth=6)**
   - Train Accuracy: 100.00%
   - Test Accuracy: 85.00%
   - Precision: 88.00%
   - Recall: 78.57%
   - F1-Score: 83.02%
   - ROC-AUC: 0.8996

### 4. Model Evaluation Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve
- **Confusion Matrix**: For detailed classification breakdown

## Key Findings
- Logistic Regression achieved the highest ROC-AUC score (0.9498), indicating excellent discrimination ability
- All models demonstrate good generalization with test accuracies around 83-85%
- Random Forest and XGBoost show signs of overfitting on training data
- Feature scaling was crucial for Logistic Regression and XGBoost performance

## Technologies & Libraries Used
- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models and metrics
- **XGBoost**: Gradient boosting framework
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## How to Use

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Run the Project
1. Open the Jupyter Notebook: `Heart_Disease_Classification_ML_Pipeline.ipynb`
2. Execute all cells to:
   - Load and explore the dataset
   - Perform data preprocessing
   - Train all three models
   - Generate evaluation metrics and visualizations
   - Compare model performance

## Project Structure
```
Heart-Disease-ML-Classification/
├── README.md
├── Heart_Disease_Classification_ML_Pipeline.ipynb
└── requirements.txt
```

## Model Performance Summary
Logistic Regression is recommended for this task due to:
- Highest ROC-AUC score
- Best precision-recall balance
- Simpler model (better interpretability)
- Fast inference time

## Future Improvements
- Hyperparameter tuning using GridSearchCV
- Cross-validation for robust performance estimation
- Feature importance analysis
- Class imbalance handling techniques
- Ensemble methods combining multiple models
- Deployment as a REST API

## Author
Data Science Portfolio - Machine Learning

## License
MIT License

## Contact
For questions or collaborations, please reach out via GitHub.
