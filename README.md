# Heart Disease ML Classification - End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python) ![Tests](https://img.shields.io/badge/Tests-pytest-green?style=flat) ![Docker](https://img.shields.io/badge/Docker-Supported-blue?style=flat&logo=docker) ![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow?style=flat&logo=github-actions) ![License](https://img.shields.io/badge/License-MIT-green?style=flat) ![Codecov](https://img.shields.io/codecov/c/github/KanakAcharya/Heart-Disease-ML-Classification?style=flat&logo=codecov) ![Trivy](https://img.shields.io/badge/Security-Trivy%20Scanned-blue?style=flat&logo=security)

## Overview
This is a comprehensive end-to-end machine learning classification project that predicts the presence of heart disease in patients. The project demonstrates a complete ML pipeline including data exploration, preprocessing, feature engineering, model training, and evaluation.

## ✨ Quick Links

- 🚀 **[Try the Live Demo](https://share.streamlit.io/kanakach arya/Heart-Disease-ML-Classification/main/app.py)** (Streamlit Cloud)
- 📖 **[Full Documentation](https://github.com/KanakAcharya/Heart-Disease-ML-Classification/blob/main/README.md)**
- 📊 **[View Models Comparison](https://github.com/KanakAcharya/Heart-Disease-ML-Classification#key-metrics)**
- 🐳 **[Docker Setup Guide](https://github.com/KanakAcharya/Heart-Disease-ML-Classification#docker-setup)**

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


## Advanced Features

### Testing Framework
- **Pytest**: Comprehensive unit and integration tests
- **Test Coverage**: Full coverage for data preprocessing and model training
- **Fixtures**: Pytest fixtures for reproducible test data
- **Coverage Reports**: CI/CD integration with Codecov

### Docker Support
- **Containerization**: Complete Docker support for reproducible environments
- **Image**: Multi-stage builds for optimized container size
- **Deployment**: Ready for cloud deployment (AWS, GCP, Azure)

### Code Quality
- **Linting**: Pylint, Flake8 for code quality checks
- **Type Checking**: Mypy for static type analysis
- **Formatting**: Black code formatter for consistent style
- **Documentation**: Comprehensive docstrings and module documentation

### Continuous Integration/Continuous Deployment
- **GitHub Actions**: Automated testing on Python 3.8, 3.9, 3.10
- **Security Scanning**: Trivy vulnerability scanner integration
- **Code Coverage**: Automatic coverage reports and tracking
- **Docker Build**: Automated Docker image creation and testing

## Installation & Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/KanakAcharya/Heart-Disease-ML-Classification.git
cd Heart-Disease-ML-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build Docker image
docker build -t heart-disease-classifier:latest .

# Run container
docker run -p 8501:8501 heart-disease-classifier:latest

# Access Streamlit app at http://localhost:8501
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with parallel execution (faster)
pytest tests/ -v -n auto
```

## Project Highlights

- ✅ **Production-Ready Code**: Follows best practices and PEP 8 standards
- ✅ **Comprehensive Documentation**: Detailed docstrings and README
- ✅ **Automated Testing**: 100% test coverage for core modules
- ✅ **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment
- ✅ **Docker Support**: Containerized application for easy deployment
- ✅ **Model Explainability**: SHAP analysis for model interpretability
- ✅ **Data Validation**: Robust data preprocessing and validation
- ✅ **Performance Optimization**: Hyperparameter tuning and model optimization

## Key Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | 83.33% | 84.62% | 78.57% | 81.48% | 0.9498 |
| Random Forest | 85.00% | 88.00% | 78.57% | 83.02% | 0.9397 |
| XGBoost | 85.00% | 88.00% | 78.57% | 83.02% | 0.8996 |

## Project Roadmap

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed information about:
- Planned enhancements and features
- Development priorities
- Timeline estimates
- Architecture improvements

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Deployment

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Contact & Support

- GitHub: [@KanakAcharya](https://github.com/KanakAcharya)
- Issues: [GitHub Issues](https://github.com/KanakAcharya/Heart-Disease-ML-Classification/issues)

---

## Core Modules (src/)

The modular architecture separates concerns into independent, testable components:

### data_preprocessing.py
- **DataPreprocessor Class**: Handles all data preparation tasks
  - `load_data()`: Load CSV files with error handling
  - `check_missing_values()`: Identify and handle missing values
  - `handle_missing_values()`: Imputation strategies
  - `get_features()`: Extract feature columns
  - `handle_outliers()`: Detect and handle outliers
  - `normalize_data()`: StandardScaler normalization
  - Full test coverage in `tests/test_preprocessing.py`

### model_trainer.py
- **ModelTrainer Class**: End-to-end model training pipeline
  - `train_logistic_regression()`: Logistic Regression classifier
  - `train_random_forest()`: Random Forest with tuned hyperparameters
  - `train_xgboost()`: XGBoost gradient boosting
  - `save_model()`: Serialize trained models using joblib
  - `load_model()`: Load pre-trained models
  - Comprehensive test coverage in `tests/test_model_trainer.py`

### model_evaluator.py
- **ModelEvaluator Class**: Performance metrics and analysis
  - Classification metrics (accuracy, precision, recall, F1)
  - ROC-AUC and confusion matrix generation
  - Cross-validation scoring
  - Model comparison utilities

### hyperparameter_tuning.py
- **HyperparameterTuner Class**: Automated hyperparameter optimization
  - GridSearchCV for exhaustive parameter search
  - RandomizedSearchCV for large parameter spaces
  - Cross-validation integration
  - Best parameters tracking and reporting

### shap_analysis.py
- **SHAPAnalyzer Class**: Model explainability and interpretability
  - SHAP value computation for feature importance
  - Force plots for individual predictions
  - Summary plots for global feature importance
  - Model agnostic explanations

## Live Demo Deployment

### Streamlit Cloud (Recommended)

1. **Push code to GitHub** (already done)
2. **Create Streamlit account**: Visit https://share.streamlit.io
3. **Deploy app**:
   ```
   - Click "New app"
   - Select your GitHub repository
   - Set main file path to: app.py
   - Click Deploy
   ```
4. **Access your app**: https://share.streamlit.io/[YourUsername]/[RepoName]/main/app.py

### Alternative Deployment Options

#### Heroku (Free tier deprecated, use paid or alternatives)
```bash
# Create Procfile
echo 'web: sh setup.sh && streamlit run app.py' > Procfile

# Push to Heroku
git push heroku main
```

#### AWS EC2
- Launch t2.micro instance
- Install Python 3.9 and dependencies
- Run: `streamlit run app.py --server.port 80`

#### Google Cloud Run
```bash
echo 'FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]' > Dockerfile

# Deploy
gcloud run deploy heart-disease-classifier --source .
```

## Model Artifacts

**Note on .pkl Model Files**: To keep the repository lightweight and enable CI/CD testing without cached artifacts:
- Trained models are NOT committed to the repository
- Models are regenerated during the Streamlit app startup
- The application trains models on first load (training takes ~5 seconds)
- For production deployments with persistent models:
  - Train models locally: `python src/model_trainer.py`
  - Save models: `joblib.dump(model, 'models/model_name.pkl')`
  - Commit to `.gitignore` to prevent large file bloat

## Documentation Files

- **README.md** (this file): Project overview and getting started
- **IMPROVEMENTS.md**: Detailed roadmap for enhancements
- **DEPLOYMENT.md**: Advanced deployment strategies
- **CONTRIBUTING.md**: Guidelines for contributors

**Note**: This project is maintained for educational and portfolio purposes. For production medical applications, please consult with healthcare professionals and ensure compliance with relevant regulations.
## Contact
For questions or collaborations, please reach out via GitHub.
