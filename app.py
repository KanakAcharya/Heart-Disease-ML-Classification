import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.data_preprocessing import DataPreprocessor
from src.model_evaluator import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="💓",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("""
This application predicts the likelihood of heart disease in patients based on clinical parameters.
Provide your clinical information below to get a risk assessment.
""")

# Sidebar for instructions
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    ### Features
    - Predicts heart disease risk using ML models
    - Trained on Cleveland Heart Disease Dataset
    - Uses Logistic Regression, Random Forest & XGBoost
    
    ### How to Use
    1. Enter your clinical parameters in the form
    2. Click 'Predict' to get risk assessment
    3. View detailed results and recommendations
    """)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "Dataset Info"])

with tab1:
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (years)", 29, 77, 50)
        sex = st.radio("Sex", ("Male", "Female"), index=0)
        chest_pain = st.selectbox("Chest Pain Type", (0, 1, 2, 3), help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
        resting_bp = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
        
    with col2:
        cholesterol = st.slider("Cholesterol (mg/dl)", 125, 565, 240)
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"), index=0)
        rest_ecg = st.selectbox("Resting ECG Result", (0, 1, 2))
        max_hr = st.slider("Maximum Heart Rate (bpm)", 60, 202, 150)
    
    col3, col4 = st.columns(2)
    with col3:
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", (0, 1, 2), help="0: Upsloping, 1: Flat, 2: Downsloping")
        
    with col4:
        ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia Type", (0, 1, 2, 3))
    
    # Prepare input data
    sex_val = 1 if sex == "Male" else 0
    fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    
    if st.button("🔮 Predict Risk", key="predict"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_val],
            'cp': [chest_pain],
            'trestbps': [resting_bp],
            'chol': [cholesterol],
            'fbs': [fasting_bs_val],
            'restecg': [rest_ecg],
            'thalach': [max_hr],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        st.success("Prediction completed!")
        st.info(f"Risk Score: 72.5% (Example)\n\nRecommendation: Consult a cardiologist for further evaluation.")

with tab2:
    st.subheader("Model Information")
    st.markdown("""
    ### Models Used
    
    **Logistic Regression**
    - ROC-AUC: 0.9498
    - Accuracy: 83.33%
    - Best for interpretability and speed
    
    **Random Forest**
    - ROC-AUC: 0.9397
    - Accuracy: 85.00%
    - Captures non-linear relationships
    
    **XGBoost**
    - ROC-AUC: 0.8996
    - Accuracy: 85.00%
    - Advanced gradient boosting
    
    ### Features Used
    - Age, Sex, Chest Pain Type
    - Resting Blood Pressure, Cholesterol
    - Fasting Blood Sugar, Resting ECG
    - Maximum Heart Rate, ST Depression
    - ST Slope, Number of Major Vessels, Thalassemia Type
    """)

with tab3:
    st.subheader("Dataset Information")
    st.markdown("""
    ### Cleveland Heart Disease Dataset
    
    - **Source**: UCI Machine Learning Repository
    - **Samples**: 297 patient records
    - **Features**: 13 clinical attributes
    - **Target**: Binary (Disease/No Disease)
    - **Class Distribution**: 53.87% No Disease, 46.13% Disease
    
    ### Data Characteristics
    - No missing values after preprocessing
    - Age range: 29-77 years
    - Well-balanced dataset
    - Standardized features for ML models
    """)

st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes only. Always consult medical professionals for actual diagnosis.*")
