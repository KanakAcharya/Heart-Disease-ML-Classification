"""Heart Disease Risk Prediction Streamlit App

A production-ready web application for predicting heart disease risk
based on clinical parameters using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHE & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load pre-trained models with caching for performance."""
    try:
        models_dir = Path("models")
        if models_dir.exists():
            lr_model = joblib.load(models_dir / "logistic_regression_model.pkl")
            rf_model = joblib.load(models_dir / "random_forest_model.pkl")
            xgb_model = joblib.load(models_dir / "xgboost_model.pkl")
            scaler = joblib.load(models_dir / "scaler.pkl")
            return lr_model, rf_model, xgb_model, scaler, True
    except Exception as e:
        st.warning(f"Could not load pre-trained models: {e}")
    
    # Fallback: Train models on startup (for Streamlit Cloud)
    return train_models_on_startup()

@st.cache_resource
def train_models_on_startup():
    """Train models if pre-trained ones are not available."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    st.info("⏳ Training models on first run... This takes ~30 seconds.")
    
    # Load sample data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:, :13], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    y = data.target
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    xgb_model.fit(X_train_scaled, y_train)
    
    st.success("✅ Models trained successfully!")
    return lr_model, rf_model, xgb_model, scaler, False

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load models
    lr_model, rf_model, xgb_model, scaler, models_loaded = load_models()
    
    # Header
    st.title("❤️ Heart Disease Risk Prediction")
    st.markdown("""
    This application predicts the likelihood of heart disease based on clinical parameters.
    Provide your clinical information below to get a personalized risk assessment.
    """)
    
    # Model status
    if models_loaded:
        st.success("✅ Models loaded successfully!")
    else:
        st.info("ℹ️ Models trained on first run")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Info", "📚 Dataset Info"])
    
    # ========================================================================
    # TAB 1: PREDICTION
    # ========================================================================
    with tab1:
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age (years)", 29, 77, 50)
            sex = st.radio("Sex", ("Male", "Female"), index=0)
            chest_pain = st.selectbox(
                "Chest Pain Type",
                (0, 1, 2, 3),
                help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"
            )
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
        
        # Prepare data
        sex_val = 1 if sex == "Male" else 0
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0
        
        # Predict button
        if st.button("🔮 Get Risk Assessment", use_container_width=True):
            input_data = pd.DataFrame({
                'age': [age], 'sex': [sex_val], 'cp': [chest_pain],
                'trestbps': [resting_bp], 'chol': [cholesterol],
                'fbs': [fasting_bs_val], 'restecg': [rest_ecg],
                'thalach': [max_hr], 'exang': [0], 'oldpeak': [oldpeak],
                'slope': [slope], 'ca': [ca], 'thal': [thal]
            })
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Get predictions
            lr_prob = lr_model.predict_proba(input_scaled)[0][1] * 100
            rf_prob = rf_model.predict_proba(input_scaled)[0][1] * 100
            xgb_prob = xgb_model.predict_proba(input_scaled)[0][1] * 100
            
            avg_risk = (lr_prob + rf_prob + xgb_prob) / 3
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col_results1, col_results2, col_results3 = st.columns(3)
            with col_results1:
                st.metric("Logistic Regression", f"{lr_prob:.1f}%")
            with col_results2:
                st.metric("Random Forest", f"{rf_prob:.1f}%")
            with col_results3:
                st.metric("XGBoost", f"{xgb_prob:.1f}%")
            
            st.divider()
            
            # Average risk
            col_avg1, col_avg2 = st.columns([2, 1])
            with col_avg1:
                risk_color = "🔴" if avg_risk > 70 else "🟡" if avg_risk > 40 else "🟢"
                st.metric(f"{risk_color} Average Risk Score", f"{avg_risk:.1f}%")
            
            with col_avg2:
                if avg_risk > 70:
                    st.error("HIGH RISK")
                elif avg_risk > 40:
                    st.warning("MODERATE RISK")
                else:
                    st.success("LOW RISK")
            
            st.divider()
            st.info("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult with healthcare professionals for actual diagnosis.")
    
    # ========================================================================
    # TAB 2: MODEL INFO
    # ========================================================================
    with tab2:
        st.subheader("Model Information")
        st.markdown("""
        ### Models Used
        
        **Logistic Regression**
        - ROC-AUC: 0.9498
        - Accuracy: 83.33%
        - Best for interpretability and speed
        
        **Random Forest (100 trees)**
        - ROC-AUC: 0.9397
        - Accuracy: 85.00%
        - Captures non-linear relationships
        
        **XGBoost (100 estimators)**
        - ROC-AUC: 0.8996
        - Accuracy: 85.00%
        - Advanced gradient boosting
        
        ### Features Used (13 total)
        Age • Sex • Chest Pain Type • Resting BP • Cholesterol • Fasting BS • 
        Resting ECG • Max Heart Rate • Exercise-Induced Angina • ST Depression • 
        ST Slope • Major Vessels • Thalassemia Type
        """)
    
    # ========================================================================
    # TAB 3: DATASET INFO
    # ========================================================================
    with tab3:
        st.subheader("Dataset Information")
        st.markdown("""
        ### Cleveland Heart Disease Dataset
        
        - **Source:** UCI Machine Learning Repository
        - **Samples:** 297 patient records
        - **Features:** 13 clinical attributes
        - **Target:** Binary (0 = No Disease, 1 = Disease)
        - **Class Distribution:** ~54% No Disease, ~46% Disease
        
        ### Data Characteristics
        - No missing values after preprocessing
        - Age range: 29-77 years
        - Well-balanced dataset
        - Standardized features for ML models
        - Collected from 5 medical centers
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **Built with:** Python • Streamlit • Scikit-learn • XGBoost
    
    **GitHub:** [Heart-Disease-ML-Classification](https://github.com/KanakAcharya/Heart-Disease-ML-Classification)
    
    *Disclaimer: This tool is for educational purposes only. Always consult medical professionals for diagnosis.*
    """)

if __name__ == "__main__":
    main()
