import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="CardioCare ML Diagnostics", page_icon="🫀", layout="wide")

# --- 2. DYNAMIC MODEL LOADING ---
@st.cache_resource
def load_ml_models():
    model_dir = './deploy_models'
    uci_path = os.path.join(model_dir, 'uci_clinical_model.joblib')
    cdc_path = os.path.join(model_dir, 'cdc_survey_model.joblib')
    try:
        return joblib.load(uci_path), joblib.load(cdc_path)
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure they are saved in the 'deploy_models' directory.")
        st.stop()

uci_model, cdc_model = load_ml_models()

# --- 3. USER INTERFACE ---
st.title("🫀 CardioCare: Two-Tier Heart Disease Prediction System")
st.markdown("""
This system utilizes a **Stacking Ensemble Machine Learning Architecture**.
* **Tier 1 (Survey):** Evaluates lifestyle risk factors.
* **Tier 2 (Clinical):** Evaluates medical biomarkers.
""")

tab1, tab2 = st.tabs(["Tier 1: Lifestyle Screening", "Tier 2: Clinical Diagnostics"])

with tab1:
    st.subheader("Lifestyle & Survey Assessment")
    col1, col2 = st.columns(2)
    cdc_inputs = {}

    with col1:
        cdc_inputs['age_numeric'] = st.number_input("Age", min_value=18, max_value=120, value=50)
        cdc_inputs['BMI'] = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=80.0, value=25.0)
        cdc_inputs['Smoking'] = st.selectbox("Smoking History", ["Yes", "No"])
        cdc_inputs['AlcoholDrinking'] = st.selectbox("Heavy Alcohol Consumption", ["Yes", "No"])
        cdc_inputs['Stroke'] = st.selectbox("History of Stroke", ["Yes", "No"])
        cdc_inputs['PhysicalHealth'] = st.slider("Days of poor physical health (last 30 days)", 0, 30, 0)
        cdc_inputs['MentalHealth'] = st.slider("Days of poor mental health (last 30 days)", 0, 30, 0)
        cdc_inputs['DiffWalking'] = st.selectbox("Difficulty Walking/Climbing Stairs", ["Yes", "No"])

    with col2:
        cdc_inputs['Race'] = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"])
        cdc_inputs['Diabetic'] = st.selectbox("Diabetic Status", ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"])
        cdc_inputs['PhysicalActivity'] = st.selectbox("Physical Activity in last 30 days", ["Yes", "No"])
        cdc_inputs['GenHealth'] = st.selectbox("General Health Assessment", ["Excellent", "Very good", "Good", "Fair", "Poor"])
        cdc_inputs['SleepTime'] = st.number_input("Average Hours of Sleep", min_value=1.0, max_value=24.0, value=7.0)
        cdc_inputs['Asthma'] = st.selectbox("Asthma History", ["Yes", "No"])
        cdc_inputs['KidneyDisease'] = st.selectbox("Kidney Disease History", ["Yes", "No"])
        cdc_inputs['SkinCancer'] = st.selectbox("Skin Cancer History", ["Yes", "No"])

    if st.button("Run Screening Pipeline", key="btn_cdc"):
        input_df = pd.DataFrame([cdc_inputs])
        prediction = cdc_model.predict(input_df)[0]
        prob = cdc_model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **Elevated Risk Detected** (Confidence: {prob:.1%}). Proceed to Clinical Diagnostics.")
        else:
            st.success(f"✅ **Low Risk Detected** (Confidence: {1-prob:.1%}). Maintain healthy lifestyle.")

with tab2:
    st.subheader("Clinical Biomarker Assessment")
    col3, col4 = st.columns(2)
    uci_inputs = {}

    with col3:
        uci_inputs['age'] = st.number_input("Patient Age (Clinical)", min_value=20, max_value=100, value=50)
        # ADDED: sex_clean to match the model's training features
        uci_inputs['sex_clean'] = st.selectbox("Biological Sex", ["Male", "Female"])
        uci_inputs['cp'] = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        uci_inputs['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80.0, max_value=200.0, value=120.0)
        uci_inputs['chol'] = st.number_input("Serum Cholesterol (mg/dl)", min_value=100.0, max_value=600.0, value=200.0)
        uci_inputs['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
        uci_inputs['restecg'] = st.selectbox("Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])

    with col4:
        uci_inputs['thalch'] = st.number_input("Maximum Heart Rate Achieved", min_value=60.0, max_value=220.0, value=150.0)
        uci_inputs['exang'] = st.selectbox("Exercise Induced Angina", [True, False])
        uci_inputs['oldpeak'] = st.number_input("ST Depression Induced by Exercise", min_value=-2.0, max_value=10.0, value=0.0, step=0.1)
        uci_inputs['slope'] = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
        uci_inputs['ca'] = st.selectbox("Major Vessels Colored by Flourosopy (ca)", [0.0, 1.0, 2.0, 3.0])
        uci_inputs['thal'] = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    if st.button("Run Clinical Diagnostics", key="btn_uci"):
        input_df = pd.DataFrame([uci_inputs])
        prediction = uci_model.predict(input_df)[0]
        prob = uci_model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"🛑 **Positive for Heart Disease** (Confidence: {prob:.1%}). Immediate medical consultation advised.")
        else:
            st.success(f"✅ **Negative for Heart Disease** (Confidence: {1-prob:.1%}).")
