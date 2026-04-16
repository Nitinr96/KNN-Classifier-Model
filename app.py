import streamlit as st
import pickle
import numpy as np

# Page configuration for a professional look
st.set_page_config(page_title="Health Diagnostic Pro", layout="wide")

# Custom CSS to center content and style the interface
st.markdown("""
    <style>
    .main {
        display: flex;
        justify-content: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007BFF;
        color: white;
    }
    .prediction-text {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Centering the layout using columns
empty_l, center_col, empty_r = st.columns([1, 2, 1])

with center_col:
    st.title("🏥 Health Diagnostic Assistant")
    st.write("Please fill in the details below to evaluate the health metrics.")
    st.divider()

    # Feature Inputs with Categorical Options
    # 1. Pregnancies (Example of keeping it numeric but user-friendly)
    pregnancies = st.slider("Number of Pregnancies", 0, 20, 0)

    # 2. Glucose (Standard input)
    glucose = st.number_input("Plasma Glucose Concentration", min_value=0, value=100)

    # 3. Blood Pressure
    blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, value=70)

    # 4. Skin Thickness
    skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, value=20)

    # 5. Insulin
    insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, value=80)

    # 6. BMI
    bmi = st.number_input("Body Mass Index (weight in kg/(height in m)^2)", min_value=0.0, value=25.0, format="%.1f")

    # 7. Diabetes Pedigree Function (Using a Categorical Proxy for better UX)
    # Since the model needs a float, we provide descriptions and map them to average values
    dpf_label = st.selectbox(
        "Family History of Diabetes (Pedigree Function)",
        options=["Low / No Family History", "Moderate / Some Family History", "High / Immediate Family History"],
        index=0
    )
    dpf_map = {"Low / No Family History": 0.2, "Moderate / Some Family History": 0.5, "High / Immediate Family History": 1.2}
    dpf = dpf_map[dpf_label]

    # 8. Age
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    st.divider()
    
    # Prediction Logic
    if st.button("Generate Diagnostic Report"):
        # Prepare features in the order: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        st.write("---")
        if prediction[0] == 1:
            st.error("### Analysis: Positive Result Identified")
            st.markdown("The model suggests a high probability of a positive diagnostic outcome. Please consult with a medical professional.")
        else:
            st.success("### Analysis: Negative Result Identified")
            st.markdown("The model suggests a low probability of a positive diagnostic outcome.")
    
    if prediction[0] == 1:
        st.error("The model predicts a positive result.")
    else:
        st.success("The model predicts a negative result.")
