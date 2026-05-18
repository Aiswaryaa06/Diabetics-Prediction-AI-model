import streamlit as st
import joblib
import numpy as np
import base64
import os

# Set page config
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Transparent header */
        .stApp > header {{
            background-color: transparent !important;
        }}
        
        /* Glassmorphism effect for the main container */
        .block-container {{
            background: rgba(10, 20, 35, 0.65) !important; /* Deep modern dark blue overlay */
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 20px !important;
            padding: 3rem !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
            margin-top: 40px !important;
            margin-bottom: 40px !important;
            max-width: 850px !important;
        }}

        /* Enhance text readability and aesthetics */
        h1, h2, h3, p, span, label {{
            color: #E2E8F0 !important;
            font-family: 'Inter', sans-serif !important;
        }}
        h1 {{
            font-weight: 800 !important;
            color: #ffffff !important;
            text-align: center;
            margin-bottom: 0.5rem !important;
        }}
        
        /* Style the input fields */
        div[data-testid="stNumberInput"] > div > div > input {{
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }}
        div[data-testid="stNumberInput"] > div > div > input:focus {{
            border: 1px solid #00C6FF !important;
            box-shadow: 0 0 10px rgba(0, 198, 255, 0.3) !important;
        }}

        /* Style the Predict Button */
        div.stButton > button {{
            background: linear-gradient(135deg, #0052D4, #4364F7, #6FB1FC) !important;
            color: white !important;
            border: none !important;
            padding: 12px 30px !important;
            border-radius: 50px !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(67, 100, 247, 0.4) !important;
            margin-top: 25px !important;
        }}
        div.stButton > button:hover {{
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(67, 100, 247, 0.6) !important;
        }}
        
        /* Adjust column gap */
        [data-testid="column"] {{
            padding: 0 1rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

# Add background image
add_bg_from_local('background.jpg')

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# App Title
st.title("🩺 Diabetes Prediction System")
st.write("""
Enter the medical parameters below to predict whether a person is likely to have diabetes.
""")

# Input Form
st.header("Patient Data")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

with col2:
    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=79.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

# Prediction logic
if st.button("Predict Diabetes Risk", type="primary"):
    # Create input array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    try:
        # Scale the input
        input_data_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_data_scaled)
        
        st.subheader("Result:")
        if prediction[0] == 1:
            st.error("🚨 The person is likely to have Diabetes.")
        else:
            st.success("✅ The person is NOT likely to have Diabetes.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Note: This is a Machine Learning model prediction and should not replace professional medical advice.")
