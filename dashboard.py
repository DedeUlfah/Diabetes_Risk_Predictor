import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Diabetes Risk Predictor")
st.write("Masukkan data pasien untuk memprediksi kemungkinan terkena diabetes.")

# input from
glucose = st.slider("Glucose", 50, 200, 100)
blood_pressure = st.slider("Blood Pressure", 40, 120, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 600, 80)
bmi = st.slider("BMI", 10.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 30)
pregnancies = st.slider("Pregnancies", 0, 20, 1)

input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# predict button
if st.button("Predict"):
    loaded_model = joblib.load("C:/Users/ASUS/Documents/miniprojek/diabetes/random_forest_diabetes.pkl")
    prediction = loaded_model.predict(input_data)[0]
    if prediction == 1:
        st.error("Hasil: Beresiko diabetes")
    else:
        st.success("Hasil: Tidak beresiko diabetes")