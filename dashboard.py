import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
# model = joblib.load(r'C:\Users\ASUS\Documents\miniprojek\diabetes\random_forest_diabetes.pkl')
from pathlib import Path
model_path = Path("C:/Users/ASUS/Documents/miniprojek/diabetes/random_forest_diabetes.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# Sidebar
st.sidebar.title("📌 Info Aplikasi")
st.sidebar.markdown("""
Aplikasi ini memprediksi risiko diabetes berdasarkan data medis pasien.
- Model: Random Forest
- Akurasi: 78% (contoh)
- Dibuat untuk portofolio
""")

st.title("🩺 Diabetes Risk Predictor")
st.write("Masukkan data pasien untuk memprediksi kemungkinan terkena diabetes.")

# Layout: 2 kolom
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("🤰 Kehamilan", 0, 20, 1)
    glucose = st.slider("🧪 Glukosa", 50, 200, 120)
    blood_pressure = st.slider("💓 Tekanan Darah", 40, 120, 70)
    skin_thickness = st.slider("📏 Ketebalan Kulit", 0, 100, 20)

with col2:
    insulin = st.slider("💉 Insulin", 0, 600, 79)
    bmi = st.slider("⚖️ BMI", 10.0, 70.0, 25.0)
    dpf = st.slider("📊 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("🎂 Usia", 10, 100, 33)

# Validasi input
if bmi < 13:
    st.warning("⚠️ Nilai BMI sangat rendah. Cek kembali input.")

# Prediksi tunggal
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Output prediksi
st.subheader("🔍 Hasil Prediksi:")
if prediction == 1:
    st.error(f"Hasil: Berisiko diabetes (Probabilitas: {probability:.2f})")
else:
    st.success(f"Hasil: Tidak berisiko diabetes (Probabilitas: {probability:.2f})")

# Gauge risiko (progress bar)
st.progress(min(int(probability * 100), 100))

# Rekomendasi kesehatan
st.markdown("### 💡 Rekomendasi:")
if prediction == 1:
    st.markdown("- Konsultasikan dengan dokter.")
    st.markdown("- Jaga pola makan dan aktivitas fisik.")
else:
    st.markdown("- Tetap pertahankan gaya hidup sehat.")

# Upload CSV untuk prediksi batch
st.markdown("---")
st.markdown("### 📁 Prediksi Banyak Pasien")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    pred_batch = model.predict(df)
    df['Prediksi'] = ["Risiko" if x == 1 else "Tidak Risiko" for x in pred_batch]
    st.success("Prediksi selesai!")
    st.dataframe(df)

# Simulasi gaya hidup (misal: turunkan BMI)
with st.expander("🔄 Simulasi Perubahan Gaya Hidup"):
    simulasi_bmi = st.slider("Turunkan BMI", 10.0, 70.0, bmi)
    input_simulasi = input_data.copy()
    input_simulasi[0][5] = simulasi_bmi
    simulasi_pred = model.predict(input_simulasi)[0]
    st.write("Jika BMI diubah menjadi:", simulasi_bmi)
    if simulasi_pred == 1:
        st.error("Masih berisiko diabetes.")
    else:
        st.success("Tidak berisiko diabetes dengan perubahan ini.")

# Optional: SHAP interpretation (jika SHAP & explainer sudah disiapkan)
# from shap import TreeExplainer, force_plot
# explainer = TreeExplainer(model)
# shap_values = explainer.shap_values(input_data)
# st_shap(force_plot(explainer.expected_value[1], shap_values[1], input_data))
