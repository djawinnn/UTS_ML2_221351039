import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model dan preprocessing tools
model = tf.lite.Interpreter(model_path="model_liver.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("🩺 Prediksi Penyakit Liver")
st.write("Silakan isi data pasien di bawah ini:")

# Form input
age = st.slider("Umur", 1, 100, 45)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])  # pilihan tetap
tb = st.number_input("Total Bilirubin", min_value=0.0, value=1.0)
db = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3)
alk_phos = st.number_input("Alkaline Phosphotase", min_value=0.0, value=210.0)
sgpt = st.number_input("Sgpt (ALT)", min_value=0.0, value=30.0)
sgot = st.number_input("Sgot (AST)", min_value=0.0, value=50.0)
total_protein = st.number_input("Total Protein", min_value=0.0, value=6.5)
albumin = st.number_input("Albumin", min_value=0.0, value=3.2)
ag_ratio = st.number_input("A/G Ratio", min_value=0.0, value=1.0)

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Manual encode gender
        gender_encoded = 1 if gender == "Male" else 0

        # Preprocess input
        data = np.array([[age, gender_encoded, tb, db, alk_phos, sgpt, sgot,
                          total_protein, albumin, ag_ratio]])

        data_scaled = scaler.transform(data).astype(np.float32)

        # Inference
        input_index = input_details[0]['index']
        model.set_tensor(input_index, data_scaled)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        prediction = (output_data[0][0] > 0.5).astype(int)

        # Tampilkan hasil
        if prediction == 1:
            st.error("❗ Pasien kemungkinan MENGIDAP penyakit liver.")
        else:
            st.success("✅ Pasien kemungkinan TIDAK mengidap penyakit liver.")
    except Exception as e:
        st.warning(f"Terjadi kesalahan saat memproses input: {e}")
