import streamlit as st
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="🧠 Tumor Detector", layout="wide")

st.title("🧠 Clasificador de Tumores Cerebrales")
st.write("Demo funcional del modelo de clasificación")

uploaded_file = st.file_uploader("Sube una imagen MRI", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    st.subheader("🔍 Resultado:")

    clases = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    pred = random.choice(clases)

    st.success(f"Predicción: {pred}")

    st.info("⚠️ Nota: Esta es una demo sin modelo cargado por limitaciones de despliegue.")

st.markdown("---")
st.markdown("© 2026 Nadia Ayala")
