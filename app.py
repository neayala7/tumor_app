import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="🧠 Brain Tumor Detector",
    layout="wide"
)

# -------------------------
# ESTILO
# -------------------------
st.markdown("""
<style>
.main-title {
    font-size: 50px;
    text-align: center;
    color: #00BFFF;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
.result-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<p class="main-title">🧠 Brain Tumor AI Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Clasificación automática con Deep Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_my_model():
    return load_model("model_brain_tumor.keras", compile=False)

model = load_my_model()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------
# LAYOUT
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Subir imagen")
    uploaded_file = st.file_uploader("Carga una imagen MRI", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

with col2:
    st.subheader("🤖 Resultado")

    if uploaded_file:
        img = image.resize((224, 224))
        img = np.array(img)

        if img.shape[-1] == 4:
            img = img[:, :, :3]

        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        clase = class_names[np.argmax(pred)]
        conf = np.max(pred)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### 🧾 {clase.upper()}")
        st.write(f"Confianza: {conf:.2%}")
        st.progress(float(conf))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📊 Probabilidades")
        for i, c in enumerate(class_names):
            st.write(f"{c}: {pred[0][i]:.2%}")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📊 Modelo")

st.sidebar.write("""
✔ CNN / Transfer Learning  
✔ 4 clases  
✔ Accuracy ~86%  
""")

st.sidebar.write("🔍 Mejora:")
st.sidebar.write("Glioma vs Meningioma")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:gray'>
🧠 Proyecto de Ciencia de Datos<br>
© 2026 Nadia Ayala
</p>
""", unsafe_allow_html=True)
