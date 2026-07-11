import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🩺 AI Pneumonia Detection")

    st.markdown("---")

    st.write("""
### About

This application detects **Pneumonia** from Chest X-ray images using **Deep Learning (MobileNetV2)**.

**Framework**
- TensorFlow
- Streamlit

**Input Size**
- 224 × 224 RGB

**Classes**
- NORMAL
- PNEUMONIA
""")

    st.markdown("---")

    st.success("👨‍💻 Developed by\n\n**Phani Kumar**")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/pneumonia_model.keras")

model = load_model()

# -----------------------------
# Main Title
# -----------------------------
st.title("🩺 AI Pneumonia Detection System")

st.write(
    "Upload a Chest X-ray image and the AI model will predict whether it is **NORMAL** or **PNEUMONIA**."
)

st.markdown("---")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    with col2:

        img_resized = img.resize((224, 224))

        img_array = np.array(img_resized, dtype=np.float32)

        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 AI is analyzing the X-ray..."):
            prediction = float(model.predict(img_array, verbose=0)[0][0])

        st.subheader("Prediction Result")

        if prediction >= 0.5:

            confidence = prediction

            st.error("🦠 PNEUMONIA DETECTED")

        else:

            confidence = 1 - prediction

            st.success("✅ NORMAL")

        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )

        st.progress(float(confidence))

        st.write(f"Raw Prediction Score: {prediction:.6f}")

        st.markdown("---")

        st.subheader("🩺 AI Medical Report")

        if prediction >= 0.5:

            st.error("""
Possible signs of **Pneumonia** detected.

**Recommendation**
- Consult a radiologist.
- Perform further clinical examination.
- This result should not be considered a final medical diagnosis.
""")

        else:

            st.success("""
No obvious signs of **Pneumonia** detected.

**Recommendation**
- Lung appearance seems normal.
- Continue routine medical care if symptoms persist.
- This result should not replace professional medical advice.
""")

# -----------------------------
# Model Information
# -----------------------------
with st.expander("📋 Model Information"):

    st.write("""
**Model:** MobileNetV2

**Framework:** TensorFlow

**Transfer Learning:** Yes

**Input Size:** 224 × 224

**Output Classes**
- NORMAL
- PNEUMONIA

**Project Type**
Deep Learning based Chest X-ray Classification
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "⚠ This application is for educational purposes only. "
    "Always consult a qualified healthcare professional for diagnosis."
)