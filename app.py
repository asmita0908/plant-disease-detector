import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import os

# -------------------------------
# Download model from HuggingFace
# -------------------------------
MODEL_URL = "https://huggingface.co/09asmita/plant-disease-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"

@st.cache_resource
def load_my_model():
    # agar model already download hai to dubara nahi karega
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... please wait ⏳"):
            r = requests.get(MODEL_URL)
            open(MODEL_PATH, "wb").write(r.content)

    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

# -------------------------------
# Class Labels (dataset ke folder names)
# IMPORTANT: apne dataset ke exact names likhna
# -------------------------------
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and AI will detect the disease")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🧠 Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

