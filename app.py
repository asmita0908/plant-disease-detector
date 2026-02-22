import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

MODEL_URL = "https://huggingface.co/09asmita/plant-disease-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"


@st.cache_resource
def load_my_model():

    # Download model only once
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI Model... (First time only, wait 1-2 min) ⏳")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    # ⭐ IMPORTANT FIX (Keras 3 compatible)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )
    return model


model = load_my_model()

st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and AI will detect the disease.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

class_names = ["Healthy", "Powdery", "Rust", "Leaf Spot"]

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
