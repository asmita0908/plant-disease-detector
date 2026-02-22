import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tfimport streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

MODEL_URL = "https://huggingface.co/09asmita/plant-disease-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI model... Please wait ⏳")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_my_model()

st.title("🌿 Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

class_names = ["Healthy", "Powdery Mildew", "Rust", "Leaf Spot"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
import requests
import os

MODEL_URL = "https://huggingface.co/09asmita/plant-disease-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI model... Please wait ⏳")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_my_model()

st.title("🌿 Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

class_names = ["Healthy", "Powdery Mildew", "Rust", "Leaf Spot"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
