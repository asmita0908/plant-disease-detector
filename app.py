from pyexpat import model

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

st.set_page_config(page_title="Plant Disease Detection")

MODEL_ID = "1Lu752xt99Nbi5Lsh09hIg9nSJ0m-wREo"
MODEL_PATH = "plant_disease_model.h5"

# -------- Download model from Drive --------

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... first run only (2-3 minutes)"):
            url = f"https://drive.google.com/uc?id={MODEL_ID}"
gdown.download(url, MODEL_PATH, quiet=False)

download_model()

# -------- Load Model --------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# -------- Load labels --------

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# -------- UI --------

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# -------- Preprocess --------

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------- Prediction --------

if uploaded_file is not None:
 image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)


img = preprocess_image(image)
prediction = model.predict(img)

class_id = np.argmax(prediction)
confidence = np.max(prediction)

st.success(f"Disease: {labels[class_id]}")
st.write(f"Confidence: {confidence*100:.2f}%")

