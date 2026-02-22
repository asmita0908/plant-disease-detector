import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requestsimport streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json
import requests
import os

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image and the AI model will detect the disease.")

# HuggingFace model url
MODEL_URL = "https://huggingface.co/09asmita/plant-disease-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"

# Download model once
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading AI model... (first time only ⏳)")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

# class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    disease_name = labels[index]

    st.success(f"Prediction: {disease_name}")
import os

st.set_page_config(page_title="Plant Disease Detector")

MODEL_URL = "https://huggingface.co/09asmita/plant-disease-model/resolve/main/model.h5"
MODEL_PATH = "model.h5"


# ---------------- DOWNLOAD MODEL SAFELY ----------------
def download_model():

    # agar corrupt file ho to delete
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        if size < 1000000:   # 1MB se choti = broken
            os.remove(MODEL_PATH)

    # download only if not present
    if not os.path.exists(MODEL_PATH):

        st.write("⏳ Downloading AI model first time... please wait (1-3 min)")

        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    download_model()
    model = load_model(MODEL_PATH, compile=False)
    return model


model = load_my_model()

# -------- CLASS NAMES ----------
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

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image and AI will detect the disease.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    disease_name = CLASS_NAMES[index]
    confidence = np.max(prediction)*100

    st.success(f"Prediction: {disease_name}")
    st.info(f"Confidence: {confidence:.2f}%")

