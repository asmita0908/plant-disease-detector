from pyexpat import model

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# ---------------- DOWNLOAD MODEL ----------------

FILE_ID = "17ord-IG_5zYhRF5Y2L48G_pLSUJIVsks"
MODEL_PATH = "plant_disease_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... first time only ⏳"):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------- LOAD CLASSES ----------------

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ---------------- UI ----------------

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# -------- IMAGE PREPROCESS --------

def preprocess_image(image):
    image = image.resize((224, 224))
    img = np.array(image, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------- PREDICTION --------

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)


img = preprocess_image(image)
prediction = model.predict(img)
class_id = np.argmax(prediction)
confidence = np.max(prediction)

predicted_label = labels[int(class_id)]

st.subheader("Prediction:")
st.success(predicted_label.replace("_", " "))

st.subheader("Confidence:")
st.write(f"{confidence*100:.2f}%")

