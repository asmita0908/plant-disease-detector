import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# ---------------- DOWNLOAD MODEL ----------------
MODEL_ID = "1FcseqnC56K9Nf7pO716--0BCE4YmBDjE"
MODEL_PATH = "plant_disease_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... (first time only, 2-3 min)"):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- LOAD CLASSES ----------------
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v:k for k,v in class_indices.items()}

# ---------------- UI ----------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

def preprocess_image(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess_image(image)
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction:")
    st.success(f"{labels[class_id]}")

    st.subheader("Confidence:")
    st.write(f"{confidence*100:.2f}%")
