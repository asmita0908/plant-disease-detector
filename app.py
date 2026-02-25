import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import gdown
import os
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
# ---------------- DOWNLOAD MODEL FROM DRIVE ----------------

MODEL_PATH = "plant_disease_model.h5"
url = "https://drive.google.com/uc?id=1sC_EKoACZPOeq0rDPtyDyxj6DJns6RZZ"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... Please wait (first run only)"):
        gdown.download(url, MODEL_PATH, quiet=False)

# ---------------- LOAD CLASS MAPPING ----------------

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ---------------- PAGE ----------------

st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
    return model

model = load_my_model()

# ---------------- IMAGE PREPROCESS ----------------

def preprocess_image(img):
    img = img.resize((224, 224))   # same size jo training me use kiya
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# ---------------- UPLOAD ----------------

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)


img = preprocess_image(image)

# prediction
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

result = idx_to_class[predicted_class]
result = result.replace("___", " - ").replace("_", " ")

st.success(f"🧪 Prediction: {result}")


