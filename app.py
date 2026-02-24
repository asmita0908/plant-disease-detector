from pyexpat import model
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# ---------------- PAGE ----------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect disease")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("plant_disease_model.h5")
    return model

model = load_my_model()

# ---------------- LOAD CLASSES ----------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# reverse mapping (number → class name)
idx_to_class = {v: k for k, v in class_indices.items()}

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img):
    img = img.resize((224, 224))        # IMPORTANT (same as training)
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

# ---------------- ONLY RUN AFTER UPLOAD ----------------
if uploaded_file is not None:

    # open image
    image = Image.open(uploaded_file).convert("RGB")

    # show image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img = preprocess_image(image)

    # prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    result = idx_to_class[predicted_class]

    st.success(f"🧪 Prediction: {result}")
