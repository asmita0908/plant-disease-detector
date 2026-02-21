import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image and the AI model will detect the disease.")

@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    prediction = model.predict(img_array)
    index = np.argmax(prediction)

    disease_name = labels[index]

    st.success(f"Prediction: {disease_name}")
