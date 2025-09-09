import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_model.h5")
    return model

model = load_model()

# Edit this to match your dataset
CLASS_NAMES = ["cat", "dog", "elephant", "lion", "tiger", "zebra"]

# ----------------------
# Preprocessing
# ----------------------
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))   # adjust to model input
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ----------------------
# Streamlit App
# ----------------------
st.title("🐾 Animal Classifier (Streamlit Cloud)")
st.write("Upload an image and the model will identify the animal.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to array
    img_array = np.array(image)

    # Preprocess + Predict
    processed = preprocess_image(img_
