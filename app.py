import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

st.set_page_config(page_title="Camera Input Animal Detector")

# --- load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_model.h5")

model = load_model()

CLASS_NAMES = ["cat", "dog", "lion", "elephant"]  # change to your classes
INPUT_SIZE = (224, 224)  # change to your model input size

st.title("🐾 Camera Input — Animal Detection")
st.write("Take a photo with your camera. Predictions will appear below.")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Captured photo", use_column_width=True)

    # preprocess
    img_np = np.array(img)
    resized = cv2.resize(img_np, INPUT_SIZE)
    inp = resized.astype("float32") / 255.0
    inp = np.expand_dims(inp, axis=0)

    preds = model.predict(inp)[0]
    idx = int(preds.argmax())
    conf = float(preds.max())

    st.markdown(f"**Prediction:** {CLASS_NAMES[idx]}  \n**Confidence:** {conf:.2f}")
