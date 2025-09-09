import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load your trained TensorFlow model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_model.h5")  # Change to your model path
    return model

model = load_model()

# Class labels (update with your animal classes)
CLASS_NAMES = ["cat", "dog", "lion", "elephant"]

# -------------------------------
# Prediction function
# -------------------------------
def predict(img):
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = np.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    return CLASS_NAMES[idx], float(np.max(pred))

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🐾 Real-Time Animal Detection")

st.write("This app uses your camera feed and a TensorFlow model to detect animals in real time.")

# Start camera
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)  # Webcam

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not working")
        break

    # Convert color for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get prediction
    label, confidence = predict(frame_rgb)

    # Draw prediction on the frame
    cv2.putText(frame_rgb, f"{label} ({confidence:.2f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show video in streamlit
    FRAME_WINDOW.image(frame_rgb)

camera.release()
