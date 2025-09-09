import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# ----------------------
# Convert Keras model -> TFLite if not already
# ----------------------
@st.cache_resource
def load_tflite_model():
    if not os.path.exists("animal_model.tflite"):
        st.write("Converting Keras model to TFLite...")

        # Load keras model
        model = tf.keras.models.load_model("best_model.h5")

        # Convert
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save
        with open("animal_model.tflite", "wb") as f:
            f.write(tflite_model)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="animal_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🔹 Replace with your dataset classes
CLASS_NAMES = ["cat", "dog", "elephant", "lion", "tiger", "zebra"]

# ----------------------
# Preprocess function
# ----------------------
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))   # ✅ match your model input
    img = img / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)
    return img

# ----------------------
# Predict function
# ----------------------
def predict(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds

# ----------------------
# Streamlit UI
# ----------------------
st.title("🐾 Real-Time Animal Classifier (128x128, Auto TFLite)")
st.write("Press the checkbox to start your camera and see predictions in real-time.")

start = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if start:
    cap = cv2.VideoCapture(0)  # open default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("⚠️ Cannot access camera")
            break

        # Convert frame (BGR → RGB for Streamlit)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict
        processed = preprocess_image(rgb)
        preds = predict(processed)
        label = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds)

        # Overlay label
        cv2.putText(rgb, f"{label} ({confidence:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show frame
        FRAME_WINDOW.image(rgb)

    cap.release()
