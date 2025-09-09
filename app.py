import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
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
    img = cv2.resize(img, (224, 224))  # adjust to your input size
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
st.title("🐾 Animal Classifier (Auto TFLite)")
st.write("Upload an image. First run will convert model to TensorFlow Lite.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    processed = preprocess_image(img_array)
    preds = predict(processed)

    label = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.markdown(f"### 🐾 Prediction: **{label}** ({confidence:.2f})")
