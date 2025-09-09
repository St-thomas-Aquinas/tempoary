import streamlit as st
import numpy as np
import cv2
import tensorflow.lite as tflite
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ----------------------
# Load TFLite model
# ----------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="animal_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🔹 Replace with your dataset classes
CLASS_NAMES = ["cat", "dog", "elephant", "lion", "tiger", "zebra"]

# ----------------------
# Preprocessing
# ----------------------
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))   # adjust to your model input size
    img = img / 255.0
    img = np.expand_dims(img.astype(np.float32), axis=0)
    return img

# ----------------------
# Predict
# ----------------------
def predict(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds

# ----------------------
# Video Transformer
# ----------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Predict
        processed = preprocess_image(rgb)
        preds = predict(processed)
        label = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds)

        # Overlay label
        cv2.putText(img, f"{label} ({confidence:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# ----------------------
# Streamlit UI
# ----------------------
st.title("🐾 Real-Time Animal Classifier (TFLite + WebRTC, 128x128)")
st.write("Allow camera access to see predictions in real-time.")

webrtc_streamer(key="animal-demo", video_transformer_factory=VideoTransformer)
