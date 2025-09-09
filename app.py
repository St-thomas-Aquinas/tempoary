import streamlit as st
import av 
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------
# Load TensorFlow model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")  # path to your model
    return model

model = load_model()

# Class labels (change to your animal classes)
CLASS_NAMES = ["cat", "dog", "lion", "elephant"]

# -------------------------------
# Video Transformer
# -------------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize for prediction
        resized = cv2.resize(img, (224, 224))
        resized = np.expand_dims(resized, axis=0) / 255.0

        # Predict
        preds = model.predict(resized)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        conf = float(np.max(preds))

        # Draw prediction on frame
        cv2.putText(img, f"{label} ({conf:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🐾 Real-Time Animal Detection (Webcam)")
st.write("Uses TensorFlow + Streamlit WebRTC to classify animals in real time.")

webrtc_streamer(
    key="animal-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
