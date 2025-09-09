import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Dummy class labels
CLASS_NAMES = ["cat", "dog", "lion", "elephant"]

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Fake prediction (random label)
        idx = np.random.randint(0, len(CLASS_NAMES))
        label = CLASS_NAMES[idx]

        # Draw prediction
        cv2.putText(img, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        return img

st.title("🐾 Webcam Test App")
st.write("Testing Streamlit + WebRTC + OpenCV")

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
