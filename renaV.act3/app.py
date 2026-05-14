import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# Cache the model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.track(
            img,
            persist=True,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}
)