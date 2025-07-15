import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(page_title="Car Damage Detection", layout="centered")

st.title("🚗 Car Damage Detection using YOLOv8")
st.markdown(
    """
    Upload a car image, and this app will detect whether the car is **damaged** or **whole**, 
    using a trained YOLOv8 model.
    """
)

# Load YOLOv8 model (ensure best.pt is in the same folder)
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Replace with the full path if needed
    if not os.path.exists(model_path):
        st.error("Model file not found. Please place 'best.pt' in the same folder as app.py.")
        return None
    return YOLO(model_path)

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    # Save to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.image(tmp_path, caption="Uploaded Image", use_column_width=True)

    # Inference
    with st.spinner("Analyzing image..."):
        results = model.predict(source=tmp_path, imgsz=640, conf=0.25)

    # Get results
    res_img = results[0].plot()
    st.image(res_img, caption="Prediction Result", use_column_width=True)

    # Display detected classes
    st.markdown("### Detected Objects:")
    boxes = results[0].boxes
    names = results[0].names
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"**{names[cls_id]}** - Confidence: `{conf:.2f}`")

    # Optional: Show coordinates
    if st.checkbox("Show bounding box coordinates"):
        for i, box in enumerate(boxes.xyxy):
            st.write(f"Box {i+1}: {box.tolist()}")
