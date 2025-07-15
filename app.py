import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Streamlit page config
st.set_page_config(page_title="Car Damage Detection", layout="centered")

# Title and Description
st.title("🚗 Car Damage Detection using YOLOv8")
st.markdown(
    """
    This app uses a custom-trained [YOLOv8](https://github.com/ultralytics/ultralytics) model to detect 
    whether a car is **damaged** or **whole**. Upload an image of a vehicle and let AI do the inspection.
    """
)

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Must be in same directory as app.py
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please upload your trained best.pt in the project directory.")
        return None
    return YOLO(model_path)

model = load_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file for YOLOv8
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Inference
    with st.spinner("🔍 Detecting damage..."):
        results = model.predict(source=tmp_path, imgsz=640, conf=0.25)

    # Render result
    res_img = results[0].plot()
    st.image(res_img, caption="🔎 Detection Result", use_column_width=True)

    # Detection info
    st.markdown("### 📋 Detected Classes:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results[0].names[cls_id]
        st.write(f"• **{label}** — Confidence: `{conf:.2f}`")

    # Optional: Bounding box data
    if st.checkbox("📐 Show Bounding Box Coordinates"):
        for i, box in enumerate(results[0].boxes.xyxy):
            st.write(f"Box {i+1}: `{box.tolist()}`")

