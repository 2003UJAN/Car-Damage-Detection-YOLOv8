import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.title("🚗 Car Damage Detection with YOLOv8")
st.write("Upload a car image to detect damage or classify as whole.")

# Upload image
uploaded_file = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load model
    model = YOLO("best.pt")  # Replace with path to your trained model

    # Run prediction
    results = model.predict(source=tmp_path, imgsz=640)
    result_img = results[0].plot()

    # Display result
    st.image(result_img, caption="Prediction", use_column_width=True)

    # Show detected classes
    boxes = results[0].boxes
    classes = results[0].names
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        st.markdown(f"**Detected:** {classes[cls]} with confidence {box.conf[0]:.2f}")

