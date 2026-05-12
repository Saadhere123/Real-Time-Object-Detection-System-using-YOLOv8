import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

st.title("YOLOv8 Object Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = model(image_np)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, channels="BGR")