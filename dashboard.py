# dashboard.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from inference import preprocess_image, predict_image

st.title("Smart Factory Recycling Dashboard")
st.write("Upload an image from the production line to classify waste materials.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Save the uploaded image temporarily
    temp_image_path = "temp.jpg"
    cv2.imwrite(temp_image_path, image)
    
    # Run inference
    label, confidence = predict_image(temp_image_path)
    
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
