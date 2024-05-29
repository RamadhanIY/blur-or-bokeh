import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

model = tf.keras.models.load_model('model.keras')

class_labels = ['Blur', 'Bokeh']

st.title('Blur or Bokeh Detector')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((225, 225))  
    image = np.expand_dims(image, axis=0) 

    # Make prediction
    prediction = model.predict(image)

    # Get the predicted class label
    if prediction >= 0.5:
        predicted_class = 'Bokeh'
    else:
        predicted_class = 'Blur'

    st.write(f'Prediction: {predicted_class} ({prediction[0][0]:.2f})')
