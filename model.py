import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

def load_model():
    model = tf.keras.models.load_model("model.keras")  # Ensure "model.h5" is in the same directory
    return model

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return image_array

def predict_digit(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will recognize it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    model = load_model()
    predicted_digit = predict_digit(model, image)
    
    st.write(f"**Predicted Digit:** {predicted_digit}")
