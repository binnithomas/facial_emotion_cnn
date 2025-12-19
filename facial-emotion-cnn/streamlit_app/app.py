import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("../saved_models/emotion_cnn.h5")
classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

st.title("Facial Emotion Recognition")

uploaded = st.file_uploader("Upload an Image", type=["jpg","png"])

if uploaded:
    img = Image.open(uploaded).convert('L').resize((48,48))
    img_array = np.array(img).reshape(1,48,48,1) / 255.0

    pred = model.predict(img_array)
    st.image(uploaded)
    st.write("Prediction:", classes[np.argmax(pred)])
