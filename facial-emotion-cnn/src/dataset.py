import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

def emo_data(path="/facial-emotion-cnn/data/CNN/train"):
    df = pd.read_csv(path)
    images = []

    for pixel_seq in df["pixels"]:
        img = np.array(pixel_seq.split(), dtype="float32").reshape(48,48,1)
        images.append(img)

    X = np.array(images) / 255.0
    y = tf.keras.utils.to_categorical(df["emotion"], num_classes=7)

    return train_test_split(X, y, test_size=0.2, random_state=42)
