from dataset import load_fer2013
from model import build_cnn
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = load_fer2013()

model = build_cnn()

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=64
)

model.save("saved_models/emotion_cnn.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.legend(['train', 'val'])
plt.savefig('results/accuracy.png')

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.legend(['train', 'val'])
plt.savefig('results/loss.png')
