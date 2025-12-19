from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import load_fer2013

X_train, X_test, y_train, y_test = load_fer2013()

model = load_model("saved_models/emotion_cnn.h5")

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True)
plt.savefig("results/confusion_matrix.png")
