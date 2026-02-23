import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse dictionary
labels = dict((v, k) for k, v in class_indices.items())

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "dataset/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Predictions
pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

# ---------------- F1 SCORE ----------------
num_classes = len(labels)
f1_scores = []

for i in range(num_classes):
    tp = np.sum((y_true == i) & (y_pred == i))
    fp = np.sum((y_true != i) & (y_pred == i))
    fn = np.sum((y_true == i) & (y_pred != i))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    f1_scores.append(f1)

print("Average F1 Score:", np.mean(f1_scores))

# ---------------- ROC & AUC ----------------
y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes)

plt.figure()

for i in range(num_classes):
    tpr = []
    fpr = []

    thresholds = np.linspace(0, 1, 50)

    for thresh in thresholds:
        pred_class = pred_probs[:, i] >= thresh

        tp = np.sum((y_true == i) & (pred_class))
        fp = np.sum((y_true != i) & (pred_class))
        fn = np.sum((y_true == i) & (~pred_class))
        tn = np.sum((y_true != i) & (~pred_class))

        tpr.append(tp / (tp + fn + 1e-7))
        fpr.append(fp / (fp + tn + 1e-7))

    plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.show()

print("ROC curve saved as roc_curve.png")
