import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------- PATHS ----------
MODEL_PATH = "plant_disease_model.keras"
TEST_DIR = "dataset/test"   # <-- yaha tumhara test folder hona chahiye

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --------- LOAD MODEL ----------
print("Loading model...")
model = load_model(MODEL_PATH)

# --------- LOAD TEST DATA ----------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --------- PREDICTIONS ----------
print("Predicting...")
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# --------- CLASSIFICATION REPORT ----------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# --------- CONFUSION MATRIX ----------
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# --------- ROC AUC ----------
# Convert labels to one-hot
y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# --------- PLOT ROC CURVE ----------
plt.figure(figsize=(8,6))
for i in range(len(class_labels)):
    plt.plot(fpr[i], tpr[i], label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

print("\nROC curve saved as roc_curve.png")
