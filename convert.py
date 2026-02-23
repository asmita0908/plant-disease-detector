from tensorflow.keras.models import load_model

print("Loading old model...")
model = load_model("model.h5", compile=False)

print("Saving new keras model...")
model.save("plant_disease_model.keras")

print("DONE ✅ Model converted successfully")