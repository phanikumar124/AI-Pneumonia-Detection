import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("model/pneumonia_model.keras")

# Image path (replace with your test image)
img_path = "dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("Prediction: PNEUMONIA")
    print(f"Confidence: {prediction*100:.2f}%")
else:
    print("Prediction: NORMAL")
    print(f"Confidence: {(1-prediction)*100:.2f}%")
