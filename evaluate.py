import tensorflow as tf

# Paths
test_dir = "dataset/chest_xray/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Load trained model
model = tf.keras.models.load_model("model/pneumonia_model.keras")

# Evaluate
loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
