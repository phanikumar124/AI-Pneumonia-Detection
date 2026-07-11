import tensorflow as tf

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/chest_xray/train",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary"
)

print("Classes:", train_dataset.class_names)