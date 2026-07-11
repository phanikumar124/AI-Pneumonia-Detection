import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ==========================
# Dataset Paths
# ==========================
train_dir = "dataset/chest_xray/train"
val_dir = "dataset/chest_xray/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================
# Load Dataset
# ==========================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Classes:", train_dataset.class_names)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# ==========================
# Optimize Dataset
# ==========================
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .cache()
    .shuffle(1000)
    .prefetch(AUTOTUNE)
)

val_dataset = (
    val_dataset
    .cache()
    .prefetch(AUTOTUNE)
)

# ==========================
# Build Model
# ==========================
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Callbacks
# ==========================
callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "model/pneumonia_model.keras",
        save_best_only=True
    )

]

# ==========================
# Train
# ==========================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=callbacks
)

# ==========================
# Save History
# ==========================
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "model/history.csv",
    index=False
)

print("\nTraining Completed Successfully!")
print("Model Saved Successfully!")
print("History Saved Successfully!")