import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# =========================
# Configuration
# =========================
TRAIN_DIR = os.path.join("dataset", "train")
VAL_DIR = os.path.join("dataset", "test")
IMG_SIZE = (96, 96)
BATCH_SIZE = 128
EPOCHS = 25
MODEL_SAVE_PATH = os.path.join("models", "mobilenet.keras")
HISTORY_SAVE_PATH = os.path.join("models", "mobilenet-history.json")
SEED = 42

tf.random.set_seed(SEED)

# =========================
# Load datasets
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# =========================
# Optimize pipeline
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# Data augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
], name="data_augmentation")

# =========================
# Build MobileNetV2 model
# =========================
def build_mobilenetv2_fer(input_shape=(96, 96, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)

    # grayscale -> RGB
    x = layers.Concatenate(name="grayscale_to_rgb")([x, x, x])

    # MobileNetV2 preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="FER_MobileNetV2"
    )

    return model, base_model

model, base_model = build_mobilenetv2_fer(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1),
    num_classes=num_classes
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Callbacks
# =========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

# =========================
# Train top classifier first
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# Optional fine-tuning
# =========================
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=callbacks
)

# =========================
# Combine histories
# =========================
history_dict = {}
for key in history.history.keys():
    history_dict[key] = history.history[key] + history_fine.history[key]

# =========================
# Save final model and history
# =========================
model.save(MODEL_SAVE_PATH)

with open(HISTORY_SAVE_PATH, "w") as f:
    json.dump(history_dict, f)

print(f"Best model saved to: {MODEL_SAVE_PATH}")
print(f"History saved to: {HISTORY_SAVE_PATH}")

# =========================
# Plot curves
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_dict["loss"], label="Train Loss")
plt.plot(history_dict["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_dict["accuracy"], label="Train Accuracy")
plt.plot(history_dict["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()