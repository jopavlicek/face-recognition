import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Path config
MODEL_PATH = os.path.join("models", "custom.keras")
TRAIN_PATH = os.path.join("dataset", "train")
TEST_PATH = os.path.join("dataset", "test")

# Seed setting for reproduction
SEED = 42
tf.random.set_seed(SEED)

# Load datasets
train_dataset = image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    shuffle=True,
    seed=SEED
)

class_names = train_dataset.class_names
print("Classes:", class_names)

validation_dataset = image_dataset_from_directory(
    TEST_PATH,
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    shuffle=False
)

# Data pipeline optimization
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Architektura modelu
model = Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),  # Normalizace probíhá bezpečně až tady

    # 1. Block
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 2. Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # 3. Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # 4. Block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    # Classification
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
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
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

# Trénování
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=40,
    callbacks=callbacks
)

# Validace
loss, accuracy = model.evaluate(validation_dataset)
print(f"\nKonečná nejlepší validační přesnost: {accuracy:.4f}")
print("Nejlepší model je uložen.")