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
IMG_SIZE = (128, 128)  # MobileNet trénuje na 224x224, zvětšení na 128x128 je lepší
BATCH_SIZE = 64
EPOCHS = 25
MODEL_SAVE_PATH = os.path.join("models", "mobilenet.keras")
HISTORY_SAVE_PATH = os.path.join("models", "mobilenet-history.json")
SEED = 42

tf.random.set_seed(SEED)

# Zajistíme, že složka pro modely existuje
os.makedirs("models", exist_ok=True)

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
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# =========================
# Build MobileNetV2 model
# =========================
def build_mobilenetv2_fer(input_shape=(128, 128, 1), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    # 1. Aplikace augmentace
    x = data_augmentation(inputs)

    # 2. Převod Grayscale na RGB (duplikace kanálů pro MobileNet)
    x = layers.Concatenate(name="grayscale_to_rgb")([x, x, x])

    # 3. Interní preprocessing vrstva pro MobileNetV2 (převod na rozsah -1 až 1)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base model stažený z Keras aplikací
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        include_top=False,
        weights="imagenet"
    )

    # V první fázi kompletně zmrazíme základní model
    base_model.trainable = False
    x = base_model(x, training=False)

    # Globální pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Silnější klasifikační hlava pro zachycení emocí
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="FER_MobileNetV2")
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
# Callbacks pro 1. FÁZI
# =========================
callbacks_phase1 = [
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
        patience=3,      # Reaguje relativně rychle při trénování hlavy
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]

# =========================
# FÁZE 1: Trénování nové hlavy
# =========================
print("\n--- FÁZE 1: Trénování nové klasifikační hlavy ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_phase1
)

# =========================
# FÁZE 2: Hluboký fine-tuning
# =========================
print("\n--- FÁZE 2: Odmrazování sítě a Fine-tuning ---")
base_model.trainable = True

# Odmrazíme vrstvy od 80. dál, ale PONECHÁME BATCHNORMALIZATION ZMRAZENÉ
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif base_model.layers.index(layer) < 80:
        layer.trainable = False

# NOVÉ ČISTÉ CALLBACKY PRO 2. FÁZI (Reset paměti a LR)
callbacks_phase2 = [
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
        patience=4,      # Vyšší patience – dáváme odmraženému modelu čas dýchat
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,      # Trochu delší trpělivost před ukončením
        restore_best_weights=True,
        verbose=1
    )
]

# Zde compile natvrdo nastaví čistých 1e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Logika pro bezpečné dynamické navázání epoch při zasazení EarlyStoppingu
start_epoch = len(history.epoch)
fine_tune_epochs = 15
total_epochs = start_epoch + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=start_epoch,
    callbacks=callbacks_phase2
)

# =========================
# Spojení historií z obou fází
# =========================
print("\n--- Ukládání výsledků a generování grafů ---")

history_dict = {}
# Spojíme metriky z první i druhé fáze dohromady
for key in history.history.keys():
    if key in history_fine.history:
        history_dict[key] = history.history[key] + history_fine.history[key]
    else:
        history_dict[key] = history.history[key]

# Pro jistotu uložíme finální model po dokončení fine-tuningu
model.save(MODEL_SAVE_PATH)

# Zápis historie do JSON souboru
with open(HISTORY_SAVE_PATH, "w") as f:
    json.dump(history_dict, f)

print(f"Finální model uložen do: {MODEL_SAVE_PATH}")
print(f"Historie trénování uložena do: {HISTORY_SAVE_PATH}")

# =========================
# Grafy úspěšnosti (Vizualizace obou fází)
# =========================
epochs_range = range(len(history_dict["accuracy"]))
split_point = len(history.epoch)  # Bod, kde skončila 1. fáze a začal fine-tuning

plt.figure(figsize=(14, 6))

# Graf pro Loss (Chybovost)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history_dict["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
plt.plot(epochs_range, history_dict["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
plt.axvline(x=split_point - 0.5, color="red", linestyle="--", label="Start Fine-tuningu")
plt.title("Profil chybovosti (Loss)")
plt.xlabel("Epocha")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

# Graf pro Accuracy (Přesnost)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history_dict["accuracy"], label="Train Accuracy", color="#2ca02c", linewidth=2)
plt.plot(epochs_range, history_dict["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=2)
plt.axvline(x=split_point - 0.5, color="red", linestyle="--", label="Start Fine-tuningu")
plt.title("Profil přesnosti (Accuracy)")
plt.xlabel("Epocha")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()