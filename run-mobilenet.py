import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("models", "mobilenet.keras")

# Načtení modelu
print("Načítám MOBILENET model...")
model = load_model(MODEL_PATH)
print("Model úspěšně načten.")

# Názvy emocí
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Načtení vestavěného detektoru obličejů z OpenCV (Haar Cascade)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Otevření webkamery (0 je výchozí integrovaná kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Chyba: Nepodařilo se otevřít webkameru.")
    exit()

print("Spouštím detekci. Stiskni 'q' v okně videa pro ukončení.")

while True:
    # Načtení snímku z kamery
    ret, frame = cap.read()

    if not ret:
        print("Chyba: Selhal příjem snímku z webkamery.")
        break

    # Převod snímku do odstínů šedi pro detektor
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekce obličejů v obraze
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Zpracujeme pouze první detekovaný obličej
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Vykreslení modrého obdélníku kolem obličeje
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Oříznutí obličeje z šedotónovaného snímku
        face = gray[y:y+h, x:x+w]

        # ÚPRAVA pro 96x96 model: Změna velikosti výřezu na 96 px
        face = cv2.resize(face, (96, 96))

        # Převod na float32 bez dělení 255.0
        # Interní preprocess_input vrstva v modelu očekává surové hodnoty 0 až 255.
        face = face.astype("float32")

        # Přidání rozměrů pro batch size a kanál -> vznikne tvar (1, 96, 96, 1)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Předpověď emoce (verbose=0 vypne logování do konzole při každém snímku)
        prediction = model.predict(face, verbose=0)

        # Získání indexu nejpravděpodobnější emoce a její úspěšnosti (confidence)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = np.max(prediction)

        # Příprava textu (např. "Happy (0.85)")
        text = f"{emotion} ({confidence:.2f})"

        # Vykreslení zeleného textu těsně nad obdélník obličeje
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Zobrazení výsledného obrazu v okně
    cv2.imshow("Emotion Detection (MobileNetV2 96x96)", frame)

    # Ukončení smyčky, pokud uživatel stiskne klávesu 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Vyčištění paměti a zavření oken
cap.release()
cv2.destroyAllWindows()
print("Aplikace úspěšně ukončena.")