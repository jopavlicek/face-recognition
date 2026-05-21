import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("models", "custom.keras")

# Načtení modelu
print("Načítám CUSTOM model...")
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

# Otevření webkamery
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Chyba: Nepodařilo se otevřekt webkameru.")
    exit()

print("Spouštím detekci. Stiskni 'q' v okně videa pro ukončení.")

while True:
    # Načtení snímku z kamery
    ret, frame = cap.read()

    if not ret:
        print("Chyba: Selhal příjem snímku z webkamery.")
        break

    # Převod snímku do odstínů šedi pro detektor a model
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

        # Cílová velikost pro tvůj custom model: 48x48 pixelů
        face = cv2.resize(face, (48, 48))

        # POZOR ZDE: Pro custom model musíme dělit 255.0, 
        # protože model očekává hodnoty v rozsahu 0.0 až 1.0!
        face = face.astype("float32") / 255.0

        # Přidání rozměrů pro batch size a kanál -> vznikne tvar (1, 48, 48, 1)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Předpověď emoce
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
    cv2.imshow("Emotion Detection (Custom CNN 48x48)", frame)

    # Ukončení smyčky, pokud uživatel stiskne klávesu 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Vyčištění paměti a zavření oken
cap.release()
cv2.destroyAllWindows()
print("Aplikace úspěšně ukončena.")