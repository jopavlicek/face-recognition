import cv2
import os
import numpy as np
from collections import deque
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

# Inicializace fronty pro vyhlazení výsledků (klouzavý průměr z 5 snímků)
prediction_history = deque(maxlen=5)

# Načtení vestavěného detektoru obličejů z OpenCV (Haar Cascade)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Otevření webkamery
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

        # 1. PADDING: Přidáme 15 % okraj kolem obličeje pro věrnější FER2013 výřez
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        
        # Kontrola, abychom neořezávali mimo rozměry obrazu (frame)
        img_h, img_w = gray.shape
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        # Vykreslení modrého obdélníku kolem detekovaného obličeje
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Oříznutí obličeje s přidaným paddingem
        face = gray[y1:y2, x1:x2]

        # Cílová velikost pro model: 48x48 pixelů
        face = cv2.resize(face, (48, 48))

        # Převedeme na float, ale UŽ NEDĚLÍME 255.0 (dělá to vrstva v modelu)
        face = face.astype("float32")

        # Přidání rozměrů pro batch size a kanál -> vznikne tvar (1, 48, 48, 1)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Předpověď emoce (vrátí pole pravděpodobností pro každou třídu)
        raw_prediction = model.predict(face, verbose=0)[0]
        
        # 2. VYHLAZENÍ: Uložíme předpověď do historie a spočítáme průměr
        prediction_history.append(raw_prediction)
        smoothed_prediction = np.mean(prediction_history, axis=0)

        # Získání indexu nejpravděpodobnější emoce a její úspěšnosti z vyhlazených dat
        emotion_index = np.argmax(smoothed_prediction)
        emotion = emotion_labels[emotion_index]
        confidence = smoothed_prediction[emotion_index]

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
    else:
        # Pokud model ztratí obličej, vymažeme historii, aby stará emoce "nedosluhovala" na novém člověku
        prediction_history.clear()

    # Zobrazení výsledného obrazu v okně
    cv2.imshow("Emotion Detection (Custom CNN 48x48)", frame)

    # Ukončení smyčky, pokud uživatel stiskne klávesu 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Vyčištění paměti a zavření oken
cap.release()
cv2.destroyAllWindows()
print("Aplikace úspěšně ukončena.")