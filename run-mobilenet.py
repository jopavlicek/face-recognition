import cv2
import os
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("models", "mobilenet.keras")

print("Načítám MOBILENET model...")
model = load_model(MODEL_PATH)
print("Model úspěšně načten.")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
prediction_history = deque(maxlen=5)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Chyba: Nepodařilo se otevřít webkameru.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Padding 15% jako v datasetu
        pad_w = int(w * 0.15)
        pad_h = int(h * 0.15)
        
        img_h, img_w = gray.shape
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        face = gray[y1:y2, x1:x2]

        # Změna rozlišení na 128x128 pro shodu s modelem
        face = cv2.resize(face, (128, 128))
        face = face.astype("float32")

        # Úprava tvaru na (1, 128, 128, 1)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predikce a vyhlazení
        raw_prediction = model.predict(face, verbose=0)[0]
        prediction_history.append(raw_prediction)
        smoothed_prediction = np.mean(prediction_history, axis=0)

        emotion_index = np.argmax(smoothed_prediction)
        emotion = emotion_labels[emotion_index]
        confidence = smoothed_prediction[emotion_index]

        # Vykreslení
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        prediction_history.clear()

    cv2.imshow("Emotion Detection (MobileNetV2 128x128)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()