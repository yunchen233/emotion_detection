import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN


model = load_model('../model/emotion_model.h5') 
emotion_labels = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm']


detector = MTCNN()
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    results = detector.detect_faces(frame)

    for res in results:
        x1, y1, width, height = res['box']
        x2, y2 = x1 + width, y1 + height

        
        face = frame[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_input = face_resized.reshape(1, 48, 48, 1)
        face_input = (face_input / 255.0) * 2 - 1

        
        pred_prob = model.predict(face_input, verbose=0)
        pred_emotion = emotion_labels[np.argmax(pred_prob)]
        pred_confidence = round(np.max(pred_prob) * 100, 2)

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{pred_emotion} ({pred_confidence}%)', 
                    (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Real-Time Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()