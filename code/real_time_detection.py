import cv2
import numpy as np
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

# --- 模型 ---
model = load_model('../model/emotion_model_v2.h5') 
emotion_labels = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm']

# --- MTCNN 人脸检测 ---
detector = MTCNN()

# --- CSV 路径 ---
csv_path = "../data/emotion_log.csv"
if not os.path.exists("../data"):
    os.makedirs("../data")

# 如果文件不存在，加表头
write_header = not os.path.exists(csv_path)
csv_file = open(csv_path, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["time", "emotion"])

# --- 摄像头 ---
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)

    # MTCNN 需要 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    h, w, _ = frame.shape

    for res in results:
        x1, y1, width, height = res['box']

        # 边界修正
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x1 + width)
        y2 = min(h, y1 + height)
        if x2 <= x1 or y2 <= y1:
            continue

        # --- 截取人脸 ---
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))

        # --- 预处理 ---
        face_input = face_resized.astype('float32')
        face_input = (face_input / 255.0) * 2 - 1
        face_input = np.expand_dims(face_input, axis=-1)
        face_input = np.expand_dims(face_input, axis=0)

        # --- 预测 ---
        pred_prob = model.predict(face_input, verbose=0)
        pred_idx = int(np.argmax(pred_prob))
        pred_emotion = emotion_labels[pred_idx]
        pred_confidence = round(float(np.max(pred_prob)) * 100, 2)

        # --- 存入 CSV ---
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([now_str, pred_emotion])
        csv_file.flush()

        # --- 画框 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f'{pred_emotion} ({pred_confidence}%)',
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

    cv2.imshow('Real-Time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
