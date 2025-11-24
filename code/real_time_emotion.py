import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from datetime import datetime
import csv

# --------------------------
# 1. 常量与模型加载
# --------------------------

EMOTION_LABELS = [
    'Angry', 'Disgusted', 'Scared', 
    'Happy', 'Sad', 'Surprised', 'Calm'
]

# 模型路径（与你训练脚本一致）
MODEL_PATH = "../model/emotion_model_v2.h5"

# Haar 人脸检测器
CASCADE_PATH = "../haarcascades/haarcascade_frontalface_default.xml"


# --------------------------
# 2. 加载模型
# --------------------------

print("[INFO] 正在加载情绪识别模型...")
model = load_model(MODEL_PATH)
print("[INFO] 模型加载完成！")


# --------------------------
# 3. 预测函数（核心接口）
# --------------------------

def predict_emotion_from_face(face_gray):
    """
    输入：灰度人脸图像 ROI
    输出：情绪标签字符串
    """
    # 1. resize 为 48x48（模型输入要求）
    face = cv2.resize(face_gray, (48, 48))

    # 2. 转 float32
    face = face.astype("float32")

    # 3. 归一化 (-1, 1) —— 完全对齐你训练代码
    face = (face / 255.0) * 2 - 1

    # 4. 扩展维度为 (1, 48, 48, 1)
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    # 5. 模型预测
    preds = model.predict(face)
    emotion_idx = int(np.argmax(preds))
    return EMOTION_LABELS[emotion_idx]


# --------------------------
# 4. 实时摄像头捕捉
# --------------------------

def run_realtime_emotion(csv_path="../data/emotion_log.csv"):
    print("[INFO] 正在初始化摄像头...")
    
    # 载入人脸检测器
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError("无法加载 Haar 分类器，请检查路径")

    # 准备 CSV
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["time", "emotion"])

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("[INFO] 开始实时情绪识别... 按 Q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                emotion = predict_emotion_from_face(roi)

                # 显示框
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, emotion, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # 写入 CSV
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([now, emotion])
                csv_file.flush()

            cv2.imshow("Real-time Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print("[INFO] 已退出，数据已保存：", csv_path)


if __name__ == "__main__":
    run_realtime_emotion()
