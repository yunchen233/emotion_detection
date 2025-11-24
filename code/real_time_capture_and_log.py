import cv2
import os
import csv
from datetime import datetime

import numpy as np
# 这里按你的项目来导入模型
# 比如：
# from your_model_file import model, EMOTION_LABELS
# 下面先写一个占位的接口函数，你要按你自己项目里的模型实现它

# -------------------- 1. 模型接口（你要按你项目来改这里） --------------------
def predict_emotion_from_face(face_img_gray):
    """
    输入：灰度人脸图像 (numpy 数组)，例如 48x48
    输出：情绪标签字符串，比如 'Happy'
    """
    # TODO: 下面是“伪代码”，你要用你项目里的代码替换
    # 假设你的模型输入是 48x48 灰度图：
    face_resized = cv2.resize(face_img_gray, (48, 48))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=-1)  # (48,48,1)
    face_resized = np.expand_dims(face_resized, axis=0)   # (1,48,48,1)

    # y_pred = model.predict(face_resized)
    # emotion_idx = int(np.argmax(y_pred))
    # emotion_label = EMOTION_LABELS[emotion_idx]

    # 这里先返回一个占位，避免运行错误，改成你自己的：
    emotion_label = "Calm"
    return emotion_label

# -------------------- 2. 实时捕捉 + 记录 CSV 的主流程 --------------------

def run_realtime_capture_and_log(
    csv_path="../data/emotion_log.csv",
    cascade_path="../haarcascades/haarcascade_frontalface_default.xml",
    camera_index=0
):
    # 1. 准备人脸检测器
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"找不到 Haar 模型文件：{cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 2. 准备 CSV（如果不存在就写表头）
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["time", "emotion"])

    # 3. 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        csv_file.close()
        return

    print("开始捕捉摄像头情绪流，按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取摄像头画面失败")
                break

            # 转灰度做人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # 对每一张脸做情绪预测
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]

                emotion_label = predict_emotion_from_face(face_roi)
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 写入 CSV 一行：时间 + 情绪
                writer.writerow([now_str, emotion_label])
                csv_file.flush()  # 让数据实时落盘

                # 在画面上画框 + 写文字
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # 显示画面
            cv2.imshow("Real-time Emotion Capture", frame)

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"已停止捕捉，情绪序列保存在：{csv_path}")


if __name__ == "__main__":
    run_realtime_capture_and_log()
