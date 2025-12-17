import argparse
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
#python upload_video.py --input "你的视频文件路径"

# --- 配置 ---
MODEL_PATH = "../model/emotion_model_v2.h5"
model = load_model(MODEL_PATH, compile=False)
emotion_labels = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Calm", "Contempt"]
detector = MTCNN()


def validate_video(video_path, max_seconds=None, max_mb=None):
    """视频文件校验"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"文件不存在: {video_path}")

    if max_mb:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if size_mb > max_mb: raise ValueError(f"文件过大: {size_mb:.2f}MB")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("无法读取视频流")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if max_seconds and fps > 0:
        dur = frames / fps
        if dur > max_seconds: raise ValueError(f"时长过长: {dur:.2f}s")

    return True


def process_video(input_path, output_path=None):
    """
    核心处理函数：读取视频 -> 标注情绪 -> 保存视频
    """
    # 默认输出路径
    if not output_path:
        result_dir = "../result"
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        output_path = os.path.join(result_dir, "labeled_output.mp4")

    print(f"[INFO] 开始处理视频: {input_path}")
    print(f"[INFO] 输出路径: {output_path}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_cnt += 1
        if frame_cnt % 10 == 0:
            print(f"进度: {frame_cnt}/{total_frames}", end='\r')

        # 转RGB供MTCNN使用
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = detector.detect_faces(rgb_frame)
        except Exception as e:
            print(f"\n[WARN] Frame {frame_cnt} detection error: {e}")
            out.write(frame)  # 出错也写入原帧
            continue

        for res in results:
            x, y, w, h = res["box"]
            if x < 0 or y < 0: continue

            # 提取人脸
            face = frame[y:y + h, x:x + w]
            if face.size == 0: continue

            # 预处理
            try:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = (face_resized.astype("float32") / 255.0) * 2 - 1
                face_input = np.expand_dims(np.expand_dims(face_input, -1), 0)

                # 推理
                probs = model.predict(face_input, verbose=0)[0]
                label_idx = np.argmax(probs)
                label = emotion_labels[label_idx]
                conf = probs[label_idx]

                # 绘图
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{label} {int(conf * 100)}%"
                # 增加黑色背景让文字更清晰
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            except Exception:
                continue

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n[SUCCESS] 处理完成，视频已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地视频情绪标注接口")
    parser.add_argument("--input", required=True, type=str, help="输入视频路径")
    parser.add_argument("--output", type=str, help="输出视频路径 (可选)")
    parser.add_argument("--max_mb", type=float, default=500, help="最大文件大小(MB)")

    args = parser.parse_args()

    try:
        # 1. 校验
        validate_video(args.input, max_mb=args.max_mb)
        # 2. 执行
        process_video(args.input, args.output)
    except Exception as e:
        print(f"[ERROR] {e}")