import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

import cv2
import numpy as np
from fpdf import FPDF
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# 确保能导入同级目录的分析脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emotion_fluctuation_analysis import main as fluctuation_main
from emotion_transition_analysis import analyze_and_plot as transition_analyze

# --- 模型 ---
model = load_model("../model/emotion_model_v2.h5")
emotion_labels = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Calm","Contempt"]

CAPTURE_DURATION = 2 * 60  # 数据采集时间限制（2分钟，仅摄像头模式生效）

# --- MTCNN 人脸检测 ---
detector = MTCNN()


def validate_video(
    video_path: str,
    max_seconds: Optional[float],
    max_mb: Optional[float]
) -> Tuple[float, float]:
    """
    校验视频是否符合长度/大小限制，返回 (duration_seconds, size_mb)。
    如果无法获取时长（无 fps），duration_seconds 返回 -1。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"未找到视频文件：{video_path}")

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if max_mb is not None and size_mb > max_mb:
        raise ValueError(f"视频大小 {size_mb:.2f}MB 超过限制 {max_mb}MB")

    cap_probe = cv2.VideoCapture(video_path)
    if not cap_probe.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    fps = cap_probe.get(cv2.CAP_PROP_FPS)
    frame_count = cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)
    cap_probe.release()

    duration = frame_count / fps if fps and fps > 0 else -1
    if max_seconds is not None and duration > 0 and duration > max_seconds:
        raise ValueError(f"视频时长 {duration:.2f}s 超过限制 {max_seconds}s")

    return duration, size_mb


def run_detection(
    video_path: str = None,
    output_video_path: str = None,
    max_seconds: Optional[float] = None,
    max_mb: Optional[float] = None
) -> None:
    """
    运行情绪检测：
    - video_path 为空：走摄像头，2 分钟采集限制
    - video_path 指向文件：对视频逐帧检测，直到文件结束，并可保存标注后视频
    """
    use_video_file = video_path is not None
    cap_source = video_path if use_video_file else 0

    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)

    frame_skip = 2  # 每隔2帧处理1次（可调整为1/3，1=不跳帧）
    frame_count = 0  # 帧计数器
    # 视频模式：先做长度/大小校验
    if use_video_file:
        try:
            duration, size_mb = validate_video(video_path, max_seconds, max_mb)
            if duration > 0:
                print(f"[INFO] 输入视频时长: {duration:.2f}s, 大小: {size_mb:.2f}MB")
            else:
                print(f"[INFO] 输入视频大小: {size_mb:.2f}MB，未能获取时长（fps 为 0）")
        except Exception as e:
            print(f"[ERROR] {e}")
            return

    # 输出视频路径（仅在视频模式保存）
    if use_video_file and output_video_path is None:
        output_video_path = os.path.join(result_dir, "labeled_output.mp4")

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"无法打开{'视频文件' if use_video_file else '摄像头'}: {cap_source}")
        return

    # --- CSV 路径 ---
    if not os.path.exists("../data"):
        os.makedirs("../data")
    csv_path = "../data/emotion_log.csv"
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["time", "emotion"])

    # 记录开始时间
    capture_start_time = datetime.now()
    end_time_limit = capture_start_time + timedelta(seconds=CAPTURE_DURATION) if not use_video_file else None
    is_collecting = True  # 采集状态标志

    # 视频写入器（仅视频模式）
    video_writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0  # 默认 25fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if use_video_file:
            frame_count += 1
        if not use_video_file:
            frame = cv2.flip(frame, 1)
        current_time = datetime.now()

        # 跳帧逻辑：只处理第N帧（frame_count % frame_skip == 0）
        if frame_count % frame_skip != 0:
            # 不处理的帧直接写入视频（保持输出视频时长正常）
            if use_video_file and video_writer is not None:
                video_writer.write(frame)
            continue  # 跳过后续检测逻辑，直接进入下一帧

        # 摄像头模式：超过采集时间则停止写入 CSV，但仍可继续展示画面
        if end_time_limit and current_time >= end_time_limit and is_collecting:
            print("已达到2分钟数据采集上限，停止数据采集（摄像头保持打开）")
            is_collecting = False
            csv_file.close()

        # MTCNN 需要 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)

        h, w, _ = frame.shape

        for res in results:
            x1, y1, width, height = res["box"]

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
            face_input = face_resized.astype("float32")
            face_input = (face_input / 255.0) * 2 - 1
            face_input = np.expand_dims(face_input, axis=-1)
            face_input = np.expand_dims(face_input, axis=0)

            # --- 预测 ---
            pred_prob = model.predict(face_input, verbose=0)
            pred_idx = int(np.argmax(pred_prob))
            pred_emotion = emotion_labels[pred_idx]
            pred_confidence = round(float(np.max(pred_prob)) * 100, 2)

            # --- 存入 CSV ---
            if is_collecting:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([now_str, pred_emotion])
                csv_file.flush()

            # --- 画框 ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{pred_emotion} ({pred_confidence}%)",
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # 初始化视频写入器（仅视频文件模式）
        if use_video_file and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                fps if fps and fps > 0 else 25.0,
                (frame.shape[1], frame.shape[0])
            )

        # 保存带标注的视频帧
        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
    if not csv_file.closed:
        csv_file.close()
    cv2.destroyAllWindows()

    capture_end_time = datetime.now()
    print("生成情绪波动分析图...")
    fluctuation_main()
    print("生成情绪转移矩阵图...")
    transition_img_path = transition_analyze()

    # --------------------------
    # PDF生成部分（放在最后，不改变前面任何顺序）
    # --------------------------
    # 1. 字体配置（嵌入项目的宋体）
    font_relative_path = "../fonts/STFANGSO.TTF"  # 相对于当前code目录的路径
    font_path = os.path.abspath(font_relative_path)  # 转为绝对路径

    # 2. 生成PDF报告
    print("生成PDF分析报告...")
    pdf = FPDF()
    pdf.add_page()

    # 加载嵌入字体
    font_name = "STFANGSO"
    if os.path.exists(font_path):
        pdf.add_font(font_name, "", font_path, uni=True)
        print("中文字体加载成功")
    else:
        font_name = "Arial"  # 字体丢失时降级为英文
        print(f"警告：未找到字体文件 {font_path}，将使用英文显示")

    # 添加标题
    pdf.set_font(font_name, "", 16)
    if font_name == "STFANGSO":
        pdf.cell(0, 15, "实时情绪检测分析报告", 0, 1, "C")
    else:
        pdf.cell(0, 15, "Real-Time Emotion Detection Report", 0, 1, "C")
    pdf.ln(2)

    # 添加采集信息
    pdf.set_font(font_name, "", 12)
    if font_name == "STFANGSO":
        pdf.cell(
            0,
            10,
            f"采集时间：{capture_start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {capture_end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            0,
            1,
        )
        pdf.cell(0, 10, f"采集时长：{int((capture_end_time - capture_start_time).total_seconds())} 秒", 0, 1)
        if use_video_file:
            pdf.cell(0, 10, f"输入源：视频文件 {video_path}", 0, 1)
        else:
            pdf.cell(0, 10, "输入源：摄像头", 0, 1)
    else:
        pdf.cell(
            0,
            10,
            f"Collection Time: {capture_start_time.strftime('%Y-%m-%d %H:%M:%S')} to {capture_end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            0,
            1,
        )
        pdf.cell(0, 10, f"Duration: {int((capture_end_time - capture_start_time).total_seconds())} seconds", 0, 1)
        if use_video_file:
            pdf.cell(0, 10, f"Source: video file {video_path}", 0, 1)
        else:
            pdf.cell(0, 10, "Source: webcam", 0, 1)
    pdf.ln(2)

    # 添加情绪波动图
    pdf.set_font(font_name, "", 14)
    fluctuation_img_path = os.path.join(result_dir, "emotion_fluctuation_curve.png")
    if font_name == "STFANGSO":
        pdf.cell(0, 12, "情绪波动趋势图", 0, 1)
    else:
        pdf.cell(0, 12, "Emotion Fluctuation Trend", 0, 1)
    if os.path.exists(fluctuation_img_path):
        pdf.image(fluctuation_img_path, x=10, w=190)
    else:
        pdf.set_font(font_name, "", 12)
        if font_name == "STFANGSO":
            pdf.cell(0, 10, f"⚠️ 未找到波动图：{fluctuation_img_path}", 0, 1)
        else:
            pdf.cell(0, 10, f"⚠️ Fluctuation chart not found: {fluctuation_img_path}", 0, 1)
    pdf.ln(4)

    # 添加情绪转移图
    pdf.set_font(font_name, "", 14)
    if font_name == "STFANGSO":
        pdf.cell(0, 12, "情绪转移概率矩阵", 0, 1)
    else:
        pdf.cell(0, 12, "Emotion Transition Matrix", 0, 1)
    if os.path.exists(transition_img_path):
        pdf.image(transition_img_path, x=10, w=190)
    else:
        pdf.set_font(font_name, "", 12)
        if font_name == "STFANGSO":
            pdf.cell(0, 10, f"⚠️ 未找到转移图：{transition_img_path}", 0, 1)
        else:
            pdf.cell(0, 10, f"⚠️ Transition chart not found: {transition_img_path}", 0, 1)

    # 保存PDF
    pdf_path = os.path.join(result_dir, "emotion_analysis_report.pdf")
    pdf.output(pdf_path)
    print(f"PDF报告已保存至：{pdf_path}")
    print("所有操作完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时情绪检测（摄像头或视频文件）")
    parser.add_argument("--video", type=str, help="视频文件路径。省略则使用摄像头。")
    parser.add_argument("--output", type=str, help="标注后视频输出路径（仅视频模式有效，默认 result/labeled_output.mp4）")
    parser.add_argument("--max-seconds", type=float, default=300, help="视频最长秒数限制（默认 300s，视频模式有效）")
    parser.add_argument("--max-mb", type=float, default=500, help="视频大小限制，单位 MB（默认 500MB，视频模式有效）")
    args = parser.parse_args()
    run_detection(
        video_path=args.video,
        output_video_path=args.output,
        max_seconds=args.max_seconds,
        max_mb=args.max_mb
    )