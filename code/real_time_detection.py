import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
import cv2
import numpy as np
from fpdf import FPDF
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- 导入分析模块 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emotion_fluctuation_analysis import main as fluctuation_main
from emotion_transition_analysis import analyze_and_plot as transition_analyze
from snapshot_manager import SnapshotManager

# --- 配置 ---
MODEL_PATH = "../model/emotion_model_v2.h5"
FONT_PATH = "../fonts/STFANGSO.TTF"  # 字体路径

model = load_model(MODEL_PATH, compile=False)
emotion_labels = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Calm", "Contempt"]
detector = MTCNN()
snapshot_manager = SnapshotManager()


def generate_pdf_report(start_t, end_t, snapshots, res_dir, trans_img,frame_count):
    pdf = FPDF()
    pdf.add_page()

    font_name = "STFANGSO" if os.path.exists(FONT_PATH) else "Arial"
    if font_name == "STFANGSO": pdf.add_font(font_name, "", FONT_PATH, uni=True)

    # 标题
    pdf.set_font(font_name, "", 16)
    title = "情绪监测报告" if font_name == "STFANGSO" else "Emotion Report"
    pdf.cell(0, 15, title, 0, 1, "C")

    # 基础信息
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 10, f"开始时间: {start_t}s", 0, 1)
    pdf.cell(0, 10, f"结束时间: {end_t}s", 0, 1)
    pdf.cell(0, 10, f"持续时间: {int((end_t - start_t).total_seconds())}s", 0, 1)
    pdf.cell(0, 10, f"帧数: {frame_count}", 0, 1)
    pdf.ln(5)

    # 1. 抓拍
    pdf.set_font(font_name, "", 14)
    pdf.cell(0, 10, "1. 典型情绪瞬间 (Snapshots)", 0, 1)
    pdf.ln(2)

    x_start, y_start, w_img, h_img = 10, pdf.get_y(), 45, 60
    col = 0
    if not snapshots:
        pdf.set_font(font_name, "", 10)
        pdf.cell(0, 10, "无高置信度截图", 0, 1)
    else:
        for emo, path in snapshots.items():
            if not os.path.exists(path): continue
            x = x_start + (col * (w_img + 5))
            if col >= 4:
                col = 0;
                y_start += h_img + 15;
                x = x_start
            pdf.image(path, x=x, y=y_start, w=w_img, h=h_img)
            pdf.set_xy(x, y_start + h_img + 2)
            pdf.set_font(font_name, "", 10)
            pdf.cell(w_img, 5, emo, 0, 0, "C")
            col += 1
        pdf.set_xy(10, y_start + h_img + 15 + (h_img if len(snapshots) > 4 else 0))
        pdf.ln(5)

    # 2. 波动图
    pdf.set_font(font_name, "", 14)
    pdf.cell(0, 10, "2. 情绪波动曲线", 0, 1)
    fluc_img = os.path.join(res_dir, "emotion_fluctuation_curve.png")
    if os.path.exists(fluc_img): pdf.image(fluc_img, x=10, w=190)
    pdf.ln(5)

    # 3. 转移矩阵
    pdf.set_font(font_name, "", 14)
    pdf.cell(0, 10, "3. 状态转移概率", 0, 1)
    if trans_img and os.path.exists(trans_img): pdf.image(trans_img, x=20, w=170)
    pdf.ln(5)

    #4.api分析结果
    txt_path = os.path.join(res_dir, "api1分析结果.txt")
    if os.path.exists(txt_path):
        # 标题
        pdf.set_font(font_name, "", 14)
        pdf.cell(0, 10, "4. AI 智能分析总结", 0, 1)
        pdf.ln(5)
        # 正文内容
        pdf.set_font(font_name, "", 12)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # multi_cell 用于自动换行长文本
            # w=0 表示占满宽度, h=8 是行高
            pdf.multi_cell(0, 8, content)

        except Exception as e:
            print(f"[WARN] 读取分析文本失败: {e}")
            pdf.cell(0, 10, "无法读取分析文件", 0, 1)
    else:
        print(f"[INFO] 未找到分析文件: {txt_path}，跳过该部分。")

    out_file = os.path.join(res_dir, "Emotion_Report.pdf")
    try:
        pdf.output(out_file)
        print(f"[INFO] PDF 报告已生成: {out_file}")
    except Exception as e:
        print(f"[ERROR] 生成 PDF 失败 (可能是文件被占用): {e}")



def run_camera_detection():
    # 目录初始化
    if not os.path.exists("../data"): os.makedirs("../data")
    if not os.path.exists("../result"): os.makedirs("../result")

    csv_path = "../data/emotion_log.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")  # 每次覆盖模式，适合单次报告
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "emotion"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("--- 启动摄像头检测 (按ESC退出) ---")
    start_time = datetime.now()
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            frame = cv2.flip(frame, 1)  # 镜像

            # 检测逻辑
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_frame)

            for res in results:
                x, y, w, h = res["box"]
                if x < 0 or y < 0: continue

                face = frame[y:y + h, x:x + w]
                if face.size == 0: continue

                # 预处理
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = (face_resized.astype("float32") / 255.0) * 2 - 1
                face_input = np.expand_dims(np.expand_dims(face_input, -1), 0)

                # 推理
                probs = model.predict(face_input, verbose=0)[0]
                label_idx = np.argmax(probs)
                label = emotion_labels[label_idx]
                conf = probs[label_idx]

                # 记录数据
                csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label])
                snapshot_manager.update(frame, label, conf, (x, y, w, h))

                # 绘制
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(conf * 100)}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Emotion System", frame)
            if cv2.waitKey(1) & 0xFF == 27: break


    finally:
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

    # --- 后处理与报告生成 ---
    print("\n--- 开始生成分析报告 ---")
    try:
        fluctuation_main()
    except:
        pass

    try:
        trans_img = transition_analyze()
    except:
        trans_img = ""

    snaps = snapshot_manager.get_summary_images()
    generate_pdf_report(start_time, datetime.now(), snaps, "../result", trans_img,frame_count)


if __name__ == "__main__":
    run_camera_detection()