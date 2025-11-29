import cv2
import numpy as np
import csv
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os
import sys
from fpdf import FPDF

# 确保能导入同级目录的分析脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emotion_fluctuation_analysis import main as fluctuation_main
from emotion_transition_analysis import analyze_and_plot as transition_analyze

# --- 模型 ---
model = load_model('../model/emotion_model_v2.h5') 
emotion_labels = ['Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Calm']

CAPTURE_DURATION = 2 * 60# 数据采集时间限制（2分钟）

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

# 记录开始时间
start_time = datetime.now()
end_time = start_time + timedelta(seconds=CAPTURE_DURATION)
is_collecting = True  # 采集状态标志

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    current_time = datetime.now()

    # 检查是否超过采集时间，停止采集但不关闭摄像头
    if current_time >= end_time and is_collecting:
        print("已达到2分钟数据采集上限，停止数据采集（摄像头保持打开）")
        is_collecting = False  # 关闭采集状态
        csv_file.close()  # 停止采集后关闭CSV文件

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
        if is_collecting:
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
if is_collecting:
    csv_file.close()
cv2.destroyAllWindows()
result_dir = "../result"
os.makedirs(result_dir, exist_ok=True)
print("生成情绪波动分析图...")
fluctuation_main()
print("生成情绪转移矩阵图...")
transition_img_path = transition_analyze()

# --------------------------
# PDF生成部分（放在最后，不改变前面任何顺序）
# --------------------------
# 1. 字体配置（嵌入项目的宋体）
# 字体路径
font_relative_path = "../fonts/STFANGSO.TTF"  # 相对于当前code目录的路径
font_path = os.path.abspath(font_relative_path)  # 转为绝对路径

# 2. 生成PDF报告
print("生成PDF分析报告...")
pdf = FPDF()
pdf.add_page()

# 加载嵌入字体
font_name = "STFANGSO"
if os.path.exists(font_path):
    pdf.add_font(font_name, '', font_path, uni=True)
    print("中文字体加载成功")
else:
    font_name = "Arial"  # 字体丢失时降级为英文
    print(f"警告：未找到字体文件 {font_path}，将使用英文显示")

# 添加标题
pdf.set_font(font_name, '', 16)
if font_name == "STFANGSO":
    pdf.cell(0, 15, '实时情绪检测分析报告', 0, 1, 'C')
else:
    pdf.cell(0, 15, 'Real-Time Emotion Detection Report', 0, 1, 'C')
pdf.ln(2)

# 添加采集信息
pdf.set_font(font_name, '', 12)
end_time = datetime.now()
if font_name == "STFANGSO":
    pdf.cell(0, 10, f"采集时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"采集时长：{int((end_time - start_time).total_seconds())} 秒", 0, 1)
else:
    pdf.cell(0, 10, f"Collection Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Duration: {int((end_time - start_time).total_seconds())} seconds", 0, 1)
pdf.ln(2)

# 添加情绪波动图
pdf.set_font(font_name, '', 14)
fluctuation_img_path = os.path.join(result_dir, "emotion_fluctuation_curve.png")
if font_name == "STFANGSO":
    pdf.cell(0, 12, '情绪波动趋势图', 0, 1)
else:
    pdf.cell(0, 12, 'Emotion Fluctuation Trend', 0, 1)
if os.path.exists(fluctuation_img_path):
    pdf.image(fluctuation_img_path, x=10, w=190)
else:
    pdf.set_font(font_name, '', 12)
    if font_name == "STFANGSO":
        pdf.cell(0, 10, f"⚠️ 未找到波动图：{fluctuation_img_path}", 0, 1)
    else:
        pdf.cell(0, 10, f"⚠️ Fluctuation chart not found: {fluctuation_img_path}", 0, 1)
pdf.ln(4)

# 添加情绪转移图
pdf.set_font(font_name, '', 14)
if font_name == "STFANGSO":
    pdf.cell(0, 12, '情绪转移概率矩阵', 0, 1)
else:
    pdf.cell(0, 12, 'Emotion Transition Matrix', 0, 1)
if os.path.exists(transition_img_path):
    pdf.image(transition_img_path, x=10, w=190)
else:
    pdf.set_font(font_name, '', 12)
    if font_name == "STFANGSO":
        pdf.cell(0, 10, f"⚠️ 未找到转移图：{transition_img_path}", 0, 1)
    else:
        pdf.cell(0, 10, f"⚠️ Transition chart not found: {transition_img_path}", 0, 1)

# 保存PDF
pdf_path = os.path.join(result_dir, "emotion_analysis_report.pdf")
pdf.output(pdf_path)
print(f"PDF报告已保存至：{pdf_path}")
print("所有操作完成！")

