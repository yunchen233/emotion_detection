"""
双人检测模式，只支持实时摄像头，对应报告只关注两者之间的关系分析
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from fpdf import FPDF
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# --- 1. 导入项目模块 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_tracker import CentroidTracker
import dyadic_analysis  # 双人分析模块
import api_double  # 导入API分析模块

# --- 模型与配置 ---
model = load_model("../model/emotion_model_v2.h5", compile=False)
emotion_labels = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Calm", "Contempt"]
detector = MTCNN()

# 初始化追踪器
tracker = CentroidTracker(maxDisappeared=20)

def run_detection():
    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = 25.0

    # 2. CSV 初始化
    if not os.path.exists("../data"):
        os.makedirs("../data", exist_ok=True)

    csv_path = "../data/double_emotion_log.csv"

    # 清空之前的CSV文件
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["time", "face_id", "emotion"])

    # 重新以追加模式打开CSV文件
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    capture_start = datetime.now()
    is_collecting = True

    # 颜色库
    COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0)]

    print("--- 开始双人检测，请确保摄像头前有两人 ---")
    print("提示: 按ESC键结束检测并生成报告")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 摄像头镜像

        # 转化RGB为MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 检测与追踪 ---
        results = detector.detect_faces(rgb_frame)
        rects = [res["box"] for res in results]
        objects = tracker.update(rects)  # 返回id:box，质心坐标

        h_img, w_img, _ = frame.shape

        for obj_id, (box, centroid) in objects.items():
            x, y, w, h = box
            if x < 0 or y < 0:
                continue

            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue

            try:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = (face_resized.astype("float32") / 255.0) * 2 - 1
                face_input = np.expand_dims(np.expand_dims(face_input, axis=-1), axis=0)

                probs = model.predict(face_input, verbose=0)[0]
                label_idx = np.argmax(probs)
                label = emotion_labels[label_idx]
                conf = probs[label_idx]

                # --- 数据记录 ---
                if is_collecting:
                    csv_writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        obj_id,
                        label
                    ])
                    csv_file.flush()

                # --- 绘制 UI ---
                color = COLORS[obj_id % len(COLORS)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                text = f"ID:{obj_id} {label} {int(conf * 100)}%"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                print(f"[WARN] 处理错误: {e}")
                continue

        # 显示当前采集状态

        cv2.imshow("Dyadic Emotion System", frame)

        # 按ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            print("[INFO] 检测到ESC键，结束采集...")
            break
        elif key == ord('q'):  # Q键
            print("[INFO] 检测到Q键，结束采集...")
            break

    cap.release()
    if not csv_file.closed:
        csv_file.close()
    cv2.destroyAllWindows()

    # --- 生成报告 ---
    print("\n--- 生成双人交互报告 ---")

    # 初始化变量
    ids_found = []
    dashboard_path = None

    # 1. 运行双人分析 (Dashboard)
    try:
        analysis_result = dyadic_analysis.analyze_dyadic_relationship(result_dir)
        if analysis_result:
            dashboard_path, id_a, id_b = analysis_result
            ids_found = [id_a, id_b]
            print(f"[INFO] 仪表盘已生成: {dashboard_path}")
        else:
            print("[WARNING] 双人分析失败或数据不足")
    except Exception as e:
        print(f"[ERROR] 双人分析失败: {e}")
        import traceback
        traceback.print_exc()

    # 2. 调用API进行文本分析
    try:
        print("[INFO] 开始AI文本分析...")
        api_result = api_double.analyze_emotion_from_csv(csv_path)
        if not api_result:
            print("[WARNING] API分析失败或未返回结果")
    except Exception as e:
        print(f"[ERROR] API分析失败: {e}")
        import traceback
        traceback.print_exc()

    # 3. 确保有数据生成PDF
    if len(ids_found) >= 2:
        print(f"[INFO] 检测到双人数据: ID {ids_found[0]} 和 ID {ids_found[1]}")
    else:
        print("[WARNING] 未检测到足够的双人数据，PDF报告可能不完整")
        # 尝试从CSV文件中获取ID
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            unique_ids = df['face_id'].unique()
            if len(unique_ids) >= 2:
                ids_found = unique_ids[:2].tolist()
                print(f"[INFO] 从CSV文件中提取ID: {ids_found}")
        except Exception as e:
            print(f"[ERROR] 无法从CSV提取ID: {e}")

    # 4. 生成 PDF
    try:
        generate_pdf_report(capture_start, datetime.now(), ids_found, result_dir, dashboard_path)
    except Exception as e:
        print(f"[ERROR] 生成PDF报告失败: {e}")
        import traceback
        traceback.print_exc()


def generate_pdf_report(start_t, end_t, target_ids, res_dir, dashboard_img):
    """
    生成双人版 PDF 报告 (简化版，无截图)
    """
    # 确保目录存在
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)

    # 创建唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(res_dir, f"Double_emotion_Report.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = os.path.abspath("../fonts/STFANGSO.TTF")
    font_name = "STFANGSO" if os.path.exists(font_path) else "Arial"
    if font_name == "STFANGSO":
        pdf.add_font(font_name, "", font_path, uni=True)

    # --- Page 1: 封面与仪表盘 ---
    pdf.add_page()
    pdf.set_font(font_name, "", 20)
    pdf.cell(0, 15, "双人交互分析报告", 0, 1, "C")

    # 基础信息
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 10, f"时间: {start_t.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"时长: {int((end_t - start_t).total_seconds())}s", 0, 1)

    if len(target_ids) >= 2:
        pdf.cell(0, 10, f"分析对象: ID {target_ids[0]} & ID {target_ids[1]}", 0, 1)
    else:
        pdf.cell(0, 10, "分析对象: 未检测到足够的双人交互数据", 0, 1)

    pdf.ln(10)

    # 插入仪表盘图片
    if dashboard_img and os.path.exists(dashboard_img):
        try:
            # 调整图片位置和大小
            pdf.image(dashboard_img, x=10, y=60, w=190)
            pdf.ln(130)
        except Exception as e:
            print(f"[WARNING] 无法插入仪表盘图片: {e}")
            pdf.set_font(font_name, "", 12)
            pdf.cell(0, 10, "[仪表盘图片加载失败]", 0, 1)
    else:
        pdf.set_font(font_name, "", 12)
        pdf.cell(0, 10, "[无足够数据生成交互分析图表]", 0, 1)
        pdf.ln(10)

    # API分析结果
    txt_path = os.path.join(res_dir, "api2分析结果.txt")
    if os.path.exists(txt_path):
        # 添加新页
        pdf.add_page()

        # 标题
        pdf.set_font(font_name, "", 16)
        pdf.cell(0, 10, "AI智能分析总结", 0, 1, "C")
        pdf.ln(10)

        # 正文内容
        pdf.set_font(font_name, "", 12)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 分割文本为段落
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():  # 跳过空段落
                    # 处理过长的段落
                    lines = para.split('\n')
                    for line in lines:
                        if line.strip():
                            pdf.multi_cell(0, 8, line.strip())
                            pdf.ln(2)
                    pdf.ln(5)

        except Exception as e:
            print(f"[WARN] 读取分析文本失败: {e}")
            pdf.set_font(font_name, "", 12)
            pdf.cell(0, 10, "无法读取分析文件", 0, 1)
    else:
        print(f"[INFO] 未找到分析文件: {txt_path}，跳过该部分。")

    # 输出PDF
    try:
        pdf.output(out_file)
        print(f"[SUCCESS] PDF 已生成: {out_file}")
    except PermissionError as e:
        print(f"[ERROR] 权限被拒绝，无法写入PDF文件: {e}")
        # 尝试写入当前目录
        alt_out_file = f"Double_emotion_Report.pdf"
        try:
            pdf.output(alt_out_file)
            print(f"[SUCCESS] PDF 已生成到当前目录: {alt_out_file}")
        except Exception as e2:
            print(f"[ERROR] 仍然无法写入: {e2}")
            raise
    except Exception as e:
        print(f"[ERROR] 生成PDF失败: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    run_detection()
