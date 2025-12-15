import os
import cv2
import csv
import numpy as np
import tensorflow as tf
import threading
import uuid
import time
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, flash, jsonify
from datetime import datetime
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# 引入工具
from web_utils import generate_web_report

app = Flask(__name__)
app.secret_key = "healing_key_secret"  # 必须设置，用于 flash 消息

# --- 全局配置 ---
# 显存按需分配
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

MODEL_PATH = "model/emotion_model_v2.h5"
model = load_model(MODEL_PATH)
detector = MTCNN()
emotion_labels = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Calm", "Contempt"]

# --- 全局状态 ---
camera = None
is_recording = False
start_time = None
csv_file = None
csv_writer = None

# 用于存储后台任务的进度： { 'task_id': {'progress': 10, 'status': 'processing', 'result': None} }
processing_tasks = {}


def init_csv():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    csv_path = os.path.join(data_dir, "emotion_log.csv")
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["time", "emotion"])
    return f, writer


# --- 核心识别逻辑 (复用) ---
def predict_emotion_from_face(face, frame_width, frame_height):
    # 预处理
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_input = face_resized.astype("float32")
    face_input = (face_input / 255.0) * 2 - 1
    face_input = np.expand_dims(face_input, axis=-1)
    face_input = np.expand_dims(face_input, axis=0)
    # 预测
    pred_prob = model.predict(face_input, verbose=0)
    pred_idx = int(np.argmax(pred_prob))
    return emotion_labels[pred_idx]


# --- 视频处理线程函数 ---
def process_video_background(task_id, filepath, filename):
    """后台线程：逐帧处理视频，更新进度，最后生成报告"""
    global processing_tasks

    try:
        f, writer = init_csv()
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0: total_frames = 1  # 防止除以0

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1

            # 每处理 5 帧更新一次进度，减少锁竞争，也让前端不用太频繁刷新
            if frame_count % 5 == 0 or frame_count == total_frames:
                progress = int((frame_count / total_frames) * 100)
                processing_tasks[task_id]['progress'] = progress

            # --- 识别逻辑 ---
            # 简单跳帧优化：每2帧处理一次，加快速度
            if frame_count % 2 != 0: continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_frame)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 模拟时间轴

            for res in results:
                x1, y1, width, height = res["box"]
                face = frame[max(0, y1):min(frame.shape[0], y1 + height), max(0, x1):min(frame.shape[1], x1 + width)]
                if face.size == 0: continue

                try:
                    emotion = predict_emotion_from_face(face, frame.shape[1], frame.shape[0])
                    writer.writerow([now_str, emotion])
                except:
                    continue

        cap.release()
        f.close()

        # 视频处理完，开始生成报告
        processing_tasks[task_id]['status'] = 'generating'
        report_name = generate_web_report("视频开始", "视频结束", mode="video", video_path=filename)

        # 任务完成
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = report_name
        processing_tasks[task_id]['progress'] = 100

    except Exception as e:
        print(f"Error processing video: {e}")
        processing_tasks[task_id]['status'] = 'error'


# --- 路由定义 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')


def gen_frames():
    """视频流生成器（修复数据写入问题）"""
    global camera, is_recording, csv_writer, csv_file  # 引入 csv_file 用于刷新
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            # --- 核心识别逻辑 ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_frame)
            annotated_frame = frame.copy()

            current_emotion = None

            for res in results:
                x1, y1, w, h = res["box"]
                # 边界保护
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x1 + w), min(frame.shape[0], y1 + h)

                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    try:
                        # 预测情绪
                        pred = predict_emotion_from_face(face, 0, 0)
                        current_emotion = pred

                        # 画框
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100, 180, 255), 2)
                        cv2.putText(annotated_frame, pred, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 180, 255), 2)
                    except Exception as e:
                        pass  # 忽略单帧预测错误

            # --- 【修复重点】数据写入与强制刷新 ---
            if is_recording and csv_writer and current_emotion:
                try:
                    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_emotion])
                    # 强制将数据从内存刷入硬盘！
                    csv_file.flush()
                    # 可以在控制台打印一下，证明真的在写
                    # print(f"DEBUG: 写入情绪 {current_emotion}")
                except Exception as e:
                    print(f"写入CSV失败: {e}")

            # 编码图片流
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_record')
def start_record():
    global is_recording, csv_file, csv_writer, start_time
    if not is_recording:
        csv_file, csv_writer = init_csv()
        start_time = datetime.now()
        is_recording = True
    return "started"


@app.route('/stop_record')
def stop_record():
    global is_recording, csv_file, start_time

    print("DEBUG: 用户点击了停止录制")

    if not is_recording:
        print("DEBUG: 失败 - 当前状态并未在录制")
        return "error"

    # 1. 停止录制并关闭文件
    is_recording = False
    if csv_file:
        try:
            csv_file.flush()
            csv_file.close()
            csv_file = None
            print("DEBUG: CSV文件已关闭")
        except Exception as e:
            print(f"DEBUG: 关闭文件出错: {e}")
            return "error"

    # 2. 【关键】检查数据量是否足够
    try:
        data_path = os.path.join("data", "emotion_log.csv")
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            row_count = len(lines)
            print(f"DEBUG: CSV文件共有 {row_count} 行")

        if row_count < 5:  # 连表头在内少于5行（也就是只有不到4个数据点）
            print("DEBUG: 数据量太少，无法生成图表")
            return "empty"  # 返回特定状态码

    except Exception as e:
        print(f"DEBUG: 读取CSV检查失败: {e}")
        return "error"

    # 3. 开始生成报告
    end_time = datetime.now()
    try:
        print("DEBUG: 调用 generate_web_report...")
        report_name = generate_web_report(
            start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time.strftime('%Y-%m-%d %H:%M:%S'),
            mode="webcam"
        )
        print(f"DEBUG: 报告生成成功: {report_name}")
        flash("实时采集结束，报告已为您生成！", "success")
        return report_name
    except Exception as e:
        # 这里会打印出具体的 Python 报错信息，非常重要！
        print(f"ERROR: 分析脚本崩溃! 详细错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整报错堆栈
        return "error"

# --- 视频上传与进度条相关路由 ---

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """接收文件，启动后台线程，返回任务ID"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    filename = file.filename
    # 确保 data 目录存在
    if not os.path.exists('data'): os.makedirs('data')
    filepath = os.path.join('data', filename)
    file.save(filepath)

    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    processing_tasks[task_id] = {
        'progress': 0,
        'status': 'queued',
        'result': None
    }

    # 启动后台线程
    thread = threading.Thread(target=process_video_background, args=(task_id, filepath, filename))
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/progress/<task_id>')
def get_progress(task_id):
    """前端轮询此接口获取进度"""
    task = processing_tasks.get(task_id)
    if task:
        return jsonify(task)
    else:
        return jsonify({'error': 'Task not found'}), 404


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/report/<filename>')
def report(filename):
    return render_template('report.html', filename=filename)


@app.route('/download_pdf/<path:filename>')
def download_pdf(filename):
    # 1. 获取 result 文件夹的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, 'result')

    # 2. 打印调试信息到控制台（请在运行时的黑色窗口里看这个信息）
    print("-" * 30)
    print(f"DEBUG: 正在请求下载文件: {filename}")
    print(f"DEBUG: Flask 寻找目录: {directory}")
    full_path = os.path.join(directory, filename)
    print(f"DEBUG: 完整路径检查: {full_path}")
    print(f"DEBUG: 文件是否存在? {os.path.exists(full_path)}")
    print("-" * 30)

    try:
        # as_attachment=True 会强制浏览器弹出“保存”对话框
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        # 如果出错，会在网页上直接显示具体的错误原因，而不是 Generic 的 404
        return f"下载出错: {str(e)}。服务器在 {directory} 下没找到 {filename}", 404


# --- 【关键修改】图片预览路由 ---
@app.route('/result_image/<path:filename>')
def result_image(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, 'result')
    return send_from_directory(directory, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)