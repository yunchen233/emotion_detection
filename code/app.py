import os
import cv2
import csv
import threading
import time
import datetime
import shutil
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# 引入你的现有模块
from snapshot_manager import SnapshotManager
from simple_tracker import CentroidTracker
# 引入分析模块 (注意：需要确保这些文件在同一目录下或PYTHONPATH中)
import emotion_fluctuation_analysis
import emotion_transition_analysis
import api
import dyadic_analysis
import api_double
import real_time_detection as single_utils  # 用于调用生成PDF的函数
import real_time_detection_double as double_utils
import upload_video

app = Flask(__name__)

# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
db = SQLAlchemy(app)


# --- 数据库模型 ---
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mode = db.Column(db.String(50))  # 'Single', 'Double', 'Video'
    filename = db.Column(db.String(200))  # PDF或视频的文件名
    filepath = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)


# --- 全局变量与模型初始化 ---
# 为避免多线程加载模型冲突，建议在这里加载一次
print("Loading Models...")
emotion_labels = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Calm", "Contempt"]
model = load_model("../model/emotion_model_v2.h5", compile=False)  # 请确保路径正确，建议改为绝对路径或相对app.py的路径
detector = MTCNN()
print("Models Loaded.")


# 状态控制
class VideoCamera(object):
    def __init__(self, mode='single'):
        self.video = cv2.VideoCapture(0)
        self.mode = mode
        self.is_recording = False
        self.start_time = None
        self.csv_file = None
        self.csv_writer = None
        self.snapshot_manager = SnapshotManager() if mode == 'single' else None
        self.tracker = CentroidTracker(maxDisappeared=20) if mode == 'double' else None
        self.colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0)]

        # 数据路径配置
        self.data_dir = "../data"
        self.result_dir = "../result"
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        if not os.path.exists(self.result_dir): os.makedirs(self.result_dir)

        # 单人/双人特定 CSV 路径
        if self.mode == 'single':
            self.csv_path = os.path.join(self.data_dir, "emotion_log.csv")
        else:
            self.csv_path = os.path.join(self.data_dir, "double_emotion_log.csv")

    def __del__(self):
        self.video.release()
        if self.csv_file:
            self.csv_file.close()

    def start_recording(self):
        self.is_recording = True
        self.start_time = datetime.datetime.now()
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        if self.mode == 'single':
            self.csv_writer.writerow(["time", "emotion"])
        else:
            self.csv_writer.writerow(["time", "face_id", "emotion"])

        # 重置抓拍器
        if self.mode == 'single':
            self.snapshot_manager = SnapshotManager()

    def stop_recording(self):
        self.is_recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        return self.start_time

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 检测逻辑 ---
        if self.mode == 'single':
            results = detector.detect_faces(rgb_frame)
            for res in results:
                x, y, w, h = res["box"]
                if x < 0 or y < 0: continue

                face = frame[y:y + h, x:x + w]
                if face.size == 0: continue

                # 预处理与预测
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_input = (face_resized.astype("float32") / 255.0) * 2 - 1
                face_input = np.expand_dims(np.expand_dims(face_input, -1), 0)

                probs = model.predict(face_input, verbose=0)[0]
                label_idx = np.argmax(probs)
                label = emotion_labels[label_idx]
                conf = probs[label_idx]

                # 记录与抓拍
                if self.is_recording:
                    self.csv_writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label])
                    self.snapshot_manager.update(frame, label, conf, (x, y, w, h))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(conf * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

        elif self.mode == 'double':
            results = detector.detect_faces(rgb_frame)
            rects = [res["box"] for res in results]
            objects = self.tracker.update(rects)

            for obj_id, (box, centroid) in objects.items():
                x, y, w, h = box
                if x < 0 or y < 0: continue
                face = frame[y:y + h, x:x + w]
                if face.size == 0: continue

                try:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    face_input = (face_resized.astype("float32") / 255.0) * 2 - 1
                    face_input = np.expand_dims(np.expand_dims(face_input, axis=-1), axis=0)

                    probs = model.predict(face_input, verbose=0)[0]
                    label_idx = np.argmax(probs)
                    label = emotion_labels[label_idx]
                    conf = probs[label_idx]

                    if self.is_recording:
                        self.csv_writer.writerow(
                            [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), obj_id, label])

                    color = self.colors[obj_id % len(self.colors)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"ID:{obj_id} {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except:
                    pass

        # 编码为JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


# 全局摄像头对象
camera = None


# --- 路由 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/monitor/<mode>')
def monitor(mode):
    global camera
    if camera is not None:
        del camera
    camera = VideoCamera(mode=mode)
    return render_template('monitor.html', mode=mode)


@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')


@app.route('/history')
def history():
    # 获取最近50条
    records = History.query.order_by(History.timestamp.desc()).limit(50).all()
    return render_template('history.html', records=records)


@app.route('/video_feed')
def video_feed():
    def gen(camera):
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    global camera
    if camera:
        camera.start_recording()
        return jsonify({"status": "started"})
    return jsonify({"status": "error"}), 400


@app.route('/api/stop_and_generate', methods=['POST'])
def stop_and_generate():
    global camera
    if not camera:
        return jsonify({"status": "error"}), 400

    start_time = camera.stop_recording()
    end_time = datetime.datetime.now()

    # --- 生成报告逻辑 ---
    # 定义结果文件名和路径
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if camera.mode == 'single':
        # 1. 波动分析
        try:
            emotion_fluctuation_analysis.main()
        except:
            pass
        # 2. 转移矩阵
        try:
            trans_img = emotion_transition_analysis.analyze_and_plot()
        except:
            trans_img = ""
        # 3. LLM API
        try:
            api.analyze_emotion_from_csv()
        except:
            pass
        # 4. 生成PDF
        snaps = camera.snapshot_manager.get_summary_images()
        # 这里需要调用 single_utils 里的 generate_pdf_report，但需要修改该函数使其不直接输出，或者我们复制文件
        # 为了方便，我们假设原函数生成在 ../result/Emotion_Report.pdf
        # 我们重新计算帧数（简化处理，或者在VideoCamera里计数）
        frame_count = 0  # 简化
        single_utils.generate_pdf_report(start_time, end_time, snaps, "../result", trans_img, frame_count)

        src_pdf = "../result/Emotion_Report.pdf"
        final_filename = f"Single_Report_{timestamp_str}.pdf"

    else:  # Double
        result_dir = "../result"
        # 1. 仪表盘
        dashboard_path = None
        try:
            res = dyadic_analysis.analyze_dyadic_relationship(result_dir)
            if res: dashboard_path, _, _ = res
        except:
            pass
        # 2. LLM API
        try:
            api_double.analyze_emotion_from_csv()
        except:
            pass
        # 3. PDF
        double_utils.generate_pdf_report(start_time, end_time, [0, 1], result_dir, dashboard_path)

        src_pdf = os.path.join(result_dir, "Double_emotion_Report.pdf")
        final_filename = f"Double_Report_{timestamp_str}.pdf"

    # --- 移动文件到 static/results ---
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])

    dst_path = os.path.join(app.config['RESULT_FOLDER'], final_filename)
    if os.path.exists(src_pdf):
        shutil.move(src_pdf, dst_path)

        # 写入数据库
        record = History(mode=camera.mode.capitalize(), filename=final_filename, filepath=dst_path)
        db.session.add(record)
        db.session.commit()

        return jsonify({"status": "success", "file_url": f"/static/results/{final_filename}"})
    else:
        return jsonify({"status": "failed", "message": "PDF generation failed"}), 500


@app.route('/api/upload_video', methods=['POST'])
def handle_upload():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{timestamp_str}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(input_path)

        # 处理视频
        output_filename = f"Labeled_{save_name}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

        try:
            # 调用你的 process_video
            upload_video.process_video(input_path, output_path)

            # 记录数据库
            record = History(mode='Video Upload', filename=output_filename, filepath=output_path)
            db.session.add(record)
            db.session.commit()

            return jsonify({"status": "success", "file_url": f"/static/results/{output_filename}"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # 清理旧的记录如果超过50条 (可选)
    app.run(debug=True, threaded=True)