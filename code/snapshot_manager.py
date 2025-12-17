import os
import cv2


class SnapshotManager:
    def __init__(self, save_dir="../result/snapshots"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # 记录每种情绪当前的最高置信度，格式: {'Happy': 0.85, ...}
        self.best_confidences = {}
        # 记录保存的文件路径
        self.saved_files = {}

    def update(self, frame, emotion, confidence, face_box):
        """
        检查当前帧是否是该情绪的“最佳时刻”，如果是则保存截图
        """
        # 置信度太低的不保存
        if confidence < 0.5:
            return

        # 如果当前置信度 > 历史最高记录，则更新
        if confidence > self.best_confidences.get(emotion, 0.0):
            self.best_confidences[emotion] = confidence

            # 截取人脸 (稍微留一点边距)
            x, y, w, h = face_box
            h_img, w_img, _ = frame.shape
            pad = int(w * 0.2)  # 20% padding

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                filename = f"best_{emotion}.jpg"
                path = os.path.join(self.save_dir, filename)
                cv2.imwrite(path, face_img)
                self.saved_files[emotion] = path

    def get_summary_images(self):
        """
        返回所有已保存的最佳截图路径列表，用于生成报告
        """
        return self.saved_files