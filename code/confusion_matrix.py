import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.models import load_model

IMAGE_WIDTH, IMAGE_HEIGHT = 48, 48
NUM_CLASSES = 8  # 情绪类别：愤怒、厌恶、恐惧、开心、悲伤、惊讶、平静,新增：轻蔑
# 获取项目根目录路径
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)
test_csv_path = os.path.join(project_root, 'data', 'test.csv')
model = load_model(os.path.join(project_root, 'model', 'emotion_model_v2.h5'))
# 加载测试集
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    # 处理像素数据：将空格分隔的字符串转换为48x48x1的灰度图
    pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.stack(pixels, axis=0).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    # 处理标签
    y = to_categorical(df['emotion'].values, num_classes=NUM_CLASSES)
    # 归一化
    X = (X / 255.0) * 2 - 1

    return X, y
X_test, y_test = load_and_preprocess_data(test_csv_path)
# 1. 模型预测测试集
y_test_pred = model.predict(X_test)  # 预测结果是独热编码（形状：[样本数, 8]）

# 2. 把独热编码转成整数标签（真实标签+预测标签）
y_test_true = np.argmax(y_test, axis=1)  # 真实类别（0-7的整数）
y_test_pred = np.argmax(y_test_pred, axis=1)  # 预测类别（0-7的整数）

# 3. 定义你的情绪类别名称（替换成你实际的8类标签，比如）
emotion_labels = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'calm', 'contempt']

# 生成混淆矩阵（行：真实类别，列：预测类别）
cm = confusion_matrix(y_test_true, y_test_pred)

# 画图（让矩阵更直观）
plt.figure(figsize=(10, 8))  # 设置图的大小
sns.heatmap(
    cm,  # 混淆矩阵数据
    annot=True,  # 显示每个单元格的数值
    fmt='d',  # 数值格式为整数
    cmap='Blues',  # 颜色主题
    xticklabels=emotion_labels,  # 列标签：预测类别
    yticklabels=emotion_labels   # 行标签：真实类别
)

# 添加标题和标签
plt.title('emotion_detection_model - test_confusion_matrix', fontsize=14)
plt.xlabel('predicted category', fontsize=12)
plt.ylabel('real category', fontsize=12)
plt.xticks(rotation=45)  # 旋转列标签，避免重叠
plt.tight_layout()  # 自动调整布局
plt.savefig('../result/confusion_matrix.png')  # 保存图片
plt.show()

print('情绪分类模型 - 测试集分类报告：\n')
print(classification_report(
    y_test_true,
    y_test_pred,
    target_names=emotion_labels  # 显示类别名称
))