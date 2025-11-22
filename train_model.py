import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

# --- 1. 准备工作：定义文件路径和参数 ---      


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)
train_csv_path = os.path.join(project_root, 'data', 'train.csv')
model_save_path = os.path.join(project_root, 'model', 'emotion_model.h5')

# 图像尺寸和类别数
IMAGE_WIDTH, IMAGE_HEIGHT = 48, 48
NUM_CLASSES = 7  

# --- 2. 数据加载与预处理 ---

print("--- 开始加载和预处理数据 ---")

# 读取CSV文件
df = pd.read_csv(train_csv_path)

# 提取像素数据并转换为图像数组
# 我们需要将其分割成单个数字，然后转换为48x48的数组
pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

# 将所有图像数据堆叠成一个大的numpy数组，并添加通道维度
# 最终形状为 (样本数, 48, 48, 1)，1代表灰度图
X_train = np.stack(pixels, axis=0).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)

# 提取情绪标签
y_train = df['emotion'].values

# 将标签转换为独热编码 (One-Hot Encoding)

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

# 数据归一化：将像素值从 [0, 255] 缩放到 [-1, 1]（这一步是增加效率的）
X_train = (X_train / 255.0) * 2 - 1

print("--- 数据加载和预处理完成 ---")
print(f"训练集图像数据形状: {X_train.shape}")
print(f"训练集标签数据形状: {y_train.shape}")


# --- 3. 构建CNN模型 ---

print("--- 开始构建模型 ---")

model = Sequential([
    # 第一卷积块
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 第二卷积块
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 第三卷积块
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 分类头
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# 编译模型
# - optimizer: Adam优化器，学习率为0.001
# - loss: 多分类交叉熵损失函数，适合独热编码标签
# - metrics: 评估指标，我们关注准确率 (accuracy)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型结构摘要
model.summary()
print("--- 模型构建完成 ---")


# --- 4. 训练模型 ---                     加入已有验证集那块看不懂，直接改了，把0.1的训练数据当验证集使了之后再说吧

print("--- 开始训练模型 ---")

# 设置训练参数
BATCH_SIZE = 64
EPOCHS = 50 # 训练轮次，可以根据电脑性能和训练效果调整

# 开始训练
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1, #改在这里了，拔出来当验证集
    shuffle=True
)

print("--- 模型训练完成 ---")


# --- 5. 保存模型和训练结果 ---

# 创建模型保存目录（如果不存在）
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 保存训练好的模型
model.save(model_save_path)
print(f"模型已成功保存到: {model_save_path}")

# 绘制并保存训练曲线
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 4))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 保存训练曲线图片
plot_training_history(history, '../model/training_curve.png')
print("训练曲线已保存到: ../model/training_curve.png")