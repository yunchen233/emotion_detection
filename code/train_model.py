import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    Input, Add, Activation  # 新增残差连接所需层
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 新增训练调控工具
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- 1. 路径与参数配置（新增测试集路径） ---
# 获取项目根目录路径
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)

# 数据集路径：新增测试集（test.csv），实现训练-验证-测试完整流程
train_csv_path = os.path.join(project_root, 'data', 'train.csv')
val_csv_path = os.path.join(project_root, 'data', 'val.csv')
test_csv_path = os.path.join(project_root, 'data', 'test.csv')  # 新增测试集
model_save_path = os.path.join(project_root, 'model', 'emotion_model_v2.h5')

# 图像参数：48x48 灰度图
IMAGE_WIDTH, IMAGE_HEIGHT = 48, 48
NUM_CLASSES = 7  # 情绪类别：愤怒、厌恶、恐惧、开心、悲伤、惊讶、平静


# --- 2. 数据加载与预处理（完善测试集处理） ---
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

print("--- 开始加载和预处理数据 ---")
# 加载训练集
X_train, y_train = load_and_preprocess_data(train_csv_path)
# 加载验证集
X_val, y_val = load_and_preprocess_data(val_csv_path)
# 加载测试集
X_test, y_test = load_and_preprocess_data(test_csv_path)  # 新增测试集加载

# 打印数据形状，验证处理正确性
print("--- 数据加载和预处理完成 ---")
print(f"训练集: {X_train.shape} 样本，标签: {y_train.shape}")
print(f"验证集: {X_val.shape} 样本，标签: {y_val.shape}")
print(f"测试集: {X_test.shape} 样本，标签: {y_test.shape}")  # 新增测试集信息


# --- 3. 模型构建（这一部分新增残差连接，增强特征提取能力） ---
def residual_block(x, filters, kernel_size=(3, 3)):
    """
    在这里做了一步优化：不止于简单CNN模型，利用残差网络，细节特征提取能力++
    大概的原理解释如下————
    残差块：解决深层网络梯度消失问题，增强特征传递
    原理：通过跳跃连接将输入直接加到输出，让网络更易学习细微特征（如表情细节）
    """
    # 主路径：两次卷积+批归一化
    residual = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # 跳跃连接：若输入输出通道数不同，用1x1卷积调整维度
    if residual.shape[-1] != filters:
        residual = Conv2D(filters, (1, 1), padding='same')(residual)
    
    # 残差相加后激活
    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x

print("--- 开始构建增强特征提取能力的模型 ---")

    # 输入层：48x48灰度图
inputs = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    
    # 初始卷积层：提取低级特征（边缘、纹理）
x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
    
    # 残差块1：增强中级特征提取（如眼睛、嘴巴区域）
x = residual_block(x, 64)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x) # 防止过拟合(这里随机“丢弃”一些神经元)
    
    # 残差块2：提取更复杂的表情特征（如嘴角上扬、皱眉）
x = residual_block(x, 128)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
    
    # 残差块3：捕捉高级情绪特征（组合面部器官变化等等，滤波器数量增加到256）
x = residual_block(x, 256)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
    
    # 分类头：将特征映射转换为类别概率
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x) # 多分类输出

model = Model(inputs=inputs, outputs=outputs)
# 编译模型：优化器+损失函数+评估指标
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # 适合多分类的损失函数
    metrics=['accuracy']
)

# 打印模型结构，确认残差块生效
model.summary()
print("--- 模型构建完成 ---")


# --- 4. 数据增强（增强模型对细微表情变化的鲁棒性） ---
print("--- 配置增强数据生成器 ---")
datagen = ImageDataGenerator(
    rotation_range=30,         # 模拟头部旋转（±30度）
    width_shift_range=0.2,     # 水平偏移（模拟侧脸）
    height_shift_range=0.2,    # 垂直偏移
    shear_range=0.2,           # 剪切变换（模拟歪头）
    zoom_range=0.2,            # 缩放（模拟距离变化）
    horizontal_flip=True,      # 水平翻转（左右脸对称）
    brightness_range=[0.8, 1.2],  # 这里新增了一点：新增亮度调整（应对不同光照），解决FER2013数据集里样本光线变化少，不贴近真实情况
    fill_mode='nearest'        # 填充新像素的方式
)
# 备注：仅对训练集使用增强
train_generator = datagen.flow(X_train, y_train, batch_size=64)
print("--- 数据增强配置完成 ---")


# --- 5. 模型训练（这一部分增加了早停和学习率调整） ---
print("--- 开始训练模型 ---")
# 新增了早停策略：当验证集损失不再下降时停止，防止过拟合
early_stopping = EarlyStopping(
    monitor='val_loss',  # 监控验证集损失
    patience=8,          # 8个epoch无改进则停止
    restore_best_weights=True  # 恢复最优权重
)

# 新增了学习率调整：验证集损失停滞时降低学习率，精细优化。从快速优化->精细优化
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # 学习率减半
    patience=4,          # 4个epoch无改进则调整
    min_lr=1e-6          # 最小学习率
)

BATCH_SIZE = 64
EPOCHS = 100  # 最大轮次（实际会被早停截断，初次尝试在约58轮时截停）

# 开始训练
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),  # 用验证集评估泛化能力
    callbacks=[early_stopping, lr_scheduler],  # 训练调控
    shuffle=True
)
print("--- 模型训练完成 ---")


# --- 6. 模型评估与保存（这一部分新增加测试集最终评估） ---
# 创建模型保存目录
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"模型已保存到: {model_save_path}")

# 在测试集上评估
print("\n--- 测试集最终评估 ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"测试集准确率: {test_acc:.4f}，测试集损失: {test_loss:.4f}")


# --- 7. 训练曲线可视化（新增测试集指标对比） ---
def plot_training_history(history, save_path):
    plt.figure(figsize=(14, 6))
    
    # 准确率曲线（含测试集最终点）
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.scatter(len(history.history['accuracy']), test_acc, 
                color='red', s=100, label=f'test_accuracy: {test_acc:.4f}')  # 测试集点
    plt.title('model_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accurary')
    plt.legend()
    
    # 损失曲线（含测试集最终点）
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.scatter(len(history.history['loss']), test_loss, 
                color='red', s=100, label=f'test_loss: {test_loss:.4f}')  # 测试集点
    plt.title('model_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 保存训练曲线
plot_training_history(history, '../model/training_curve_v2.png')
print("训练曲线已保存到: ../model/training_curve_v2.png")