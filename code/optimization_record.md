STEP1：  
1.使用了验证集（不再是训练集中取10%）  
2.进行了数据增强，识别放缩，旋转的人脸  
datagen = ImageDataGenerator(
    rotation_range=30,        # 旋转±30度
    width_shift_range=0.2,    # 水平平移
    height_shift_range=0.2,   # 垂直平移
    shear_range=0.2,          # 剪切变换（模拟歪头）
    zoom_range=0.2,           # 随机缩放
    horizontal_flip=True,     # 水平翻转
    fill_mode='nearest'
)  
3.增加CNN架构的滤波器（翻倍），增强特征提取能力和模型容量
