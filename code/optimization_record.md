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
![img](images/step1.jpg)  
学习效果：  
训练准确率: 从20%上升到55%  
验证准确率: 从20%上升到50%  
最终准确率: 训练集55%，验证集50%  
非常糟糕，基本没学会，欠拟合  

STEP2:
1.利用残差网络优化原来的简单CNN模型，提高模型对于精细特征提取能力；
2.增加了学习率调整，前期速率较大，后期逐步减小便于精细优化；
3.增加了早停机制，更加有效地防止过拟合现象；
4.增加测试集作为最终测试，便于更直接反应结果；
5.在数据增强部分增加光线变化，有利于弥补FER2013数据集光线条件较少对真实生活模拟度较低的情况。
学习效果如下图：
<img width="1400" height="600" alt="training_curve_v2" src="https://github.com/user-attachments/assets/c22ddf93-49ec-42fb-a55a-8be34a692914" />
总结：
准确率相对提升（FER2013的人类识别准确率在65%-72%之间）。
后续优化想法：1.增加注意力机制2.针对FER2013的特点对厌恶等情绪进行权重调整3.用FER+作为升级版数据集
