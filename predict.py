import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle # 导入关键：用于绘制边界框
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF # 导入关键：Image.fromarray 依赖 PIL 的 Image
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

#######################################################################
# image_path = 'single.jpg'
image_path = 'multiple.jpg'

weights_path = 'face_landmarks.pth'

# 然后，输入 pip show opencv-python查看包的位置
# 最后找到这个位置打开键入data目录打开就会发现haarcascade各种分类器文件都在这里面

frontal_face_cascade_path = 'D:/pytorch-facekpt/haarcascade_frontalface_default.xml'
#######################################################################

class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=False)
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)

    def forward(self, x):
        x=self.model(x)
        return x

#######################################################################
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

best_network = Network()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
best_network.eval()

image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width,_ = image.shape
print(f"图像尺寸: 宽度={width}, 高度={height}")

# faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4) # 数字越大截取面积越大，过大会导致识别出多张人脸
faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)
if len(faces) > 0:
# # 绘制原始人脸检测的边界框
    plt.figure()
    plt.imshow(display_image)
    for (x, y, w, h) in faces:
        # 创建矩形补丁，颜色设置为红色，无填充，用 linewidth 模拟边框
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
    # 结束绘制边界框
    # plt.show()

all_landmarks = []
for (x, y, w, h) in faces:
    # 关键点预测的核心步骤：将OpenCV的Numpy数组转换为PIL Image对象
    #  裁剪面部
    face_crop_np = grayscale_image[y:y+h, x:x+w]
    # plt.figure()
    # plt.imshow(face_crop_np)

    image_pil = Image.fromarray(face_crop_np) # 修复 NameError

    image_resized = TF.resize(image_pil, size=(224, 224))
    image_tensor = TF.to_tensor(image_resized)
    print(f"裁剪后图像尺寸: {image_tensor.shape}")  # 打印裁剪后图像的尺寸
    image_normalized = TF.normalize(image_tensor, [0.5], [0.5])
    print(f"归一化后图像: {image_normalized.shape}")  # 打印归一化后图像的尺寸

    with torch.no_grad():
        landmarks = best_network(image_normalized.unsqueeze(0))
    print(f"裁剪区域坐标: x={x}, y={y}, w={w}, h={h}")  # 打印裁剪区域的坐标和大小
    print(f"不加上偏移量关键点坐标: {(landmarks.view(68,2).detach().numpy() + 0.5)}")  # 打印预测关键点坐标不加上偏移量
    # print(f"不加上偏移量关键点坐标_shape: {(landmarks.view(68,2).detach().numpy() + 0.5).shape}")  # 打印预测关键点坐标
    # print(f"不加上偏移量关键点坐标*wh_shape: {(landmarks.view(68,2).detach().numpy() + 0.5)*np.array([[w, h]])}")  # 打印预测关键点坐标不加上偏移量
    
    # new_landmarks = (landmarks.view(68,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
    new_landmarks = (landmarks.view(68,2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
    print(f"加上偏移量关键点坐标: {new_landmarks[0]}")  # 打印第一个关键点的坐标
    all_landmarks.append(new_landmarks)

# 绘制预测的关键点
plt.figure()
plt.imshow(display_image)
for landmarks in all_landmarks:
    plt.scatter(landmarks[:,0], landmarks[:,1], c = 'c', s = 5)

plt.show()