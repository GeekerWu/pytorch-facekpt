import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle # 导入关键：用于绘制边界框
from torchvision import models
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

#######################################################################
# image_path = 'single.jpg'
image_path = 'multiple.jpg'

# frontal_face_cascade_path = 'D:/pytorch-facekpt/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width,_ = image.shape
print(f"图像尺寸: 宽度={width}, 高度={height}")

# faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4) # 数字越大截取面积越大
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
    plt.show()