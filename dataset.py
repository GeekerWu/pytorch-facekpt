# dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

# 导入 utils.py 中的 Transforms 类，它包含了数据增强逻辑
from utils import Transforms

class FaceLandmarksDataset(Dataset):
    """
    自定义数据集类，用于加载人脸图像和关键点坐标。
    """
    def __init__(self, transform=None):
        # ==============================================================
        # 🚨 路径已根据用户反馈修正：数据根目录为 dlib_dataset
        # 🚨 假设完整的路径结构是：dlib_dataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml
        # ==============================================================
        xml_path = 'dlib_dataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml'
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except FileNotFoundError:
            print(f"错误: 无法找到XML文件路径: {xml_path}")
            raise

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'dlib_dataset/ibug_300W_large_face_landmark_dataset'

        print("开始解析数据集...")

        # 解析XML文件中的所有文件记录
        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            # 存储边界框 (Crops)
            self.crops.append(filename[0].attrib)

            # 解析 68 个关键点
            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        # 校验数据完整性
        if len(self.image_filenames) != len(self.landmarks):
            raise ValueError("图片文件名数量和关键点数量不匹配，数据解析失败。")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # 1. 读取图像 (灰度图: 0)
        image = cv2.imread(self.image_filenames[index], 0)

        # 2. 获取关键点
        landmarks = self.landmarks[index]

        # 3. 应用转换
        if self.transform:
            # 调用 utils.py 中的 Transforms 类
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        # 4. 最终坐标系处理 (调整到 PyTorch/模型期望的范围)
        landmarks = landmarks - 0.5

        # 🌟 终极修复：强制转换，确保返回的 landmarks 始终是 torch.Tensor
        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        # 返回的 image 应该是 PIL Image 类型，landmarks 是 torch.Tensor
        return image, landmarks