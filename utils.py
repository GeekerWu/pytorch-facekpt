import torch
import torchvision.transforms as transforms
from torchvision import transforms as TF
import numpy as np
from PIL import Image
import math
import imutils
import random
from typing import Union

# Define type aliases for clarity (following the doc's mixed types)
ImageInput = Union[Image.Image, np.ndarray]
TensorInput = torch.Tensor
LandmarkInput = Union[np.ndarray, torch.Tensor]

class MathUtils:
    @staticmethod
    def radians(degrees: float) -> float:
        """将度数制转换为弧度制。"""
        return math.radians(degrees)

    @staticmethod
    def cos(degrees: float) -> torch.Tensor:
        """计算给定角度的余弦值，以 PyTorch 张量形式返回。"""
        # 假设输入是一个角度值或包含角度值的张量
        return torch.cos(torch.tensor(degrees, dtype=torch.float32))

    @staticmethod
    def sin(degrees: float) -> torch.Tensor:
        """计算给定角度的正弦值，以 PyTorch 张量形式返回。"""
        return torch.sin(torch.tensor(degrees, dtype=torch.float32))

class Transforms():
    def __init__(self):
        pass

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)
        transformation_matrix = torch.tensor([[+MathUtils.cos(MathUtils.radians(angle)), -MathUtils.sin(MathUtils.radians(angle))],
                                             [+MathUtils.sin(MathUtils.radians(angle)), +MathUtils.cos(MathUtils.radians(angle))]
                                         ])
        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = image.resize(img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = TF.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = image.crop((left, top, left + width, top + height))

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5], std=[0.5])(image)
        return image, landmarks
