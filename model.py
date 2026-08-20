# model.py

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# 定义网络模型，用于预测68个关键点的 (x, y) 坐标
class Network(nn.Module):
    """
    基于ResNet-18的自定义模型，用于预测68个关键点的 (x, y) 坐标。
    """
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        # 核心修改点：使用 pretrained=False 来跳过网络下载，以确保在无网络环境也能运行测试流程
        self.model = models.resnet18(pretrained=False)

        # 核心修改点 1: 修改第一个卷积层，将输入通道数从 3 改为 1 (单通道灰度图)
        conv1_original = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 核心修改点 2: 修改全连接层，将输出维度设置为 136 (68个关键点 * 2维坐标)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入图像张量，形状应为 (BatchSize, 1, C, H, W)
        Returns:
            torch.Tensor: 预测的坐标张量，形状为 (BatchSize, 136)
        """
        x = self.model(x)
        # 展平输出，使其维度为 (BatchSize, 136)
        return x.view(x.size(0), -1)

def initialize_network(num_classes: int = 136):
    """
    初始化网络模型并将其移动到 CUDA 设备。

    Args:
        num_classes (int): 最终需要预测的维度数 (例如: 136)。

    Returns:
        tuple[Network, torch.device]: 初始化网络和目标设备。
    """
    network = Network(num_classes=num_classes)
    # 检查CUDA是否可用，并移动模型到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    return network, device