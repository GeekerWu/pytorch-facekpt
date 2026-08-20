# 🖼️ 模块文档：模型架构 (Model Architecture)

## 📖 模块概述 (Module Overview)
`model.py` 定义了本项目的核心网络结构——人脸关键点检测模型。它是一个基于预训练的 ResNet-18 骨干网络，但经过了三次关键的修改，使其功能从通用的图像分类网络，成功重构为一个用于**关键点坐标回归 (Keypoint Regression)** 的网络。

## 🧱 核心组件：`Network` 类
该类继承自 `nn.Module`，定义了模型的前向传播逻辑。

### 1. 结构继承与定制 (Structure Inheritance & Customization)
模型基于 `torchvision.models.resnet18`，并通过以下三个关键修改适应本任务：

*   **输入通道修改 (Input Channel)**:
    *   **挑战**: 原始 ResNet 默认期望 3 通道输入（RGB）。
    *   **解决方案**: 将 `conv1` 的输入通道数硬编码为 **1** (`nn.Conv2d(1, 64, ...)`), 以匹配 `dataset.py` 中预处理的单通道灰度图。
*   **输出维度修改 (Output Dimension)**:
    *   **挑战**: 原始模型用于分类，输出维度很高（如 1000）。
    *   **解决方案**: 将全连接层 (`self.model.fc`) 的输出维度修改为 `num_classes`（默认值 **136**）。$136$ 对应 $68 \text{个关键点} \times 2 \text{维}(\text{x, y})$ 的坐标维度。
*   **预训练状态控制**: 在初始化时设置 `pretrained=False`，确保了在无网络连接的情况下，代码仍能成功运行和测试。

### 2. 前向传播 (`forward` 方法)
*   **输入**: 接收一个 PyTorch 张量 $x$，期望的形状是 `(BatchSize, 1, C, H, W)`。
*   **过程**: 经过整个 ResNet 结构后，最后通过 `x.view(x.size(0), -1)` 将特征图展平，输出一个 $(BatchSize, 136)$ 的向量，完美匹配回归任务的输出格式。

## ⚙️ 辅助功能：`initialize_network`
该函数负责模型的实例化和环境配置：
1.  **设备检测**: 检查 CUDA 可用性，并确保模型被移动到正确的设备（GPU 或 CPU）。
2.  **资源管理**: 返回模型对象和当前设备信息，保证了代码的跨平台性和鲁棒性。

---
**[模块总结]**：本模块是项目架构的骨架，它成功地将一个通用模型适配到了特定任务的回归目标上。