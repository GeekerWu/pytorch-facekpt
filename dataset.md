# 📊 模块文档：数据集加载器 (Dataset Loader)

## 📖 模块概述 (Module Overview)
`dataset.py` 定义了 `FaceLandmarksDataset` 类，这是一个继承自 `torch.utils.data.Dataset` 的核心数据加载器。它的职责是自动化地解析大型的、基于 XML 的数据集结构，从中提取所有样本所需的原始数据（图像路径、人脸边界框、关键点坐标），并将这些数据与 `utils.py` 中的数据增强管线 (`Transforms`) 有效绑定，最终为模型提供统一格式的、随时可用的训练样本。

## 📦 核心依赖与数据源 (Core Dependencies & Data Sources)

*   **数据源**: 从指定的 XML 文件（`labels_ibug_300W_train.xml`）读取。
*   **关键外部依赖**: `utils.py` 中的 `Transforms` 实例，用于所有样本的同步增强。
*   **数据结构**: 维护了三个核心列表：
    1.  `self.image_filenames`: 图像文件的完整路径列表。
    2.  `self.crops`: 每个样本的人脸边界框 $(\text{left}, \text{top}, \text{width}, \text{height})$ 坐标列表。
    3.  `self.landmarks`: 每个样本的 68 个关键点坐标，存储为 `float32` 的 NumPy 数组。

## 🧱 核心流程：`__init__` (数据解析阶段)
这是数据预处理的关键阶段，所有数据（路径、坐标）都在这里一次性加载到内存。

1.  **XML解析**: 解析 XML 文件，遍历所有 `<file>` 节点。
2.  **数据捕获**: 为每个文件记录：
    *   图像的文件路径。
    *   人脸的边界框坐标（Crop）。
    *   68 个关键点 $(x, y)$ 的坐标。
3.  **数据类型统一**: 将所有关键点坐标统一转换为 `float32` 类型的 NumPy 数组，保证后续运算的精度一致性。

## ♻️ 核心方法：`__getitem__` (样本获取阶段)
这是数据加载器（DataLoader）每次请求数据时调用的核心方法。

1.  **图像读取**: 使用 `cv2.imread` 读取原始图像，并指定为灰度图（`cv2.COLOR_BGR2GRAY`），以适配模型输入。
2.  **同步增强**: 调用传入的 `self.transform` 管道。这是数据增强的执行点，它负责确保图像和关键点同步经历裁剪、缩放、旋转等变换。
3.  **坐标系修正**: 在增强流程结束后，代码执行了手动坐标修正：`landmarks = landmarks - 0.5`。此步骤用于补偿潜在的全局坐标系偏移或归一化误差。
4.  **最终类型转换**: 最后，`landmarks` 被强制转换为 `torch.tensor`，确保返回的坐标张量类型一致。

---
**[模块总结]**：该模块是数据层面的核心，负责将散落在 XML 文件和磁盘上的原始信息，整合成可供模型高效学习的、经过预处理和增强的 PyTorch 数据批次。
***
**[End of Document]**