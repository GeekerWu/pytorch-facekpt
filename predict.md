# 🚀 模块文档：模型推理引擎 (Inference Engine)

## 📖 模块概述 (Module Overview)
`predict.py` 是整个项目部署和推理的入口点。它负责将训练阶段学到的模型能力，应用于一张新的、未经处理的图像（如 `head.jpg`）上，实现**从像素到关键点坐标的落地预测**。其核心工作流包括人脸的检测、数据预处理、模型推理和坐标系反向转换，最终输出可视化结果。

## 🏗️ 流程依赖 (Workflow Dependencies)
该模块的运行严格依赖于三个前置阶段的产出：

1.  **模型结构**: 依赖 `model.py` 定义的 `Network` 架构。
2.  **模型权重**: 依赖 `train.py` 训练并保存的 `face_landmarks.pth` 权重文件。
3.  **预处理标准**: 依赖 `utils.py` 和 `dataset.py` 统一的预处理流程（如 `224x224` 缩放、`[0.5], [0.5]` 归一化），确保输入的张量格式完全一致。

## 🎯 核心推理流程 (Core Inference Pipeline)

`predict.py` 遵循一个严格的四阶段流程：

### 1. 人脸检测 (Face Detection)
*   **工具**: 使用 OpenCV 的 Haar Cascade 分类器 (`face_cascade`)。
*   **输入**: 原始的彩色输入图像，先转换为灰度图。
*   **输出**: 一个包含所有检测到人脸的边界框坐标列表 $\left(x, y, w, h\right)$。

### 2. 样本预处理 (Sample Preprocessing)
对每一个检测到的人脸区域执行以下操作：
*   **裁剪**: 从灰度图中裁剪出人脸 ROI。
*   **缩放与格式转换**: 使用 `torchvision.transforms` 将裁剪的图像强制缩放到模型预期的输入尺寸 $(224 \times 224)$，并转换为归一化后的 PyTorch Tensor。

### 3. 模型推理 (Model Inference)
*   **执行**: 在 `with torch.no_grad():` 上下文块内进行前向传播。
*   **关键步骤**: 将预处理后的张量送入模型，获取原始的、归一化后的关键点坐标张量 $\text{Landmarks}_{raw}$ (维度为 $B \times 136$)。

### 4. 后处理与可视化 (Post-processing & Visualization)
这是将模型输出转化为人类可读结果的最复杂步骤，它负责执行**坐标系逆变换**：
1.  **重塑**: 将 $136$ 维的张量重塑为 $68 \times 2$ 的二维坐标系。
2.  **反向坐标转换（关键公式）**: 预测的坐标 $\text{Landmarks}_{raw}$ 必须经过以下三步逆变换，恢复到原始大图的像素坐标 $\text{Landmarks}_{final}$：
    $$\text{Landmarks}_{final} = (\text{Landmarks}_{raw} + 0.5) \times \text{ScaleFactor} + \text{Offset}$$
    其中 $\text{ScaleFactor}$ 来自于人脸尺寸 $(w, h)$，$\text{Offset}$ 来自于人脸框的左上角 $(x, y)$。
3.  **显示**: 最后，使用 Matplotlib 绘制原始图像，并在其上用散点图标出所有反向转换得到的关键点。

---
**[模块总结]**：该模块是整个系统的**消费端**，它封装了从“像素识别”到“人工可读结果”的全过程，完美体现了整个项目流水线的可运行性。
***
**[End of Document]**