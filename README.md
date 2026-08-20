# 🖥️ PyTorch 人脸关键点检测系统 (Face Keypoint Detection System)

## 📜 项目描述 (Project Description)
本项目是一个基于 PyTorch 框架的深度学习系统，用于实现高精度的**人脸关键点（Landmarks）检测**。系统能够接收一张包含人脸的图像，通过训练流程学习人脸的关键特征，并在推理阶段准确地定位并输出人脸上的 68 个关键点（如眼睛、鼻子、嘴角等）的像素级坐标。

该系统流程设计严谨，分为数据加载、模型训练、模型推理三个阶段，并通过 `utils.py` 统一管理复杂的图像几何变换和坐标同步，确保了整个管道的鲁棒性和准确性。

## 🌳 项目文件结构 (Project Structure)

```
pytorch-facekpt/
├── train.py              # [核心] 主训练入口：控制整个训练流程的执行。
├── predict.py            # [核心] 主推理入口：执行模型预测和可视化展示。
├── dataset.py            # 数据加载：负责解析原始 XML 数据集，并创建 DataLoader。
├── utils.py              # 增强工具：包含所有数据增强（旋转、裁剪、缩放）和数学辅助函数。
├── model.py              # 模型定义：定义了基于 ResNet-18 的回归网络结构。
├── requirements.txt      # 运行时依赖包列表。
├── face_landmarks.pth    # 训练产物：训练过程中保存的最佳模型权重。
    通过网盘分享的文件：face_landmarks.pth
    链接: https://pan.baidu.com/s/15qsZjsYumujQy5fYcuMYPA 提取码: 6666 
        --来自百度网盘超级会员v4的分享
└── dlib_dataset/         # 原始数据集：包含 XML 标签和图像文件。
        训练用数据集
        http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
```

## 🚀 运行前准备 (Prerequisites)

### 1. 环境依赖安装
请根据 `requirements.txt` 文件安装所需所有依赖项。
```bash
# 强烈推荐使用虚拟环境
pip install -r requirements.txt
```

### 2. 依赖文件准备
请确保以下资源文件已在项目根目录下就位：
*   **数据集**: 完整的 `dlib_dataset/` 文件夹及其内部的 XML 标签和图像文件。
*   **人脸检测模型**: `haarcascade_frontalface_default.xml` (OpenCV 依赖)。
*   **待测图像**: `single.jpg` (用于推理测试)。

## <0xF0><0x9F><0x8E><0x9B>️ 核心功能模块说明 (Module Functionality)

### 1. `dataset.py`：数据层
*   **功能**: 负责数据的 I/O 和预处理。它解析 XML 文件，定位原始图像和 68 个关键点，并实例化 `DataLoader`，将所有数据准备好送入训练流。
*   **核心逻辑**: 严格执行 $\text{Raw Data} \xrightarrow{\text{Parsing}} \text{Internal State} \xrightarrow{\text{Transforms}} \text{Batch Data}$ 的转换过程。

### 2. `utils.py`：增强与几何变换层
*   **功能**: 集中管理所有数据增强逻辑。它保证了图像、关键点在所有变换（如旋转、缩放、裁剪）中始终保持**几何和坐标系的一致性**。
*   **关键点**: $\text{Transform} \Rightarrow \text{Coordinates}$. (坐标的几何变换必须与图像同步)。

### 3. `model.py`：网络架构层
*   **功能**: 定义了模型的网络拓扑结构。它将标准的 ResNet-18 分支修改为回归任务所需的结构：
    *   输入层修改为 $1 \text{ Channel}$ (适应灰度图)。
    *   输出层修改为 $136 \text{ Dimensions}$ ($68 \text{ Landmarks} \times 2$)。

### 4. `train.py`：训练执行层 (Training Execution)
*   **功能**: 协调整个学习过程。它通过 PyTorch 的训练循环 (`optimizer.zero_grad() -> forward -> backward() -> optimizer.step()`) 实现权重更新。
*   **流程亮点**: 实现了**基于验证损失的自动模型权重保存**（Checkpointing），只保留性能最佳的模型权重到 `face_landmarks.pth`。

### 5. `predict.py`：推理执行层 (Inference Execution)
*   **功能**: 部署阶段的主入口。它使用加载的模型权重，接收一张新的图片，并在图像上逐步定位人脸，进行关键点预测，并将结果可视化。
*   **核心步骤**: 执行了从 $2D$ 图像 $\rightarrow$ 人脸检测 $\rightarrow$ 预处理 $\rightarrow$ 归一化 $\rightarrow$ 预测 $\rightarrow$ **反向坐标还原** 的完整流程。

## ⚙️ 运行示例 (Usage Examples)

### 🎬 1. 训练模型 (Training)
这是项目的构建阶段。

**命令**:
```bash
python train.py
```
**参数说明**:
*   **无额外参数**: 脚本会使用 `train.py` 中定义的全局配置（如 `NUM_EPOCHS=10`，`LEARNING_RATE=0.0001`）。
*   **自定义 Epoch**: 如需修改训练轮次，请直接修改代码顶部的 `NUM_EPOCHS` 变量。

### 🔎 2. 模型推理 (Inference/Testing)
这是项目实际应用阶段。

**命令**:
```bash
python predict_CV.py
python predict.py
```
**参数说明**:
*   **输入图像**: 脚本默认加载 `single.jpg` 进行测试。如需更改测试图，请修改代码顶部的 `image_path` 变量。
*   **模型权重**: 使用训练得到的 `face_landmarks.pth` 进行推理。

---
**[文档更新完成]**