# 📚 训练脚本逻辑文档：train.py

## 📖 概述 (Overview)
`train.py` 是整个 PyTorch 人脸关键点检测项目的核心入口文件，负责协调数据集的加载、模型的初始化、优化器的配置，并管理整个多 Epoch 的训练和验证流程。其目标是最小化关键点预测的均方误差 (Mean Squared Error, MSE)。

## ⚙️ 核心依赖与模块 (Dependencies)
本脚本依赖于以下核心模块，它们的职责和功能是训练流程不可分割的一部分：

*   **`dataset.py`**: 负责数据集的实例化和数据增强管线（通过 `FaceLandmarksDataset`）。
*   **`utils.py`**: 包含 `Transforms` 类，实现图像和关键点的各种增强和几何变换（如 `resize`, `rotate`, `color_jitter`）。
*   **`model.py`**: 包含 `initialize_network` 函数，负责根据定义的类别数量（`NUM_CLASSES`）构建和返回模型结构。

## 📐 关键配置参数 (Configuration Parameters)
脚本的多个关键参数在文件顶部定义，允许用户通过修改这些常量来调整实验配置。

| 参数 | 描述 | 默认值/类型 | 作用 |
| :--- | :--- | :--- | :--- |
| `NUM_CLASSES` | 类别数量 | `136` (int) | 模型输出的维度，对应人脸关键点的数量。 |
| `NUM_EPOCHS` | 总训练轮次 | `10` (int) | 模型将迭代训练的总次数。 |
| `LEARNING_RATE` | 学习率 | `0.0001` (float) | 优化器更新权重时的步长。 |
| `IMAGE_SIZE` | 图像尺寸 | `(224, 224)` (tuple) | 模型输入图像的尺寸，所有输入都会被缩放到此尺寸。 |
| `TRAIN_VAL_RATIO` | 训练集比例 | `0.8` (float) | 用于分割数据集的比例（80% 训练，20% 验证）。 |
| `BATCH_SIZE` | 批次大小 | `32` (int) | 每次迭代处理的样本数量。 |

## 🏃‍♂️ 执行流程详解 (Execution Flow)

`train_model()` 函数严格遵循以下五个阶段的顺序执行：

### 1. 数据集初始化与分割 (Dataset Initialization)
*   **加载**: 初始化 `FaceLandmarksDataset`，并传入 `Transforms` 管道，确保加载的每个样本都经过标准的数据增强处理。
*   **分割**: 根据 `TRAIN_VAL_RATIO`，使用 `torch.utils.data.random_split` 将总数据集随机划分为训练集和验证集。
*   **DataLoader**: 创建 `DataLoader` 对象，用于批量（Batch）高效地将数据送入训练循环。*(注：代码中设置了 `num_workers=0`，这是确保多进程环境下数据加载不会引发类型错误的最佳实践)*。

### 2. 模型和优化器配置 (Model & Optimizer Setup)
*   **网络构建**: 调用 `initialize_network()` 函数，根据 `NUM_CLASSES` 实例化模型。
*   **损失函数**: 使用 `nn.MSELoss()` (Mean Squared Error Loss)，这表明任务是回归问题，模型目标是最小化关键点预测与真实关键点之间的均方误差。
*   **优化器**: 使用 `optim.Adam()` 优化器，并传入学习率 `LEARNING_RATE`。

### 3. 训练主循环 (Training Loop: Epoch by Epoch)
整个过程包含两个子阶段，并在每个 Epoch 结束时进行汇总和对比。

#### **A. 训练模式 (`network.train()`):**
*   **目标**: 更新模型权重。
*   **过程**:
    1.  `optimizer.zero_grad()`: 清零上一步计算的梯度。
    2.  **前向传播**: 将当前批次 `images` 和 `landmarks` 输入模型，得到预测结果 `predictions`。
    3.  **计算损失**: 使用 `criterion(predictions, landmarks)` 计算当前批次的损失。
    4.  **反向传播**: `loss_train_step.backward()` 计算损失相对于所有参数的梯度。
    5.  **参数更新**: `optimizer.step()` 根据计算出的梯度，按学习率更新模型的所有权重。

#### **B. 验证模式 (`network.eval()`):**
*   **目标**: 评估模型在未见过的数据上的泛化能力，不进行权重更新。
*   **过程**:
    1.  使用 `with torch.no_grad()` 上下文块，禁用梯度计算，以节省内存和计算资源。
    2.  进行前向传播和损失计算，得出验证损失。

### 4. 性能监控与保存 (Evaluation & Checkpointing)
*   **记录**: 在每个 Epoch 结束时，脚本会打印出当前 Epoch 的训练损失和验证损失。
*   **模型保存**: 实现了**基于验证损失的早停和保存机制**。如果当前 Epoch 的验证损失 (`loss_valid`) 小于历史记录的最小验证损失 (`loss_min`)，说明模型性能得到提升，脚本会立即将当前的模型状态 (`network.state_dict()`) 保存到 `face_landmarks.pth` 文件中。

## 🐛 关键注意事项与潜在问题 (Notes & Caveats)
1.  **设备管理**: 脚本首先检查 CUDA 设备是否可用。如果不可用，会使用 CPU，并发出警告（这会显著降低训练速度）。
2.  **数据加载 (修复点)**: 代码内加入了 `num_workers=0` 的设置，这是一个针对数据加载器常见多进程错误的安全修复。
3.  **损失函数**: 使用 MSE Loss 确认这是典型的回归任务，意味着模型预测的是连续的、具有空间坐标的数值。

***
**[End of Document]**