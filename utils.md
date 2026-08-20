### `utils.py` 代码逻辑文档 (整合版)

#### 🔬 概述 (Overview)

`utils.py` 是项目级的数据处理工具箱。它封装了两个核心模块：`MathUtils`（数学辅助函数）和 `Transforms` 类（数据增强管道）。它负责实现数据预处理的所有复杂步骤，确保了数据加载的一致性和高质量。该模块是 `dataset.py` 等上层模块进行数据增强和格式转换的底层依赖。

#### 🛠️ 核心模块一：`MathUtils` (数学工具类)

该类提供了基本的三角函数计算，但其核心价值在于能将数学计算结果直接封装为 PyTorch 张量，保证了与后续深度学习框架的兼容性。

*   **`radians(degrees: float)`:**
    *   **功能:** 标准的度数转弧度转换。
    *   **实现:** `math.radians(degrees)`。
*   **`cos(degrees: float)` / `sin(degrees: float)`:**
    *   **功能:** 分别计算给定角度的余弦和正弦值。
    *   **实现细节:** 它们将输入的角度值转换为 `torch.tensor`，然后使用 `torch.cos`/`torch.sin` 计算，确保返回类型为 `torch.Tensor`，可以直接用于张量数学运算。

#### 🌐 核心模块二：`Transforms` (数据增强管道类)

这是数据增强的核心实现，所有变换都在此管道中按顺序执行。

1.  **`crop_face(image, landmarks, crops)`:**
    *   **功能:** 裁剪出人脸区域 (ROI) 并同步处理关键点坐标系。
    *   **流程:**
        *   使用 PIL 的 `crop` 方法裁剪图像。
        *   **坐标转换 (去偏移):** 将关键点坐标减去边界框的左上角偏移量 $[left, top]$，使坐标的原点变为裁剪人脸的左上角。
        *   **坐标归一化:** 将得到的相对坐标除以裁剪后图像的宽度和高度 $[img\_shape[1], img\_shape[0]]$，完成归一化。
    *   **输入/输出:** `PIL Image` $\rightarrow$ `PIL Image`，`landmarks` $\rightarrow$ `torch.Tensor`。

2.  **`resize(image, landmarks, img_size)`:**
    *   **功能:** 将图像强制缩放到目标尺寸 `(img_size)`。
    *   **实现:** 使用 `image.resize(img_size)`。
    *   **注意:** 关键点坐标的相应调整（如果需要）应在调用此函数之前或之后手动处理。

3.  **`color_jitter(image, landmarks)`:**
    *   **功能:** 对图像进行随机的色彩抖动，增强光照和色彩鲁棒性。
    *   **实现:** 利用 `torchvision.transforms.functional.ColorJitter` 模块进行操作。

4.  **`rotate(image, landmarks, angle)`:**
    *   **功能:** 对整个图像进行随机角度的旋转，并同步更新关键点坐标。
    *   **数学核心:** 旋转操作是本模块最复杂的数学步骤。它通过构建一个二维旋转矩阵 $M$ 来执行坐标系的旋转变换：
        *   `image = imutils.rotate(np.array(image), angle)`: 图像本身被旋转。
        *   `new_landmarks = np.matmul(landmarks, transformation_matrix)`: 关键点坐标通过矩阵乘法实现几何旋转。
        *   坐标的最终调整（`new_landmarks + 0.5`）是为了抵消浮点运算和坐标系原点偏移。
    *   **输出:** 图像 (`PIL Image`) 和关键点坐标 (`np.array`)。

5.  **`__call__(self, image, landmarks, crops)` (数据管道):**
    *   **功能:** 作为一个完整的执行流程，将所有增强步骤串联起来。
    *   **执行顺序:** **Crop $\rightarrow$ Resize $\rightarrow$ ColorJitter $\rightarrow$ Rotate**。
    *   **最终标准化:** 最后，将经过所有变换的 `PIL Image` 转换为 PyTorch 的 `Tensor` 格式，并使用 `transforms.normalize()` 标准化到 $[-1, 1]$ 范围。

#### 🚀 总结与作用 (Summary & Impact)

`utils.py` 是一个完整的、可复用的数据预处理流水线。它不仅执行简单的格式转换（如 `ToTensor`），更重要的是实现了高精度的**几何和数学变换**（旋转、坐标系对齐），确保了训练数据在面对各种真实世界变化时，模型的鲁棒性。这是实现端到端训练效果的关键。