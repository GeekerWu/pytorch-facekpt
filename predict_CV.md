### `predict_CV.py` 代码逻辑文档

#### 🔬 概述 (Overview)

`predict_CV.py` 是一个基于 OpenCV (OpenCV-Python) 和 Matplotlib 的图像处理脚本，其主要功能是读取一张指定图像，利用预训练的 Haar Cascade 算法对图像中的人脸进行实时检测，并在原始图像上绘制红色边界框来可视化这些检测结果。

#### ⚙️ 核心逻辑流程 (Core Logic Flow)

1.  **初始化与设置 (Initialization):**
    *   导入必要的库：`cv2` (OpenCV)、`matplotlib.pyplot`、`PIL` 等。
    *   设置 OpenCV 环境变量 `KMP_DUPLICATE_LIB_OK`，以避免潜在的库冲突警告。
    *   加载人脸检测器：使用 `cv2.CascadeClassifier` 加载 `haarcascade_frontalface_default.xml` 文件。

2.  **图像加载与预处理 (Image Loading & Preprocessing):**
    *   指定输入图像文件 (`multiple.jpg`)，并使用 `cv2.imread()` 加载原始图像。
    *   将原始图像转换为灰度图 (`grayscale_image`)，这是人脸检测算法通常需要的格式。
    *   将原始图像转换为 RGB 格式 (`display_image`)，用于后续 Matplotlib 的彩色显示。
    *   打印出图像的尺寸信息（宽度和高度）。

3.  **人脸检测 (Face Detection):**
    *   调用 `face_cascade.detectMultiScale()` 方法，在灰度图上执行多尺度的人脸检测。
    *   `detectMultiScale` 的参数：`1.1` (尺度因子)，`4` (最小检测面积乘数)。该函数返回一个包含所有检测到人脸边界框的数组，每个边界框格式为 `(x, y, w, h)` (x: 左上角X坐标, y: 左上角Y坐标, w: 宽度, h: 高度)。

4.  **结果可视化 (Visualization):**
    *   脚本检查是否成功检测到人脸 (`len(faces) > 0`)。
    *   如果检测到人脸，使用 Matplotlib 进行绘图：
        *   显示原始的彩色图像 (`plt.imshow(display_image)`).
        *   遍历所有检测到的边界框 `(x, y, w, h)`。
        *   为每个边界框创建一个红色的、无填充的 `Rectangle` 补丁，并将其添加到图表上。
        *   最后，调用 `plt.show()` 展示包含边界框的图像。

#### 💾 输入与输出 (I/O Specification)

*   **输入 (Input):**
    *   **图像文件:** `multiple.jpg` (脚本配置的输入图像路径)。
    *   **模型文件:** `haarcascade_frontalface_default.xml` (人脸检测模型的配置文件)。
*   **输出 (Output):**
    *   程序控制台输出：图像的宽度和高度信息。
    *   屏幕显示：一个包含原始图像和红色矩形边界框的 Matplotlib 图窗，边界框代表检测到的人脸位置。

#### ✨ 使用建议与注意事项 (Usage Notes)

1.  **依赖环境 (Dependencies):**
    *   需要安装 `opencv-python` (cv2)，`matplotlib`，`Pillow` 等库。
2.  **参数修改 (Customization):**
    *   **输入图像:** 如果需要检测不同的图像，需要修改脚本顶部的 `image_path` 变量。
    *   **人脸检测参数:** 可以调整 `detectMultiScale` 的尺度因子和面积参数，以改变检测的灵敏度和准确性。
3.  **性能考虑:** 该方法依赖传统的 Haar Cascade 算法，虽然简单易用，但在面对复杂背景或非标准人脸（如侧脸）时，精度可能会受到影响。