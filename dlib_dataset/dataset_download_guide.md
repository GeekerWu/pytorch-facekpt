if not os.path.exists('/content/ibug_300W_large_face_landmark_dataset'):
    !wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
    !tar -xvzf 'ibug_300W_large_face_landmark_dataset.tar.gz'    
    !rm -r 'ibug_300W_large_face_landmark_dataset.tar.gz'

下载之后解压windows

pytorch-facekpt/
├── train.py              # [核心] 主训练入口：控制整个训练流程的执行。
├── predict.py            # [核心] 主推理入口：执行模型预测和可视化展示。
├── dataset.py            # 数据加载：负责解析原始 XML 数据集，并创建 DataLoader。
├── utils.py              # 增强工具：包含所有数据增强（旋转、裁剪、缩放）和数学辅助函数。
├── model.py              # 模型定义：定义了基于 ResNet-18 的回归网络结构。
├── requirements.txt      # 运行时依赖包列表。
├── face_landmarks.pth    # 训练产物：训练过程中保存的最佳模型权重。
└── dlib_dataset/         # 原始数据集：包含 XML 标签和图像文件。
        └──ibug_300W_large_face_landmark_dataset