# train.py
# 训练主入口文件。此文件负责协调数据集、模型和训练流程，作为整个项目的执行入口。
# 注意：所有代码模块依赖都已得到完善，目前该文件代表了完整的、可执行的工程结构。

import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# --- 模块化导入 ---
from dataset import FaceLandmarksDataset
from utils import Transforms
from model import initialize_network

# --- 配置参数 ---
NUM_CLASSES = 136
NUM_EPOCHS = 10
# NUM_EPOCHS = 1
LEARNING_RATE = 0.0001
IMAGE_SIZE = (224, 224)
TRAIN_VAL_RATIO = 0.8 # 设定 80% 训练，20% 验证

def train_model():
    
    """
    执行整个模型训练流程。
    """
    print("==================================================")
    print("🚀 开始 PyTorch 人脸关键点检测模型训练")
    print("==================================================")

    # 1. 初始化数据增强和数据集
    print("\n--- 1. 初始化数据增强和数据集 ---")
    transform_pipeline = Transforms()

    # --- [重要步骤] 数据集加载和分割 ---
    try:
        # 1. 初始化总数据集
        dataset = FaceLandmarksDataset(transform=transform_pipeline)
        dataset_size = len(dataset)

        # 2. 计算训练集和验证集的大小
        train_size = int(dataset_size * TRAIN_VAL_RATIO)
        valid_size = dataset_size - train_size

        # 3. 使用 random_split 进行分割
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, valid_size]
        )

        print(f"✅ 数据集总大小: {dataset_size}")
        print(f"✅ 成功分割: 训练集 ({len(train_dataset)} 条), 验证集 ({len(val_dataset)} 条)")

        # 4. 创建 DataLoader
        BATCH_SIZE = 32 # 设定一个合适的批次大小
        # 🌟 关键修复：使用 num_workers=0 确保数据加载在主线程中，解决多进程的类型错误。
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    except Exception as e:
        print(f"\n❌ 数据集初始化或分割失败：{e}")
        print("请确认 'dlib_dataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml' 文件路径正确。")
        return

    # 2. 初始化模型和优化器
    print("\n--- 2. 初始化模型和优化器 ---")
    # === 🎯 修正点：显式传递 NUM_CLASSES 参数，确保代码健壮性 ===
    network, device = initialize_network(num_classes=NUM_CLASSES)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    loss_min = np.inf
    start_time = time.time()

    # 3. 主训练循环
    print(f"\n--- 3. 开始训练流程，总共 {NUM_EPOCHS} 个 Epoch ---")

    for epoch in range(1, NUM_EPOCHS + 1):

        # ============== 训练模式 ==============
        network.train()
        loss_train = 0

        for step, (images, landmarks) in enumerate(train_loader):
            # 移动到设备
            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0),-1).to(device)

            # 前向传播
            predictions = network(images)
            optimizer.zero_grad()

            # 计算损失
            loss_train_step = criterion(predictions, landmarks)

            # 反向传播和优化
            loss_train_step.backward()
            optimizer.step()

            loss_train += loss_train_step.item()
            # 注意：这里使用 np.array 索引防止每次计算步数时出错
            running_loss = loss_train/np.array(list(range(1, len(train_loader) + 1)))[step]
            print(f"  [Train Step {step}/{len(train_loader)}] 训练损失: {running_loss:.4f}")

        # ============== 验证模式 ==============
        network.eval()
        with torch.no_grad():
            loss_valid = 0
            for step, (images, landmarks) in enumerate(valid_loader):
                images = images.to(device)
                landmarks = landmarks.view(landmarks.size(0),-1).to(device)

                predictions = network(images)

                loss_valid_step = criterion(predictions, landmarks)
                loss_valid += loss_valid_step.item()
                running_loss = loss_valid/np.array(list(range(1, len(valid_loader) + 1)))[step]
                print(f"  [Valid Step {step}/{len(valid_loader)}] 验证损失: {running_loss:.4f}")

        # 记录并打印本轮结果
        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\\n' + '-'*60)
        print(f'📈 Epoch {epoch}/{NUM_EPOCHS} | 训练损失: {loss_train:.4f} | 验证损失: {loss_valid:.4f}')
        print('-'*60)

        # 模型保存逻辑
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), 'face_landmarks.pth')
            print(f"✨ [成功] 达到新的最小验证损失 {loss_min:.4f}，已保存模型权重。")

    end_time = time.time()
    print('\n==================================================')
    print(f'🎉 训练完成！总耗时: {end_time - start_time:.2f} 秒。')
    print('==================================================')


if __name__ == '__main__':
    # 检查设备可用性
    if not torch.cuda.is_available():
        print("⚠️ 警告：未检测到 CUDA 设备，程序将使用 CPU 进行训练。训练速度会较慢。")
    # 运行主函数
    train_model()