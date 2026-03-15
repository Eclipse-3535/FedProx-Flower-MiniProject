import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def load_datasets(num_clients: int):
    print("正在下载和处理 MNIST 数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # ==========================================
    # 制造 Non-IID 数据 (按标签排序后切割)
    # ==========================================
    # 获取所有训练数据的标签
    train_labels = np.array(trainset.targets)

    # 根据标签对数据索引进行排序
    sorted_indices = np.argsort(train_labels)

    # 将排好序的索引平均分给每个客户端
    chunk_size = len(sorted_indices) // num_clients
    trainloaders = []

    for i in range(num_clients):
        # 客户端 0 拿到前半部分 (0~4)，客户端 1 拿到后半部分 (5~9)
        client_indices = sorted_indices[i * chunk_size: (i + 1) * chunk_size]
        client_ds = Subset(trainset, client_indices)
        trainloaders.append(DataLoader(client_ds, batch_size=32, shuffle=True))

    valloader = DataLoader(testset, batch_size=32)

    return trainloaders, valloader