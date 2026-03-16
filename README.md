# Federated Learning 论文复现: FedProx

本项目使用 PyTorch 和 Flower 框架，在单机模拟环境下复现了联邦学习论文 **FedProx** 的核心算法及其在 Non-IID（非独立同分布）数据下的优势。

**论文出处：**
> Li T, Sahu A K, Zaheer M, Sanjabi M, Talwalkar A, Smith V. Federated Optimization in Heterogeneous Networks[C]//Proceedings of Machine Learning and Systems. 2020, 2: 429–450.

---

## 核心复现目标 (Minimum Reproducible Result, MRR)

1. **跑通核心算法**：在客户端的本地损失函数中成功引入近端项（Proximal Term），实现了对本地模型偏离全局模型的惩罚。
2. **复现关键结果**：构造了极端的 Non-IID 数据划分（按标签排序后切割），证明了 FedAvg 在数据异构且本地迭代次数较多（Epochs=5）时会出现“客户端漂移（Client Drift）”导致的震荡与性能下降，而 FedProx 能够有效缓解这一问题。
3. **消融实验**：通过调整近端项系数 $\mu$，对比了 $\mu=0$ (退化为 FedAvg)、 $\mu=0.05$ 和 $\mu=0.1$（过大的惩罚导致欠拟合）三种情况下的收敛曲线。

---

## 环境依赖与安装

本项目使用 Python 3.9 并在 Anaconda 环境下测试通过。

1. **克隆仓库**
git clone https://github.com/Eclipse-3535/FedProx-Flower-MiniProject.git
cd FedProx-Flower-MiniProject

2. **创建并激活虚拟环境**
conda create -n fl_env python=3.9 -y
conda activate fl_env

3. **安装依赖项**
pip install flwr[simulation]>=1.5.0 torch torchvision matplotlib numpy

---

## 一键复现实验命令

本项目将超参数集中在 `server.py` 顶部，可以通过修改 `MU` 的值来运行不同的基线或算法。

### 1. 运行 FedAvg ($\mu=0.0$)
将 `server.py` 中第 10 行修改为 `MU = 0.0`，然后运行：
python server.py
*(运行结束后，将在当前目录生成 `new_results_mu_0.0.json` 文件)*

### 2. 运行 FedProx ($\mu=0.05$)
将 `server.py` 中第 10 行修改为 `MU = 0.05`，然后运行：
python server.py
*(运行结束后，将在当前目录生成 `new_results_mu_0.05.json` 文件)*

### 3. 生成对比图表
确保上述两个 JSON 文件都已生成，运行：
python plot.py
*(该脚本将读取 JSON 文件，并生成 `new_fedprox_vs_fedavg.png` 包含 Accuracy 和 Loss 的对比曲线图)*

---

## 关键超参数与实验设定声明

为了适应 CPU-only 的单机模拟环境，我在复现范围上做了以下缩放（不影响核心结论）：
*   **模型**：使用轻量级的 3 层 SimpleCNN。
*   **数据集**：使用 MNIST（代替原论文的部分复杂数据集）。
*   **客户端数量**：缩减为 2 个客户端参与全量训练（代替原论文的 100+ 客户端抽样）。
*   **数据异质性**：采用极端的 Non-IID 划分（客户端 0 仅含标签 0-4，客户端 1 仅含标签 5-9）。
*   **训练烈度放大**：为了在简单的 MNIST 数据集上体现出客户端漂移现象，我故意调大了局部训练参数：**局部学习率 (lr) 提升至 0.05，局部迭代次数 (Epochs) 提升至 10**。
*   **通信轮数 (Rounds)**：设置为 20 轮，以观察长期的收敛与震荡趋势。

此降配与参数调整并没有改变 FedProx 应对异质性的底层逻辑，反而使得 FedAvg 的漂移问题被放大，从而在小规模实验中直观呈现了 FedProx 近端项的价值。