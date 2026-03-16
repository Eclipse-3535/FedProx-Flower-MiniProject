import json
import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # 把 round 和 value 分别提取出来
    rounds = [item[0] for item in data['accuracy']]
    accuracies = [item[1] for item in data['accuracy']]
    losses = [item[1] for item in data['loss']]
    return rounds, accuracies, losses


try:
    # 提取 FedAvg (mu=0.0) 的数据
    rounds, acc_fedavg, loss_fedavg = load_data('new_results_mu_0.0.json')
    # 提取 FedProx (mu=0.05) 的数据
    _, acc_fedprox, loss_fedprox = load_data('new_results_mu_0.05.json')
except FileNotFoundError:
    print("错误：找不到 json 文件")
    exit()

# 画图
plt.figure(figsize=(12, 5))

# --- 图 1：Accuracy vs Rounds ---
plt.subplot(1, 2, 1)
plt.plot(rounds, acc_fedavg, label='FedAvg ($\mu=0.0$)', marker='o', linestyle='-', color='blue')
plt.plot(rounds, acc_fedprox, label='FedProx ($\mu=0.05$)', marker='s', linestyle='--', color='red')
plt.title('Test Accuracy vs Communication Rounds')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.legend()

# --- 图 2：Loss vs Rounds ---
plt.subplot(1, 2, 2)
plt.plot(rounds, loss_fedavg, label='FedAvg ($\mu=0.0$)', marker='o', linestyle='-', color='blue')
plt.plot(rounds, loss_fedprox, label='FedProx ($\mu=0.05$)', marker='s', linestyle='--', color='red')
plt.title('Test Loss vs Communication Rounds')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
# 保存图片到本地
plt.savefig('new_fedprox_vs_fedavg.png', dpi=300)
print("图表已生成，保存为 new_fedprox_vs_fedavg.png")
plt.show()