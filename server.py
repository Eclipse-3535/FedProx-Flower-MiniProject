import flwr as fl
import json
from client import FedProxClient
from models import SimpleCNN
from dataset import load_datasets

# ==========================================
# 核心实验参数
# ==========================================
MU = 0.1  # 0.0 代表运行 FedAvg； 0.1 代表运行 FedProx
NUM_ROUNDS = 10
NUM_CLIENTS = 2

trainloaders, valloader = load_datasets(NUM_CLIENTS)


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


# --- 每一轮开始前，告诉客户端当前轮次的 mu 是多少 ---
def fit_config(server_round: int):
    return {"mu": MU}


def client_fn(context: fl.common.Context) -> fl.client.Client:
    client_id = int(context.node_config["partition-id"])
    net = SimpleCNN()
    trainloader = trainloaders[client_id]
    return FedProxClient(cid=str(client_id), net=net, trainloader=trainloader, valloader=valloader).to_client()


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,  # 把参数发送给客户端
)

if __name__ == "__main__":
    print(f"开始联邦学习模拟 (当前 MU = {MU}, 总轮数 = {NUM_ROUNDS})...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # --- 把结果保存到本地文件 ---
    filename = f"results_mu_{MU}.json"
    results = {
        "loss": history.losses_distributed,
        "accuracy": history.metrics_distributed["accuracy"]
    }
    with open(filename, "w") as f:
        json.dump(results, f)

    print(f"实验完成，结果已保存到 {filename}")