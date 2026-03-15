import torch
import flwr as fl
from collections import OrderedDict
from models import SimpleCNN


# --- FedProx 的核心训练逻辑 ---
def train_fedprox(local_model, global_parameters, trainloader, mu=0.1, epochs=1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)

    # 参照模型
    global_model = SimpleCNN()
    global_state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), global_parameters)}
    )
    global_model.load_state_dict(global_state_dict)
    global_model.eval()

    local_model.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)

            # FedProx 核心公式
            proximal_term = 0.0
            for local_w, global_w in zip(local_model.parameters(), global_model.parameters()):
                proximal_term += ((local_w - global_w) ** 2).sum()

            loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()


# --- PyTorch 测试逻辑 ---
def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# --- Flower 客户端包装器 ---
class FedProxClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # --- 从 config 中动态读取服务器发来的 mu 值 ---
        # 如果没发，默认为 0.0 (即 FedAvg)
        mu = config.get("mu", 0.0)

        # 把动态获取的 mu 传进训练函数
        train_fedprox(self.net, parameters, self.trainloader, mu=mu, epochs=5)
        return self.get_parameters(config=None), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # 计算真实准确率
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}