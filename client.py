import torch
import flwr as fl
from collections import OrderedDict
from models import SimpleCNN


def train_fedprox(local_model, global_parameters, trainloader, mu=0.1, epochs=1):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)

    # 将全局参数还原为一个不参与梯度更新的参照模型
    global_model = SimpleCNN()
    global_state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), global_parameters)}
    )
    global_model.load_state_dict(global_state_dict)
    global_model.eval()  # 参照模型不训练

    local_model.train()
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)

            proximal_term = 0.0
            for local_w, global_w in zip(local_model.parameters(), global_model.parameters()):
                proximal_term += ((local_w - global_w) ** 2).sum()

            # 总 Loss = 原本的Loss + (mu / 2) * 参数差异
            loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()


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
        train_fedprox(self.net, parameters, self.trainloader, mu=0.1, epochs=1)
        return self.get_parameters(config=None), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # test()
        loss, accuracy = 0.0, 0.0
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}