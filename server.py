import flwr as fl

# 配置联邦学习策略
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,         # 每轮抽取 100% 的客户端参与训练
    fraction_evaluate=1.0,    # 每轮抽取 100% 的客户端参与评估
    min_fit_clients=2,        # 至少 2 个客户端
    min_evaluate_clients=2,
    min_available_clients=2,
)

# 启动 Simulation
# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=2,
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )