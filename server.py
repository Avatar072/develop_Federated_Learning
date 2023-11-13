from typing import List, Tuple
import sys
import flwr as fl
from flwr.common import Metrics
global_round=50
try:
    global_round=sys.argv[1]
    global_round=int(global_round)
except:
    global_round

#python server.py 5 5是gobal round數
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
#strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,min_fit_clients=2,min_available_clients=2)
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,min_fit_clients=2,min_available_clients=2)
# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:53388",
    config=fl.server.ServerConfig(num_rounds=global_round),
    strategy=strategy,
)
print("\n     Round: ", global_round)