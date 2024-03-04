from .metrics import *
import numpy as np
import torch

class InMemoryStorage:
    def __init__(self, data_loader, config):
        self.percent_in_memory = float(config["top_percent_in_mem"])
        total_nodes = data_loader.get_num_nodes()
        nodes_in_mem = int((total_nodes * self.percent_in_memory) / 100.0)

        # Get the top nodes based on incoming neighbors
        self.in_memory_nodes = data_loader.get_nodes_sorted_by_incoming()[ : nodes_in_mem].numpy()
        print("Loaded", self.in_memory_nodes.shape[0], "nodes in memory")

    def node_in_mem_storage(self, node_id):
        return node_id in self.in_memory_nodes

    def get_percentage_in_mem(self):
        return self.percent_in_memory

    def in_mem_nodes_count(self):
        return len(self.in_memory_nodes)
    
    def remove_in_mem_nodes(self, nodes):
        return torch.tensor(np.setdiff1d(nodes, self.in_memory_nodes))