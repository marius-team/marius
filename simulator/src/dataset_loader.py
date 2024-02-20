import subprocess
import os
import numpy as np
import torch
from collections import defaultdict
import marius.storage

class DatasetLoader:
    SAVE_DIR = "datasets"
    EDGES_PATH = "edges/train_edges.bin"

    def __init__(self, config):
        self.name = config["dataset_name"]
        self.sampling_depth = config["sampling_depth"]
        os.makedirs(DatasetLoader.SAVE_DIR, exist_ok=True)
        self.save_dir = os.path.join(DatasetLoader.SAVE_DIR, self.name)
        if not os.path.exists(self.save_dir):
            self.create_dataset()
        self.load_dataset()

    def create_dataset(self):
        command_to_run = f"marius_preprocess --dataset {self.name} --output_directory {self.save_dir}"
        print("Running command", command_to_run)
        subprocess.check_output(command_to_run, shell=True)

    def load_dataset(self):
        # Load the file
        edges_path = os.path.join(self.save_dir, DatasetLoader.EDGES_PATH)
        with open(edges_path, "rb") as reader:
            edges_bytes = reader.read()

        # Create the adjacency map
        edges_flaten_arr = np.frombuffer(edges_bytes, dtype=np.int32)
        edges_arr = edges_flaten_arr.reshape((-1, 2))

        '''
        # Create the graph
        self.edge_list = torch.from_numpy(edges_arr)
        self.nodes = torch.unique(self.edge_list)
        self.current_graph = MariusGraph(self.edge_list, self.edge_list[torch.argsort(edge_list[:, -1])], self.get_num_nodes())
        self.sampler = LayeredNeighborSampler(full_graph, [-1 for _ in range(self.sampling_depth)])

        # Neighbors cache
        '''

    def get_num_nodes(self):
        return self.nodes.shape(0)

    def get_neigbhors_for_node(self, node_id, all_depths = False):
        if node_id not in self.adjacency_map:
            return []

        return list(self.adjacency_map[node_id])

    def get_incoming_neighbors(self, node_id):
        if node_id not in self.num_incoming_edges:
            return 0

        return self.num_incoming_edges[node_id]

    def get_num_edges(self):
        return self.edge_list.shape(0)

    def get_average_neighbors(self):
        neighbors_count = []
        for node_neighbors in self.adjacency_map.values():
            neighbors_count.append(len(node_neighbors))

        return np.mean(np.array(neighbors_count))

    def get_average_incoming(self):
        incoming_counts = []
        for num_incoming in self.num_incoming_edges.values():
            incoming_counts.append(num_incoming)

        return np.mean(np.array(incoming_counts))

    def get_values_to_log(self):
        return {
            "Average Node Out Degree": str(round(self.get_average_neighbors(), 2)),
            "Average Node In Degree": str(round(self.get_average_incoming(), 2)),
        }
