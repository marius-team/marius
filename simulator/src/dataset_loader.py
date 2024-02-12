import subprocess
import os
import numpy as np
from collections import defaultdict


class DatasetLoader:
    SAVE_DIR = "datasets"
    EDGES_PATH = "edges/train_edges.bin"

    def __init__(self, name):
        self.name = name
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
        self.nodes = set(edges_flaten_arr)
        edges_arr = edges_flaten_arr.reshape((-1, 2))
        self.num_edges = len(edges_arr)

        self.adjacency_map = {}
        for source, target in edges_arr:
            if source not in self.adjacency_map:
                self.adjacency_map[source] = []
            self.adjacency_map[source].append(target)

    def get_num_nodes(self):
        return len(self.nodes)

    def get_neigbhors_for_node(self, node_id):
        if node_id not in self.adjacency_map:
            return []

        return self.adjacency_map[node_id]

    def get_num_edges(self):
        return self.num_edges
