import subprocess
import os
import numpy as np
import torch
import time
import traceback
import threading
from collections import defaultdict
from marius.data import Batch, DENSEGraph, MariusGraph
from marius.data.samplers import LayeredNeighborSampler

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

        # Create the graph
        self.edge_list = torch.tensor(edges_arr, dtype = torch.int64)
        self.total_nodes = torch.max(self.edge_list).item() + 1
        self.current_graph = MariusGraph(self.edge_list, self.edge_list[torch.argsort(self.edge_list[:, -1])], self.total_nodes)
        self.sampler = LayeredNeighborSampler(self.current_graph, [-1 for _ in range(self.sampling_depth)])

        self.batch_graphs = []
    
    def get_num_nodes(self):
        return self.total_nodes
    
    def get_graph_for_batch(self, batch_idx):
        while batch_idx not in self.batch_graphs:
            batch_idx += 1
            batch_idx -= 1

        return self.batch_graphs[batch_idx]
    
    def start_graph_initialization(self, batches, in_memory_storage):
        self.thread = threading.Thread(target = self.generate_graph_per_batch, args = (batches, in_memory_storage, ))
        self.thread.daemon = True
        self.thread.start()
        time.sleep(20)
    
    def generate_graph_per_batch(self, batches, in_memory_storage):
        for batch in batches:
            batch_graph = None
            try:
                if in_memory_storage is not None:
                    batch = in_memory_storage.remove_in_mem_nodes(batch)
                sampled_nodes = self.sampler.getNeighbors(batch)
                sampled_nodes.performMap()
                batch_graph = sampled_nodes
            except:
                batch_graph = None
            
            self.batch_graphs.append(batch_graph)

    def get_num_edges(self):
        return self.edge_list.size(0)
    
    def get_nodes_sorted_by_incoming(self):
        return torch.argsort(torch.bincount(self.edge_list[ : , 1]), descending=True)

    def get_average_neighbors(self):
        outgoing_nodes = self.edge_list[ : , 0]
        outgoing_unique_nodes = torch.unique(outgoing_nodes)
        return outgoing_nodes.size(0)/outgoing_unique_nodes.size(0)

    def get_average_incoming(self):
        incoming_nodes = self.edge_list[ : , 1]
        incoming_unique_nodes = torch.unique(incoming_nodes)
        return incoming_nodes.size(0)/incoming_unique_nodes.size(0)

    def get_values_to_log(self):
        return {
            "Average Node Out Degree": str(round(self.get_average_neighbors(), 2)),
            "Average Node In Degree": str(round(self.get_average_incoming(), 2)),
        }