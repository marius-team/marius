from .in_mem_storage import *
import humanfriendly
import math


class SubgraphSampler:
    def __init__(self, data_loader, features_loader, config):
        self.data_loader = data_loader
        self.features_loader = features_loader
        self.in_memory_storage = None
        if "top_percent_in_mem" in config:
            self.in_memory_storage = InMemoryStorage(data_loader, config["top_percent_in_mem"])

    def perform_sampling_for_nodes(self, nodes):
        # Get the neighbors of the node
        node_neigbhors = self.data_loader.get_neigbhors_for_nodes(nodes)
        if node_neigbhors.shape[0] == 0:
            return 0
        
        # Remove the in memory nodes
        if self.in_memory_storage is not None:
            node_neigbhors = self.in_memory_storage.remove_in_mem_nodes(node_neigbhors)
        if node_neigbhors.shape[0] == 0:
            return 0
        
        # Get the average pages per node
        total_pages_loaded = self.features_loader.num_pages_for_nodes(node_neigbhors)
        return total_pages_loaded/nodes.shape[0]

    def get_values_to_log(self):
        values_to_return = {}
        if self.in_memory_storage is not None:
            nodes_in_memory = self.in_memory_storage.in_mem_nodes_count()
            values_to_return["Percentage Nodes In Memory"] = self.in_memory_storage.get_percentage_in_mem()
            in_mem_pages = int(math.ceil(nodes_in_memory / self.features_loader.get_nodes_per_page()))
            all_pages_size = humanfriendly.format_size(in_mem_pages * self.features_loader.get_page_size())
            values_to_return["In Memory Space Used"] = all_pages_size

        return values_to_return
