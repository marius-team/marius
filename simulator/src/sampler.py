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
        
        self.nodes_loaded = set()
        self.pages_loaded = 0
        self.depth = config["sampling_depth"]
    
    def reset(self):
        self.nodes_loaded.clear()
        self.pages_loaded = 0

    def is_node_in_mem(self, node_id):
        return self.in_memory_storage is not None and self.in_memory_storage.node_in_mem_storage(node_id)
    
    def get_node_features(self, src_node, neighbor_node):
        if neighbor_node in self.nodes_loaded or self.is_node_in_mem(neighbor_node):
            return
        
        self.nodes_loaded.update(self.features_loader.get_node_page(src_node, neighbor_node))
        self.pages_loaded += 1

    def perform_sampling_for_node(self, node_id):
        # Read for this node
        self.reset()

        # Perform bfs
        curr_queue = [(node_id, node_id)]
        curr_depth = 0
        while curr_depth <= self.depth and len(curr_queue) > 0:
            # Get all of the nodes in the level
            level_nodes = len(curr_queue)
            for _ in range(level_nodes):
                src_node, curr_node = curr_queue.pop(0)
                self.get_node_features(src_node, curr_node)
                for neighbor in self.data_loader.get_neigbhors_for_node(curr_node):
                    curr_queue.append((curr_node, neighbor))
            
            # Move to the next level
            curr_depth += 1

        return self.pages_loaded

    def get_values_to_log(self):
        values_to_return = {}
        if self.in_memory_storage is not None:
            nodes_in_memory = self.in_memory_storage.in_mem_nodes_count()
            values_to_return["Percentage Nodes In Memory"] = self.in_memory_storage.get_percentage_in_mem()
            in_mem_pages = int(math.ceil(nodes_in_memory / self.features_loader.get_nodes_per_page()))
            all_pages_size = humanfriendly.format_size(in_mem_pages * self.features_loader.get_page_size())
            values_to_return["In Memory Space Used"] = all_pages_size

        return values_to_return
