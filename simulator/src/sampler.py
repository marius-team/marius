from .in_mem_storage import *
import humanfriendly
import math
import time
from .metrics import *

class SubgraphSampler:
    def __init__(self, data_loader, features_loader, config):
        self.data_loader = data_loader
        self.features_loader = features_loader
        self.sampling_depth = config["sampling_depth"]
        self.in_memory_storage = None
        if "top_percent_in_mem" in config:
            self.in_memory_storage = InMemoryStorage(data_loader, config)
        self.metrics = MetricTracker()

    def remove_high_degree(self, nodes):
        if self.in_memory_storage is not None:
            nodes = self.in_memory_storage.remove_in_mem_nodes(nodes)
        return nodes
    
    def get_in_mem_storage(self):
        return self.in_memory_storage

    def perform_sampling_for_nodes(self, batch_idx):
        # Get the graph for those nodes
        graph_get_start = time.time()
        current_graph = self.data_loader.get_graph_for_batch(batch_idx)
        if current_graph is None:
            return False, -1
        graph_get_time = time.time() - graph_get_start
        self.metrics.record_metric("graph_get_time", graph_get_time)

        current_depth = 0
        total_pages_loaded = 0
        while current_depth < self.sampling_depth:
            # Get the neigbhors for the current level
            get_start_time = time.time()
            nodes = current_graph.getNeighborIDs(True, True)
            num_search_nodes = nodes.shape[0]
            get_end_time = time.time()
            self.metrics.record_metric("get_neighbor_time", get_end_time - get_start_time)
            self.metrics.record_metric("search_nodes_num", num_search_nodes)

            # Remove any in memory nodes for future levels
            remove_start_time = time.time()
            nodes = self.remove_high_degree(nodes)
            if nodes.shape[0] == 0:
                break
            remove_end_time = time.time()
            self.metrics.record_metric("node_removal_time", remove_end_time - remove_start_time)
            
            # Get the pages for this level
            features_start_time = time.time()
            total_pages_loaded += self.features_loader.num_pages_for_nodes(nodes.numpy())
            features_end_time = time.time()
            self.metrics.record_metric("features_get_time", features_end_time - features_start_time)
            self.metrics.record_metric("features_nodes_num", nodes.shape[0])

            # Record level time
            prepare_start_time = time.time()
            current_depth += 1
            if current_depth < self.sampling_depth:
                current_graph.prepareForNextLayer()  
            overall_prepare_time = time.time() - prepare_start_time
            self.metrics.record_metric("prepare_next_layer_time", overall_prepare_time)

            level_end_time = time.time()
            level_process_time = level_end_time - get_start_time
            self.metrics.record_metric("level_processing_time", level_process_time)
        
        return True, total_pages_loaded

    def get_values_to_log(self):
        values_to_return = {}
        if self.in_memory_storage is not None:
            nodes_in_memory = self.in_memory_storage.in_mem_nodes_count()
            values_to_return["Percentage Nodes In Memory"] = self.in_memory_storage.get_percentage_in_mem()
            in_mem_pages = int(math.ceil(nodes_in_memory / self.features_loader.get_nodes_per_page()))
            all_pages_size = humanfriendly.format_size(in_mem_pages * self.features_loader.get_page_size() * self.sampling_depth)
            values_to_return["In Memory Space Used"] = all_pages_size

        return values_to_return
    
    def get_metrics(self):
        return self.metrics.get_metrics()
