import humanfriendly
import os
import math
import random
import numpy as np
import metis
import networkx as nx


class FeaturesLoader:
    def __init__(self, data_loader, features_stat):
        self.data_loader = data_loader
        self.features_stat = features_stat
        self.page_size = humanfriendly.parse_size(features_stat["page_size"])
        self.feature_size = np.dtype(features_stat["feature_size"]).itemsize
        self.node_feature_size = self.feature_size * features_stat["feature_dimension"]
        self.nodes_per_page = max(int(self.page_size / self.node_feature_size), 1)
        self.initialize()

    def initialize(self):
        total_nodes = self.data_loader.get_num_nodes()
        self.total_pages = int(math.ceil(total_nodes / (1.0 * self.nodes_per_page)))
        self.node_location_map = np.arange(total_nodes)
        if "feature_layout" in self.features_stat and self.features_stat["feature_layout"] == "random":
            random.shuffle(self.node_location_map)
        self.node_to_page = (self.node_location_map/self.nodes_per_page).astype(int)

    def get_node_page(self, src_node, neighbor_node):
        start_node = int(self.node_location_map[neighbor_node] / self.nodes_per_page)
        curr_page_nodes = set(range(start_node, start_node + self.nodes_per_page))
        return curr_page_nodes
    
    def num_pages_for_nodes(self, nodes):
        return np.unique(self.node_to_page[nodes]).shape[0]

    def get_single_node_feature_size(self):
        return self.node_feature_size

    def get_nodes_per_page(self):
        return self.nodes_per_page

    def get_page_size(self):
        return self.page_size

    def get_total_file_size(self):
        total_bytes = self.page_size * self.total_pages
        return humanfriendly.format_size(total_bytes)

    def get_values_to_log(self):
        return {"Nodes per Page": self.nodes_per_page, "Total File Size": self.get_total_file_size()}


class NeighborFeaturesLoader(FeaturesLoader):
    def __init__(self, data_loader, features_stat):
        super().__init__(data_loader, features_stat)

    def initialize(self):
        total_nodes = self.data_loader.get_num_nodes()
        self.total_pages = total_nodes
        num_neighbors = self.nodes_per_page - 1

        self.neighbors_in_page = {}
        for curr_node in range(total_nodes):
            all_neighbors = self.data_loader.get_neigbhors_for_node(curr_node)
            neighbors_to_keep = min(len(all_neighbors), num_neighbors)
            self.neighbors_in_page[curr_node] = all_neighbors[:neighbors_to_keep]

    def get_page_for_node(self, node):
        page_nodes = set()
        page_nodes.add(node)
        for neighbor_node in self.neighbors_in_page[node]:
            page_nodes.add(neighbor_node)
        return page_nodes

    def get_node_page(self, src_node, neighbor_node):
        if neighbor_node in self.neighbors_in_page[src_node]:
            return self.get_page_for_node(src_node)
        return self.get_page_for_node(neighbor_node)


class MetisPartitionLoader(FeaturesLoader):
    def __init__(self, data_loader, features_stat):
        super().__init__(data_loader, features_stat)

    def initialize(self):
        total_nodes = self.data_loader.get_num_nodes()
        num_partitions = int(math.ceil(total_nodes / (1.0 * self.nodes_per_page)))

        # Create the adjancency list
        edge_list = []
        for node_id in range(total_nodes):
            for neighbor in self.data_loader.get_neigbhors_for_node(node_id):
                edge_list.append((node_id, neighbor))
        graph = nx.Graph(edge_list)

        # Partition the graph and store the result
        (_, parts) = metis.part_graph(graph, num_partitions)
        self.node_to_page = parts
        self.page_nodes = {}
        self.additional_page = max(self.node_to_page) + 1

        # Determine the nodes in each page
        for node_idx, node_page in enumerate(self.node_to_page):
            # Determine if we need to spill over
            if node_page in self.page_nodes and len(self.page_nodes[node_page]) >= self.nodes_per_page:
                node_page = self.additional_page
                self.node_to_page[node_idx] = node_page

            # Record the node for this page
            if node_page not in self.page_nodes:
                self.page_nodes[node_page] = set()
            self.page_nodes[node_page].add(node_idx)
            assert len(self.page_nodes[node_page]) <= self.nodes_per_page

            # Move to the next additonal page
            if node_page == self.additional_page and len(self.page_nodes[node_page]) >= self.nodes_per_page:
                self.additional_page += 1

        self.total_pages = len(self.page_nodes)
        print("Finished metis partitioning")

    def get_node_page(self, src_node, neighbor_node):
        return self.page_nodes[self.node_to_page[neighbor_node]]

features_class_map = {
    "default": FeaturesLoader,
    "neighbors_nearby": NeighborFeaturesLoader,
    "metis_partition": MetisPartitionLoader,
}

def get_featurizer(data_loader, features_stat):
    featurizer_type = features_stat["featurizer_type"]
    if featurizer_type not in features_class_map:
        raise Exception("Invalid featurizer type of " + str(featurizer_type))

    return features_class_map[featurizer_type](data_loader, features_stat)
