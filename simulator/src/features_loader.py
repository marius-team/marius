import humanfriendly
import os
import math


class FeaturesLoader:
    def __init__(self, data_loader, features_stat):
        self.data_loader = data_loader
        self.page_size = humanfriendly.parse_size(features_stat["page_size"])
        self.feature_size = int("".join(c for c in features_stat["feature_size"] if c.isdigit()))
        self.node_feature_size = self.feature_size * features_stat["feature_dimension"]

        self.nodes_per_page = max(int(self.page_size / self.node_feature_size), 1)
        self.total_pages = int(math.ceil(data_loader.get_num_nodes() / (1.0 * self.nodes_per_page)))

    def get_node_page(self, node_id):
        return int(node_id / self.nodes_per_page)

    def get_total_file_size(self):
        total_bytes = self.page_size * self.total_bytes
        return humanfriendly.format_size(total_bytes)
