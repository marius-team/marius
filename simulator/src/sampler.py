class SubgraphSampler:
    def __init__(self, data_loader, features_loader):
        self.data_loader = data_loader
        self.features_loader = features_loader

    def perform_sampling_for_node(self, node_id):
        # Read for this node
        neighbor_nodes = self.data_loader.get_neigbhors_for_node(node_id)
        if len(neighbor_nodes) == 0:
            return 0

        # Load the nodes features
        nodes_features_loaded = self.features_loader.get_node_page(node_id, node_id)
        pages_loaded = 1
        log_value = len(neighbor_nodes) > len(nodes_features_loaded)

        for neighbor in neighbor_nodes:
            if neighbor in nodes_features_loaded:
                continue

            # We haven't loaded the page for this node
            neighbors_page = self.features_loader.get_node_page(node_id, neighbor)
            nodes_features_loaded.update(neighbors_page)
            pages_loaded += 1

        return pages_loaded
