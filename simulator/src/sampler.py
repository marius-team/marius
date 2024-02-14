class SubgraphSampler:
    def __init__(self, data_loader, features_loader):
        self.data_loader = data_loader
        self.features_loader = features_loader

    def perform_sampling_for_node(self, node_id):
        # Read for this node
        pages_read = set()
        pages_read.add(self.features_loader.get_node_page(node_id, node_id))

        # Read the features for this node's neighbors
        for neighbor in self.data_loader.get_neigbhors_for_node(node_id):
            pages_read.add(self.features_loader.get_node_page(node_id, neighbor))

        return len(pages_read)
