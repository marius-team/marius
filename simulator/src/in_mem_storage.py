import heapq


class InMemoryStorage:
    def __init__(self, data_loader, percent_in_memory):
        self.percent_in_memory = percent_in_memory
        total_nodes = data_loader.get_num_nodes()
        nodes_in_mem = int((total_nodes * self.percent_in_memory) / 100.0)

        # Get the top nodes based on incoming neighbors
        heap = []
        for node_id in range(total_nodes):
            num_incoming = data_loader.get_incoming_neighbors(node_id)
            heapq.heappush(heap, (num_incoming, node_id))

        top_pairs = heapq.nlargest(nodes_in_mem, heap)
        self.in_memory_nodes = set([pair[1] for pair in top_pairs])

    def node_in_mem_storage(self, node_id):
        return node_id in self.in_memory_nodes

    def get_percentage_in_mem(self):
        return self.percent_in_memory

    def in_mem_nodes_count(self):
        return len(self.in_memory_nodes)
