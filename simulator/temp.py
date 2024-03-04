from marius.data import *
from marius.data.samplers import *
import numpy as np
import torch

def load_graph():
    # Load the file
    with open("/working_dir/simulator/datasets/tree/edges/train_edges.bin", "rb") as reader:
        edges_bytes = reader.read()

    # Create the adjacency map
    edges_flaten_arr = np.frombuffer(edges_bytes, dtype=np.int32)
    edges_arr = edges_flaten_arr.reshape((-1, 2))

    # Create the graph
    edge_list = torch.tensor(edges_arr, dtype = torch.int64)
    total_nodes = torch.max(edge_list).item()
    return MariusGraph(edge_list, edge_list[torch.argsort(edge_list[:, -1])], total_nodes)

def main(sampling_depth = 3):
    # Load the graph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = load_graph()
    graph.to(device)

    # Load the features config
    features_config = FeaturesLoaderConfig()
    features_config.features_type = "linear"
    features_config.page_size = 100
    features_config.feature_dimension = 2
    features_config.feature_size = 20
    features_loader = get_feature_loader(features_config, graph)

    # Creat the sampler
    in_memory_nodes = torch.tensor([6], dtype = torch.int64, device = device)
    levels = [-1, -1, -1]
    sampler = LayeredNeighborSampler(graph, levels, in_memory_nodes, features_config)

    # Run for pages
    sample_nodes = torch.tensor([1], dtype = torch.int64, device = device)
    num_pages = sampler.getNeighborsPages(sample_nodes)
    print("Pages for nodes", sample_nodes.item(), "are", num_pages)
    print("Sampler scaling factor of", sampler.getAvgScalingFactor())
    print("Sampler avg percent removed of", sampler.getAvgPercentRemoved())

if __name__ == "__main__":
    main()