from dataclasses import dataclass


@dataclass
class PathConstants:
    model_file: str = "model.pt"
    model_state_file: str = "model_state.pt"
    edges_directory: str = "edges/"
    nodes_directory: str = "nodes/"
    training_file_prefix: str = "train_"
    validation_file_prefix: str = "validation_"
    test_file_prefix: str = "test_"
    partition_offsets_file: str = "partition_offsets.txt"
    node_mapping_file: str = "node_mapping.txt"
    relation_mapping_file: str = "relation_mapping.txt"
    edge_file_name: str = "edges"
    node_file_name: str = "nodes"
    features_file_name: str = "features"
    labels_file_name: str = "labels"
    node_embeddings_file_name: str = "embeddings"
    node_embeddings_state_file_name: str = "embeddings_state"
    file_ext: str = ".bin"

    train_edges_path: str = edges_directory + training_file_prefix + edge_file_name + file_ext
    valid_edges_path: str = edges_directory + validation_file_prefix + edge_file_name + file_ext
    test_edges_path: str = edges_directory + test_file_prefix + edge_file_name + file_ext
    train_edge_buckets_path: str = edges_directory + training_file_prefix + partition_offsets_file
    valid_edge_buckets_path: str = edges_directory + validation_file_prefix + partition_offsets_file
    test_edge_buckets_path: str = edges_directory + test_file_prefix + partition_offsets_file

    node_features_path: str = nodes_directory + features_file_name + file_ext
    relation_features_path: str = edges_directory + features_file_name + file_ext
    labels_path: str = nodes_directory + labels_file_name + file_ext

    train_nodes_path: str = nodes_directory + training_file_prefix + node_file_name + file_ext
    valid_nodes_path: str = nodes_directory + validation_file_prefix + node_file_name + file_ext
    test_nodes_path: str = nodes_directory + test_file_prefix + node_file_name + file_ext

    node_mapping_path: str = nodes_directory + node_mapping_file
    relation_mapping_path: str = edges_directory + relation_mapping_file

