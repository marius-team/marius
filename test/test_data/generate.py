import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter, split_edges


def get_random_graph(num_nodes, num_edges, num_rels=1):
    src_nodes = np.random.randint(0, num_nodes, size=[num_edges])
    dst_nodes = np.random.randint(0, num_nodes, size=[num_edges])

    if num_rels > 1:
        rels = np.random.randint(0, num_rels, size=[num_edges])
        edges = np.stack([src_nodes, rels, dst_nodes], axis=1)
    else:
        edges = np.stack([src_nodes, dst_nodes], axis=1)

    return edges


def generate_features(num_nodes, feature_dim):
    return np.random.randn(num_nodes, feature_dim).astype(np.float32)


def generate_labels(num_nodes, num_classes):
    return np.random.randint(0, num_classes - 1, size=[num_nodes]).astype(np.int32)


def shuffle_with_map(values, node_mapping):
    random_map = node_mapping[:, 1].astype(values.dtype)
    random_map_argsort = np.argsort(random_map)
    return values[random_map_argsort]


def apply_mapping(values, node_mapping):
    random_map = node_mapping[:, 1].astype(values.dtype)
    return random_map[values]


def remap_nc(output_dir, train_nodes, labels, num_nodes, valid_nodes=None, test_nodes=None, features=None):
    node_mapping = np.genfromtxt(output_dir / Path(PathConstants.node_mapping_path), delimiter=",")

    train_nodes = apply_mapping(train_nodes, node_mapping)

    if valid_nodes is not None:
        valid_nodes = apply_mapping(valid_nodes, node_mapping)

    if test_nodes is not None:
        test_nodes = apply_mapping(test_nodes, node_mapping)

    if features is not None:
        features = shuffle_with_map(features, node_mapping)

    if labels.shape[0] != num_nodes:
        labels = np.concatenate((labels, -np.ones([num_nodes - labels.shape[0]], dtype=np.int32)))

    labels = shuffle_with_map(labels, node_mapping)

    return train_nodes, labels, valid_nodes, test_nodes, features


def remap_lp(output_dir, features=None):
    node_mapping = np.genfromtxt(output_dir / Path(PathConstants.node_mapping_path), delimiter=",")
    features = shuffle_with_map(features, node_mapping)

    return features


def generate_random_dataset_nc(
    output_dir,
    num_nodes,
    num_edges,
    num_rels=1,
    splits=None,
    num_partitions=1,
    partitioned_eval=False,
    sequential_train_nodes=False,
    remap_ids=True,
    feature_dim=-1,
    num_classes=10,
):
    edges = get_random_graph(num_nodes, num_edges, num_rels)
    edges_df = pd.DataFrame(data=edges)

    src_col, dst_col, edge_type_col = None, None, None
    if edges.shape[1] == 3:
        src_col, dst_col, edge_type_col = 0, 2, 1
    else:
        src_col, dst_col = 0, 1

    raw_edges_filename = output_dir / Path("raw_edges.csv")
    edges_df.to_csv(raw_edges_filename, ",", header=False, index=False)

    all_nodes = np.arange(0, num_nodes, dtype=np.int32)
    train_nodes = all_nodes

    valid_nodes = None
    test_nodes = None
    if splits is not None:
        train_nodes, train_weights, valid_nodes, valid_weights, test_nodes, test_weights = split_edges(
            all_nodes, None, splits
        )

    converter = TorchEdgeListConverter(
        output_dir,
        train_edges=Path(raw_edges_filename),
        delim=",",
        remap_ids=remap_ids,
        num_partitions=num_partitions,
        partitioned_evaluation=partitioned_eval,
        sequential_train_nodes=sequential_train_nodes,
        known_node_ids=[train_nodes, valid_nodes, test_nodes],
        format="CSV",
        src_column=src_col,
        dst_column=dst_col,
        edge_type_column=edge_type_col,
    )

    dataset_stats = converter.convert()

    features = None
    if feature_dim != -1:
        features = generate_features(num_nodes, feature_dim)

    labels = generate_labels(num_nodes, num_classes)

    train_nodes, labels, valid_nodes, test_nodes, features = remap_nc(
        output_dir, train_nodes, labels, num_nodes, valid_nodes, test_nodes, features
    )

    if features is not None:
        node_features_file = output_dir / Path(PathConstants.node_features_path)
        with open(node_features_file, "wb") as f:
            f.write(bytes(features))

    labels_file = output_dir / Path(PathConstants.labels_path)
    with open(labels_file, "wb") as f:
        f.write(bytes(labels))

    if train_nodes is not None:
        train_nodes_file = output_dir / Path(PathConstants.train_nodes_path)
        with open(train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))

    if valid_nodes is not None:
        valid_nodes_file = output_dir / Path(PathConstants.valid_nodes_path)
        with open(valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))

    if test_nodes is not None:
        test_nodes_file = output_dir / Path(PathConstants.test_nodes_path)
        with open(test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))

    # update dataset yaml
    dataset_stats.num_train = train_nodes.shape[0]

    if valid_nodes is not None:
        dataset_stats.num_valid = valid_nodes.shape[0]
    else:
        dataset_stats.num_valid = -1

    if test_nodes is not None:
        dataset_stats.num_test = test_nodes.shape[0]
    else:
        dataset_stats.num_test = -1

    if features is not None:
        dataset_stats.node_feature_dim = features.shape[1]
    else:
        dataset_stats.node_feature_dim = -1

    dataset_stats.num_classes = num_classes

    dataset_stats.num_nodes = num_nodes

    with open(output_dir / Path("dataset.yaml"), "w") as f:
        yaml_file = OmegaConf.to_yaml(dataset_stats)
        f.writelines(yaml_file)


def generate_random_dataset_lp(
    output_dir,
    num_nodes,
    num_edges,
    num_rels=1,
    splits=None,
    num_partitions=1,
    partitioned_eval=False,
    sequential_train_nodes=False,
    remap_ids=True,
    feature_dim=-1,
):
    edges = get_random_graph(num_nodes, num_edges, num_rels)
    edges_df = pd.DataFrame(data=edges)

    src_col, dst_col, edge_type_col = None, None, None
    if edges.shape[1] == 3:
        src_col, dst_col, edge_type_col = 0, 2, 1
    else:
        src_col, dst_col = 0, 1

    raw_edges_filename = output_dir / Path("raw_edges.csv")

    edges_df.to_csv(raw_edges_filename, ",", header=False, index=False)

    converter = TorchEdgeListConverter(
        output_dir,
        train_edges=raw_edges_filename,
        delim=",",
        splits=splits,
        num_partitions=num_partitions,
        remap_ids=remap_ids,
        partitioned_evaluation=partitioned_eval,
        sequential_train_nodes=sequential_train_nodes,
        format="CSV",
        src_column=src_col,
        dst_column=dst_col,
        edge_type_column=edge_type_col,
    )

    dataset_stats = converter.convert()

    if feature_dim != -1:
        features = generate_features(num_nodes, feature_dim)

        if remap_ids:
            features = remap_lp(output_dir, features)

        node_features_file = output_dir / Path(PathConstants.node_features_path)
        with open(node_features_file, "wb") as f:
            f.write(bytes(features))

        dataset_stats.node_feature_dim = feature_dim
        with open(output_dir / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)


def generate_random_dataset(
    output_dir,
    num_nodes,
    num_edges,
    num_rels=1,
    splits=None,
    num_partitions=1,
    partitioned_eval=False,
    sequential_train_nodes=False,
    remap_ids=True,
    feature_dim=-1,
    num_classes=10,
    task="lp",
):
    os.makedirs(output_dir, exist_ok=True)

    if task == "lp":
        generate_random_dataset_lp(
            output_dir,
            num_nodes,
            num_edges,
            num_rels,
            splits,
            num_partitions,
            partitioned_eval,
            sequential_train_nodes,
            remap_ids,
            feature_dim,
        )
    elif task == "nc":
        generate_random_dataset_nc(
            output_dir,
            num_nodes,
            num_edges,
            num_rels,
            splits,
            num_partitions,
            partitioned_eval,
            sequential_train_nodes,
            remap_ids,
            feature_dim,
            num_classes,
        )
    else:
        raise RuntimeError("Unsupported dataset type for generator.")
