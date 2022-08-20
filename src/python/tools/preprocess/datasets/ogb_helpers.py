import numpy as np


def remap_ogbn(node_mapping, train_nodes, valid_nodes, test_nodes, features, labels):

    num_nodes = node_mapping.shape[0]

    random_map = node_mapping[:, 1]
    random_map = random_map.astype(train_nodes.dtype)
    random_map_argsort = np.argsort(random_map)

    train_nodes = random_map[train_nodes]
    valid_nodes = random_map[valid_nodes]
    test_nodes = random_map[test_nodes]

    features = features[random_map_argsort]

    if labels.shape[0] != num_nodes:
        labels = np.concatenate((labels, -np.ones([num_nodes-labels.shape[0]], dtype=np.int32)))

    labels = labels[random_map_argsort]

    return train_nodes, valid_nodes, test_nodes, features, labels
