import os.path as osp
from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.data import Data
import torch
import numpy as np

from omegaconf import OmegaConf


def get_marius_dataset_lp(base_directory, add_reverse_edges):

    base_directory = base_directory.rstrip("/")

    dataset_stats = OmegaConf.load(base_directory + "/dataset.yaml")

    num_columns = 3
    if dataset_stats.num_relations == 1:
        num_columns = 2

    kwargs = {}

    train_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/train_edges.bin", dtype=np.int32)).view(-1, num_columns)
    valid_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/validation_edges.bin", dtype=np.int32)).view(-1, num_columns)
    test_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/test_edges.bin", dtype=np.int32)).view(-1, num_columns)

    kwargs["train_split"] = torch.stack([train_edges[:, 0], train_edges[:, -1]]).contiguous().to(torch.int64)
    kwargs["valid_split"] = torch.stack([valid_edges[:, 0], valid_edges[:, -1]]).contiguous().to(torch.int64)
    kwargs["test_split"] = torch.stack([test_edges[:, 0], test_edges[:, -1]]).contiguous().to(torch.int64)

    edge_index = kwargs["train_split"]

    # add reverse edges
    if add_reverse_edges:
        row, col = torch.cat([edge_index[0], edge_index[1]], dim=0), torch.cat([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.stack([row, col], dim=0)

    num_nodes = dataset_stats.num_nodes
    num_relations = dataset_stats.num_relations

    if num_columns == 3:
        kwargs["train_edge_type"] = train_edges[:, 1].contiguous().to(torch.int64)
        kwargs["valid_edge_type"] = valid_edges[:, 1].contiguous().to(torch.int64)
        kwargs["test_edge_type"] = test_edges[:, 1].contiguous().to(torch.int64)

        edge_type = kwargs["train_edge_type"]

        if add_reverse_edges:
            edge_type = torch.cat([edge_type, edge_type])

        data = Data(num_nodes=num_nodes, edge_index=edge_index,
                    edge_type=edge_type, **kwargs)
    else:
        data = Data(num_nodes=num_nodes, edge_index=edge_index, **kwargs)

    data.__setattr__("num_relations", num_relations*2)

    return data


def get_marius_dataset_nc(base_directory, add_reverse_edges, only_sampling=False):

    base_directory = base_directory.rstrip("/")

    dataset_stats = OmegaConf.load(base_directory + "/dataset.yaml")

    num_columns = 3
    if dataset_stats.num_relations == 1:
        num_columns = 2

    input_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/train_edges.bin", dtype=np.int32)).view(-1, num_columns)

    edge_index = torch.stack([input_edges[:, 0], input_edges[:, -1]]).contiguous().to(torch.int64)

    # add reverse edges
    if add_reverse_edges:
        row, col = torch.cat([edge_index[0], edge_index[1]], dim=0), torch.cat([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.stack([row, col], dim=0)

    kwargs = {}

    kwargs["train_nodes"] = torch.from_numpy(np.fromfile(base_directory + "/nodes/train_nodes.bin", dtype=np.int32))
    kwargs["valid_nodes"] = torch.from_numpy(np.fromfile(base_directory + "/nodes/validation_nodes.bin", dtype=np.int32))
    kwargs["test_nodes"] = torch.from_numpy(np.fromfile(base_directory + "/nodes/test_nodes.bin", dtype=np.int32))

    num_nodes = dataset_stats.num_nodes
    num_relations = dataset_stats.num_relations

    if not only_sampling:
        features = torch.from_numpy(np.fromfile(base_directory + "/nodes/features.bin", dtype=np.float32)).view(-1, dataset_stats.feature_dim)
        labels = torch.from_numpy(np.fromfile(base_directory + "/nodes/labels.bin", dtype=np.int32))

        if num_columns == 3:
            edge_type = input_edges[:, 1]

            if add_reverse_edges:
                edge_type = torch.cat([edge_type, edge_type])

            data = Data(num_nodes=num_nodes,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        x=features,
                        y=labels,
                        **kwargs)
        else:
            data = Data(num_nodes=num_nodes,
                        edge_index=edge_index,
                        x=features,
                        y=labels,
                        **kwargs)

        data.__setattr__("num_relations", num_relations*2)
    else:
        if num_columns == 3:
            edge_type = input_edges[:, 1]

            if add_reverse_edges:
                edge_type = torch.cat([edge_type, edge_type])

            data = Data(num_nodes=num_nodes,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        **kwargs)
        else:
            data = Data(num_nodes=num_nodes,
                        edge_index=edge_index,
                        **kwargs)

        data.__setattr__("num_relations", num_relations * 2)

    return data


def select_dataset(base_directory, learning_task, add_reverse_edges, only_sample=False):

    if learning_task.upper() == "LINK_PREDICTION":
        return get_marius_dataset_lp(base_directory, add_reverse_edges)
    else:
        return get_marius_dataset_nc(base_directory, add_reverse_edges, only_sample)
