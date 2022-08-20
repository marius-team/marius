import dgl
import torch
import numpy as np

from omegaconf import OmegaConf


class MariusLPDataset(object):

    def __init__(self, base_directory, add_reverse_edges=True, only_train=False):
        self.base_directory = base_directory
        self.add_reverse_edges = add_reverse_edges
        self.only_train = only_train

    def process(self):
        base_directory = self.base_directory.rstrip("/")

        dataset_stats = OmegaConf.load(base_directory + "/dataset.yaml")

        num_columns = 3
        if dataset_stats.num_relations == 1:
            num_columns = 2

        if self.only_train:
            edges = [torch.from_numpy(np.fromfile(base_directory + "/edges/train_edges.bin",
                                                  dtype=np.int32)).view(-1, num_columns)]

            edges = torch.cat(edges)
        else:
            train_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/train_edges.bin",
                                                       dtype=np.int32)).view(-1, num_columns)
            valid_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/validation_edges.bin",
                                                       dtype=np.int32)).view(-1, num_columns)
            test_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/test_edges.bin",
                                                      dtype=np.int32)).view(-1, num_columns)

            edges = [train_edges, valid_edges, test_edges]
            edges = torch.cat(edges)

            if self.add_reverse_edges:
                train_mask = torch.zeros([edges.size(0) * 2], dtype=torch.bool)
                valid_mask = torch.zeros([edges.size(0) * 2], dtype=torch.bool)
                test_mask = torch.zeros([edges.size(0) * 2], dtype=torch.bool)

                train_mask[:dataset_stats.num_train] = True
                valid_mask[dataset_stats.num_train:dataset_stats.num_train + dataset_stats.num_valid] = True
                test_mask[dataset_stats.num_train + dataset_stats.num_valid:dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_train] = True
            else:
                train_mask = torch.zeros([edges.size(0)], dtype=torch.bool)
                valid_mask = torch.zeros([edges.size(0)], dtype=torch.bool)
                test_mask = torch.zeros([edges.size(0)], dtype=torch.bool)

                train_mask[:dataset_stats.num_train] = True
                valid_mask[dataset_stats.num_train:dataset_stats.num_train + dataset_stats.num_valid] = True
                test_mask[dataset_stats.num_train + dataset_stats.num_valid:] = True


        edges_src = edges[:, 0]
        edges_dst = edges[:, -1]

        if self.add_reverse_edges:
            new_edges_src = torch.cat([edges_src, edges_dst])
            new_edges_dst = torch.cat([edges_dst, edges_src])

            edges_src = new_edges_src
            edges_dst = new_edges_dst

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=dataset_stats.num_nodes)

        if num_columns == 3:
            edges_type = edges[:, 1]
            if self.add_reverse_edges:
                edges_type = torch.cat([edges_type, edges_type])
                self.graph.edata['etype'] = edges_type
            else:
                self.graph.edata['etype'] = edges_type

        if not self.only_train:
            self.graph.edata['train_mask'] = train_mask
            self.graph.edata['val_mask'] = valid_mask
            self.graph.edata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class MariusNCDataset(object):

    def __init__(self, base_directory, add_reverse_edges=True, only_sample=False):
        self.base_directory = base_directory
        self.add_reverse_edges = add_reverse_edges
        self.only_sample = only_sample

    def process(self):
        base_directory = self.base_directory.rstrip("/")

        dataset_stats = OmegaConf.load(base_directory + "/dataset.yaml")

        num_columns = 3
        if dataset_stats.num_relations == 1:
            num_columns = 2

        train_edges = torch.from_numpy(np.fromfile(base_directory + "/edges/train_edges.bin",
                                                   dtype=np.int32)).view(-1, num_columns)

        edges_src = train_edges[:, 0]
        edges_dst = train_edges[:, -1]

        if self.add_reverse_edges:
            new_edges_src = torch.cat([edges_src, edges_dst])
            new_edges_dst = torch.cat([edges_dst, edges_src])

            edges_src = new_edges_src
            edges_dst = new_edges_dst

        train_nodes = torch.from_numpy(np.fromfile(base_directory + "/nodes/train_nodes.bin", dtype=np.int32))
        valid_nodes = torch.from_numpy(np.fromfile(base_directory + "/nodes/validation_nodes.bin", dtype=np.int32))
        test_nodes = torch.from_numpy(np.fromfile(base_directory + "/nodes/test_nodes.bin", dtype=np.int32))

        num_nodes = dataset_stats.num_nodes
        num_relations = dataset_stats.num_relations

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=dataset_stats.num_nodes)

        if num_columns == 3:
            edges_type = train_edges[:, 1]
            if self.add_reverse_edges:
                edges_type = torch.cat([edges_type, edges_type])
                self.graph.edata['etype'] = edges_type
            else:
                self.graph.edata['etype'] = edges_type
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_nodes.to(torch.int64)] = True
        val_mask[valid_nodes.to(torch.int64)] = True
        test_mask[test_nodes.to(torch.int64)] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        if not self.only_sample:
            features = torch.from_numpy(np.fromfile(base_directory + "/nodes/features.bin",
                                                    dtype=np.float32)).view(-1, dataset_stats.feature_dim)
            labels = torch.from_numpy(np.fromfile(base_directory + "/nodes/labels.bin", dtype=np.int32))

            self.graph.ndata['feat'] = features
            self.graph.ndata['label'] = labels

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def get_marius_dataset_lp(base_directory, add_reverse_edges, only_train=False):
    dataset = MariusLPDataset(base_directory, add_reverse_edges, only_train=False)
    dataset.process()
    return dataset[0]


def get_marius_dataset_nc(base_directory, add_reverse_edges, only_sample=False):
    dataset = MariusNCDataset(base_directory, add_reverse_edges, only_sample)
    dataset.process()
    return dataset[0]


def select_dataset(base_directory, learning_task, add_reverse_edges):
    if learning_task.upper() == "LINK_PREDICTION":
        return get_marius_dataset_lp(base_directory, add_reverse_edges)
    else:
        return get_marius_dataset_nc(base_directory, add_reverse_edges)
