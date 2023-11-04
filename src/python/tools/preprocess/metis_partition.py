import argparse
import torch
import numpy as np

from omegaconf import OmegaConf

TRAIN_EDGES_PATH = "/edges/train_edges.bin"
VALID_EDGES_PATH = "/edges/validation_edges.bin"
TEST_EDGES_PATH = "/edges/test_edges.bin"
TRAIN_NODES_PATH = "/nodes/train_nodes.bin"
VALID_NODES_PATH = "/nodes/validation_nodes.bin"
TEST_NODES_PATH = "/nodes/test_nodes.bin"
FEATURES_PATH = "/nodes/features.bin"
LABELS_PATH = "/nodes/labels.bin"
EDGE_BUCKETS_PATH = "/edges/train_partition_offsets.txt"


class MariusDataset(object):

    def __init__(self, base_directory, learning_task='link_prediction', num_partitions=2):
        self.base_directory = base_directory.rstrip("/")
        self.learning_task = learning_task
        self.num_partitions = num_partitions

        self.num_columns = 3
        self.num_nodes = None

        self.train_edges = None
        self.valid_edges = None
        self.test_edges = None

        self.train_nodes = None
        self.valid_nodes = None
        self.test_nodes = None

        self.features = None
        self.labels = None

        self.edge_bucket_sizes = None

    def load(self):
        base_directory = self.base_directory

        dataset_stats = OmegaConf.load(base_directory + "/dataset.yaml")

        self.num_nodes = dataset_stats.num_nodes

        if dataset_stats.num_relations == 1:
            self.num_columns = 2

        num_columns = self.num_columns

        if self.learning_task == 'node_classification':
            self.train_edges = torch.from_numpy(np.fromfile(base_directory + TRAIN_EDGES_PATH, dtype=np.int32)).view((-1, num_columns))

            self.train_nodes = torch.from_numpy(np.fromfile(base_directory + TRAIN_NODES_PATH, dtype=np.int32))
            self.valid_nodes = torch.from_numpy(np.fromfile(base_directory + VALID_NODES_PATH, dtype=np.int32))
            self.test_nodes = torch.from_numpy(np.fromfile(base_directory + TEST_NODES_PATH, dtype=np.int32))

            self.features = torch.from_numpy(np.fromfile(base_directory + FEATURES_PATH, dtype=np.float32)).view((-1, dataset_stats.node_feature_dim))
            self.labels = torch.from_numpy(np.fromfile(base_directory + LABELS_PATH, dtype=np.int32))

        elif self.learning_task == 'link_prediction':
            self.train_edges = torch.from_numpy(np.fromfile(base_directory + TRAIN_EDGES_PATH, dtype=np.int32)).view((-1, num_columns))
            self.valid_edges = torch.from_numpy(np.fromfile(base_directory + VALID_EDGES_PATH, dtype=np.int32)).view((-1, num_columns))
            self.test_edges = torch.from_numpy(np.fromfile(base_directory + TEST_EDGES_PATH, dtype=np.int32)).view((-1, num_columns))

        else:
            raise Exception()

    def metis_partition(self):
        from partitioning_helpers import relabel_edges, pymetis_partitioning, add_missing_nodes, balance_parts, create_edge_buckets

        # partition based on the train_edges
        edges = self.train_edges.numpy()
        if self.num_columns == 3:
            edges = np.stack((edges[:, 0], edges[:, -1]), axis=1)

        edges, unique_nodes, node_mapping = relabel_edges(edges, self.num_nodes, return_map=True)
        num_unique = unique_nodes.shape[0]

        parts = pymetis_partitioning(self.num_partitions, num_unique, edges, 0)
        parts = add_missing_nodes(parts, self.num_nodes)
        parts = balance_parts(parts, np.ceil(self.num_nodes/self.num_partitions), None)
        edge_bucket_sizes, _, _ = create_edge_buckets(edges, parts, 0, plot=False)
        self.edge_bucket_sizes = edge_bucket_sizes.flatten()

        # put the nodes in partition order
        mapped_nodes_in_part_order = parts.argsort()
        second_map = np.zeros(self.num_nodes, dtype=parts.dtype) - 1
        second_map[mapped_nodes_in_part_order] = np.arange(0, self.num_nodes)
        node_mapping = second_map[node_mapping[np.arange(0, self.num_nodes)]]
        assert np.count_nonzero(node_mapping.flatten() == -1) == 0

        node_mapping = torch.from_numpy(node_mapping)
        node_mapping = node_mapping.to(torch.int32)

        train_edges_src = node_mapping[self.train_edges[:, 0]]
        train_edges_dest = node_mapping[self.train_edges[:, -1]]
        if self.num_columns == 3:
            self.train_edges = torch.stack((train_edges_src, self.train_edges[:, 1], train_edges_dest), dim=1)
        else:
            self.train_edges = torch.stack((train_edges_src, train_edges_dest), dim=1)

        # sort the edges into the edge_buckets
        indices = torch.argsort(self.train_edges[:, 0])
        self.train_edges = self.train_edges[indices]

        src_splits = torch.searchsorted(self.train_edges[:, 0].contiguous(),
                                        np.ceil(self.num_nodes/self.num_partitions) * torch.arange(self.num_partitions))
        for ii in range(self.num_partitions): # src partition index
            end_index = self.train_edges.shape[0] if ii == self.num_partitions - 1 else src_splits[ii+1]

            indices = torch.argsort(self.train_edges[src_splits[ii]:end_index, -1])
            self.train_edges[src_splits[ii]:end_index] = self.train_edges[src_splits[ii]:end_index][indices]

        if self.learning_task == "link_prediction":
            valid_edges_src = node_mapping[self.valid_edges[:, 0]]
            valid_edges_dest = node_mapping[self.valid_edges[:, -1]]
            if self.num_columns == 3:
                self.valid_edges = torch.stack((valid_edges_src, self.valid_edges[:, 1], valid_edges_dest), dim=1)
            else:
                self.valid_edges = torch.stack((valid_edges_src, valid_edges_dest), dim=1)

            test_edges_src = node_mapping[self.test_edges[:, 0]]
            test_edges_dest = node_mapping[self.test_edges[:, 2]]
            if self.num_columns == 3:
                self.test_edges = torch.stack((test_edges_src, self.test_edges[:, 1], test_edges_dest), dim=1)
            else:
                self.test_edges = torch.stack((test_edges_src, test_edges_dest), dim=1)
        else:
            self.train_nodes = node_mapping[self.train_nodes]
            self.valid_nodes = node_mapping[self.valid_nodes]
            self.test_nodes = node_mapping[self.test_nodes]

            random_map_argsort = torch.argsort(node_mapping)
            if self.features is not None:
                self.features = self.features[random_map_argsort]
            if self.labels is not None:
                self.labels = self.labels[random_map_argsort]

    def write(self):
        base_directory = self.base_directory

        with open(base_directory + TRAIN_EDGES_PATH, "wb") as f:
            f.write(bytes(self.train_edges.numpy()))

        if self.valid_edges is not None:
            with open(base_directory + VALID_EDGES_PATH, "wb") as f:
                f.write(bytes(self.valid_edges.numpy()))

        if self.test_edges is not None:
            with open(base_directory + TEST_EDGES_PATH, "wb") as f:
                f.write(bytes(self.test_edges.numpy()))

        if self.num_partitions > 1:
            with open(base_directory + EDGE_BUCKETS_PATH, "w") as f:
                f.writelines([str(o) + "\n" for o in self.edge_bucket_sizes])

            # if valid_edges_offsets is not None:
            #     with open(self.output_dir / Path(PathConstants.valid_edge_buckets_path), "w") as f:
            #         f.writelines([str(o) + "\n" for o in valid_edges_offsets])
            #
            # if test_edges_offsets is not None:
            #     with open(self.output_dir / Path(PathConstants.test_edge_buckets_path), "w") as f:
            #         f.writelines([str(o) + "\n" for o in test_edges_offsets])

        if self.train_nodes is not None:
            with open(base_directory + TRAIN_NODES_PATH, "wb") as f:
                f.write(bytes(self.train_nodes.numpy()))

        if self.valid_nodes is not None:
            with open(base_directory + VALID_NODES_PATH, "wb") as f:
                f.write(bytes(self.valid_nodes.numpy()))

        if self.test_nodes is not None:
            with open(base_directory + TEST_NODES_PATH, "wb") as f:
                f.write(bytes(self.test_nodes.numpy()))

        if self.features is not None:
            with open(base_directory + FEATURES_PATH, "wb") as f:
                f.write(bytes(self.features.numpy()))

        if self.labels is not None:
            with open(base_directory + LABELS_PATH, "wb") as f:
                f.write(bytes(self.labels.numpy()))



def main(args):
    dataset = MariusDataset(args.dataset_dir, learning_task=args.learning_task, num_partitions=args.num_partitions)
    dataset.load()
    dataset.metis_partition()
    dataset.write()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="path to dataset directory")
    p.add_argument("--learning_task", default="link_prediction", type=str)
    p.add_argument("--num_partitions", default=2, type=int)

    main(p.parse_args())