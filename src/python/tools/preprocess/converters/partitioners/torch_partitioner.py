import numpy as np

from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner

import torch  # isort:skip


def dataframe_to_tensor(df):
    return torch.tensor(df.to_numpy())


def partition_edges(edges, num_nodes, num_partitions, edge_weights=None):
    partition_size = int(np.ceil(num_nodes / num_partitions))

    src_partitions = torch.div(edges[:, 0], partition_size, rounding_mode="trunc")
    dst_partitions = torch.div(edges[:, -1], partition_size, rounding_mode="trunc")

    _, dst_args = torch.sort(dst_partitions, stable=True)
    _, src_args = torch.sort(src_partitions[dst_args], stable=True)
    sort_order = dst_args[src_args]

    edges = edges[sort_order]
    if edge_weights is not None:
        edge_weights = edge_weights[sort_order]

    edge_bucket_ids = torch.div(edges, partition_size, rounding_mode="trunc")
    offsets = np.zeros([num_partitions, num_partitions], dtype=int)
    unique_src, num_source = torch.unique_consecutive(edge_bucket_ids[:, 0], return_counts=True)

    num_source_offsets = torch.cumsum(num_source, 0) - num_source

    curr_src_unique = 0
    for i in range(num_partitions):
        if curr_src_unique < unique_src.size(0) and unique_src[curr_src_unique] == i:
            offset = num_source_offsets[curr_src_unique]
            num_edges = num_source[curr_src_unique]
            dst_ids = edge_bucket_ids[offset : offset + num_edges, -1]

            unique_dst, num_dst = torch.unique_consecutive(dst_ids, return_counts=True)

            offsets[i][unique_dst] = num_dst
            curr_src_unique += 1

    offsets = list(offsets.flatten())

    return edges, offsets, edge_weights


class TorchPartitioner(Partitioner):
    def __init__(self, partitioned_evaluation):
        super().__init__()

        self.partitioned_evaluation = partitioned_evaluation

    def partition_edges(
        self, train_edges_tens, valid_edges_tens, test_edges_tens, num_nodes, num_partitions, edge_weights=None
    ):
        # Extract the edge weights
        train_edge_weights, valid_edge_weights, test_edge_weights = None, None, None
        if edge_weights is not None:
            train_edge_weights, valid_edge_weights, test_edge_weights = (
                edge_weights[0],
                edge_weights[1],
                edge_weights[2],
            )

        train_edges_tens, train_offsets, train_edge_weights = partition_edges(
            train_edges_tens, num_nodes, num_partitions, edge_weights=train_edge_weights
        )

        valid_offsets = None
        test_offsets = None

        if self.partitioned_evaluation:
            if valid_edges_tens is not None:
                valid_edges_tens, valid_offsets, valid_edge_weights = partition_edges(
                    valid_edges_tens, num_nodes, num_partitions, edge_weights=valid_edge_weights
                )

            if test_edges_tens is not None:
                test_edges_tens, test_offsets, test_edge_weights = partition_edges(
                    test_edges_tens, num_nodes, num_partitions, edge_weights=test_edge_weights
                )

        return (
            train_edges_tens,
            train_offsets,
            valid_edges_tens,
            valid_offsets,
            test_edges_tens,
            test_offsets,
            [train_edge_weights, valid_edge_weights, test_edge_weights],
        )
