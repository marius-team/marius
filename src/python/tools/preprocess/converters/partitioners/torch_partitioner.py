import numpy as np
import torch

from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner


def dataframe_to_tensor(input_dataframe):
    np_array = input_dataframe.to_dask_array().compute()
    return torch.from_numpy(np_array)


def partition_edges(edges, num_nodes, num_partitions):
    partition_size = int(np.ceil(num_nodes / num_partitions))

    src_partitions = torch.div(edges[:, 0], partition_size, rounding_mode="trunc")
    dst_partitions = torch.div(edges[:, -1], partition_size, rounding_mode="trunc")

    _, dst_args = torch.sort(dst_partitions, stable=True)
    _, src_args = torch.sort(src_partitions[dst_args], stable=True)

    edges = edges[dst_args[src_args]]
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

    return edges, offsets


class TorchPartitioner(Partitioner):
    def __init__(self, partitioned_evaluation):
        super().__init__()

        self.partitioned_evaluation = partitioned_evaluation

    def partition_edges(self, train_edges_tens, valid_edges_tens, test_edges_tens, num_nodes, num_partitions):
        """ """

        train_edges_tens, train_offsets = partition_edges(train_edges_tens, num_nodes, num_partitions)

        valid_offsets = None
        test_offsets = None

        if self.partitioned_evaluation:
            if valid_edges_tens is not None:
                valid_edges_tens, valid_offsets = partition_edges(valid_edges_tens, num_nodes, num_partitions)

            if test_edges_tens is not None:
                test_edges_tens, test_offsets = partition_edges(test_edges_tens, num_nodes, num_partitions)

        return train_edges_tens, train_offsets, valid_edges_tens, valid_offsets, test_edges_tens, test_offsets
