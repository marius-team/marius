import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.partitioners.torch_partitioner import TorchPartitioner
from marius.tools.preprocess.converters.readers.pandas_readers import PandasDelimitedFileReader
from marius.tools.preprocess.converters.writers.torch_writer import TorchWriter

SUPPORTED_DELIM_FORMATS = ["CSV", "TSV", "TXT", "DELIM", "DELIMITED"]
SUPPORTED_IN_MEMORY_FORMATS = ["NUMPY", "NP", "PYTORCH", "TORCH"]


def dataframe_to_tensor(df):
    return torch.tensor(df.to_numpy())


def apply_mapping_edges(input_edges, node_mapping_df, rel_mapping_df=None):
    if isinstance(input_edges, torch.Tensor):
        assert len(input_edges.shape) == 2

        src = input_edges[:, 0]
        dst = input_edges[:, -1]

        src = apply_mapping1d(src, node_mapping_df)
        dst = apply_mapping1d(dst, node_mapping_df)

        stack_tens = []
        if rel_mapping_df is None:
            assert input_edges.shape[1] == 2
            stack_tens = [src, dst]
        else:
            assert input_edges.shape[1] == 3
            rel = input_edges[:, 1]
            rel = apply_mapping1d(rel, rel_mapping_df)
            stack_tens = [src, rel, dst]

        return torch.stack(stack_tens, dim=1)

    elif isinstance(input_edges, pd.DataFrame):
        src = input_edges.iloc[:, 0]
        dst = input_edges.iloc[:, -1]

        src = apply_mapping1d(src, node_mapping_df)
        dst = apply_mapping1d(dst, node_mapping_df)

        concat_df = []
        if rel_mapping_df is None:
            assert input_edges.shape[1] == 2

            concat_df = [src, dst]
        else:
            assert input_edges.shape[1] == 3
            rel = input_edges[:, 1]
            rel = apply_mapping1d(rel, rel_mapping_df)

            concat_df = [src, rel, dst]

        return pd.concat(concat_df, axis=1)
    else:
        raise RuntimeError("Unsupported datatype for input. Must be a pandas.DataFrame or a 2D torch.Tensor")


def apply_mapping1d(input_ids, mapping_df):
    if isinstance(input_ids, torch.Tensor):
        assert len(input_ids.shape) == 1
        mapping = dataframe_to_tensor(mapping_df)
        return mapping[:, 1][input_ids]
    elif isinstance(input_ids, pd.Series):
        return input_ids.map(mapping_df.iloc[:, 1])
    else:
        raise RuntimeError("Unsupported datatype for input. Must be a pandas.Series or a 1D torch.Tensor")


def map_edge_list_dfs(edge_lists: list, known_node_ids=None, sequential_train_nodes=False, sequential_deg_nodes=0):
    if sequential_train_nodes or sequential_deg_nodes > 0:
        raise RuntimeError("sequential_train_nodes not yet supported for map_edge_list_dfs")

    all_edges_df = pd.concat(edge_lists)

    unique_src = all_edges_df.iloc[:, 0].unique()
    unique_dst = all_edges_df.iloc[:, -1].unique()

    if known_node_ids is None:
        unique_nodes = np.unique(np.concatenate([unique_src.astype(str), unique_dst.astype(str)]))
    else:
        node_ids = [unique_src.astype(str), unique_dst.astype(str)]
        for n in known_node_ids:
            node_ids.append(n.numpy().astype(str))

        unique_nodes = np.unique(np.concatenate(node_ids))

    num_nodes = unique_nodes.shape[0]
    mapped_node_ids = np.random.permutation(num_nodes)
    nodes_dict = dict(zip(list(unique_nodes), list(mapped_node_ids)))

    has_rels = False
    unique_rels = torch.empty([0])
    mapped_rel_ids = torch.empty([0])
    rels_dict = None
    if len(all_edges_df.columns) == 3:
        has_rels = True

    if has_rels:
        unique_rels = all_edges_df.iloc[:, 1].unique()
        num_rels = unique_rels.shape[0]
        mapped_rel_ids = np.random.permutation(num_rels)
        rels_dict = dict(zip(list(unique_rels), list(mapped_rel_ids)))

    all_edges_df = None  # can safely free this df

    output_edge_lists = []
    for edge_list in edge_lists:
        node_columns = edge_list.columns[[0, -1]]
        edge_list[node_columns] = edge_list[node_columns].applymap(nodes_dict.get)

        if has_rels:
            rel_columns = edge_list.columns[1]
            edge_list[rel_columns] = edge_list[rel_columns].map(rels_dict.get)

        output_edge_lists.append(dataframe_to_tensor(edge_list))

    node_mapping = np.stack([unique_nodes, mapped_node_ids], axis=1)
    rel_mapping = None
    if has_rels:
        rel_mapping = np.stack([unique_rels, mapped_rel_ids], axis=1)
    return output_edge_lists, node_mapping, rel_mapping


def map_edge_lists(
    edge_lists: list, perform_unique=True, known_node_ids=None, sequential_train_nodes=False, sequential_deg_nodes=0
):
    print("Remapping Edges")

    defined_edges = []
    for edge_list in edge_lists:
        if edge_list is not None:
            defined_edges.append(edge_list)

    edge_lists = defined_edges

    if isinstance(edge_lists[0], pd.DataFrame):
        if isinstance(edge_lists[0].iloc[0][0], str):
            # need to take uniques using pandas for string datatypes, since torch doesn't support strings
            return map_edge_list_dfs(edge_lists, known_node_ids, sequential_train_nodes, sequential_deg_nodes)

        new_edge_lists = []
        for edge_list in edge_lists:
            new_edge_lists.append(dataframe_to_tensor(edge_list))

        edge_lists = new_edge_lists

    all_edges = torch.cat(edge_lists)

    has_rels = False
    num_rels = 1
    unique_rels = torch.empty([0])
    mapped_rel_ids = torch.empty([0])
    if all_edges.size(1) == 3:
        has_rels = True

    output_dtype = torch.int32

    if perform_unique:
        unique_src = torch.unique(all_edges[:, 0])
        unique_dst = torch.unique(all_edges[:, -1])
        if known_node_ids is None:
            unique_nodes = torch.unique(torch.cat([unique_src, unique_dst]), sorted=True)
        else:
            unique_nodes = torch.unique(torch.cat([unique_src, unique_dst] + known_node_ids), sorted=True)

        num_nodes = unique_nodes.size(0)

        if has_rels:
            unique_rels = torch.unique(all_edges[:, 1], sorted=True)
            num_rels = unique_rels.size(0)
    else:
        num_nodes = torch.max(all_edges[:, 0])[0]
        unique_nodes = torch.arange(num_nodes).to(output_dtype)

        if has_rels:
            num_rels = torch.max(all_edges[:, 1])[0]
            unique_rels = torch.arange(num_rels).to(output_dtype)

    if sequential_train_nodes or sequential_deg_nodes > 0:
        seq_nodes = None

        if sequential_train_nodes and sequential_deg_nodes <= 0:
            print("Sequential Train Nodes")
            seq_nodes = known_node_ids[0]
        else:
            out_degrees = torch.zeros(
                [
                    num_nodes,
                ],
                dtype=torch.int32,
            )
            out_degrees = torch.scatter_add(
                out_degrees,
                0,
                torch.squeeze(edge_lists[0][:, 0]).to(torch.int64),
                torch.ones(
                    [
                        edge_lists[0].shape[0],
                    ],
                    dtype=torch.int32,
                ),
            )

            in_degrees = torch.zeros(
                [
                    num_nodes,
                ],
                dtype=torch.int32,
            )
            in_degrees = torch.scatter_add(
                in_degrees,
                0,
                torch.squeeze(edge_lists[0][:, -1]).to(torch.int64),
                torch.ones(
                    [
                        edge_lists[0].shape[0],
                    ],
                    dtype=torch.int32,
                ),
            )

            degrees = in_degrees + out_degrees

            deg_argsort = torch.argsort(degrees, dim=0, descending=True)
            high_degree_nodes = deg_argsort[:sequential_deg_nodes]

            print("High Deg Nodes Degree Sum:", torch.sum(degrees[high_degree_nodes]).numpy())

            if sequential_train_nodes and sequential_deg_nodes > 0:
                print("Sequential Train and High Deg Nodes")
                seq_nodes = torch.unique(torch.cat([high_degree_nodes, known_node_ids[0]]))
                seq_nodes = seq_nodes.index_select(0, torch.randperm(seq_nodes.size(0), dtype=torch.int64))
                print("Total Seq Nodes: ", seq_nodes.shape[0])
            else:
                print("Sequential High Deg Nodes")
                seq_nodes = high_degree_nodes

        seq_mask = torch.zeros(num_nodes, dtype=torch.bool)
        seq_mask[seq_nodes.to(torch.int64)] = True
        all_other_nodes = torch.arange(num_nodes, dtype=seq_nodes.dtype)
        all_other_nodes = all_other_nodes[~seq_mask]

        mapped_node_ids = -1 * torch.ones(num_nodes, dtype=output_dtype)
        mapped_node_ids[seq_nodes.to(torch.int64)] = torch.arange(seq_nodes.shape[0], dtype=output_dtype)
        mapped_node_ids[all_other_nodes.to(torch.int64)] = seq_nodes.shape[0] + torch.randperm(
            num_nodes - seq_nodes.shape[0], dtype=output_dtype
        )
    else:
        mapped_node_ids = torch.randperm(num_nodes, dtype=output_dtype)

    if has_rels:
        mapped_rel_ids = torch.randperm(num_rels, dtype=output_dtype)

    # TODO may use too much memory if the max id is very large
    # Needed to support indexing w/ the remap
    if torch.max(unique_nodes) + 1 > num_nodes:
        extended_map = torch.zeros(torch.max(unique_nodes) + 1, dtype=output_dtype)
        extended_map[unique_nodes] = mapped_node_ids
    else:
        extended_map = mapped_node_ids

    all_edges = None  # can safely free this tensor

    output_edge_lists = []
    for edge_list in edge_lists:
        new_src = extended_map[edge_list[:, 0].to(torch.int64)]
        new_dst = extended_map[edge_list[:, -1].to(torch.int64)]

        if has_rels:
            new_rel = mapped_rel_ids[edge_list[:, 1].to(torch.int64)]
            output_edge_lists.append(torch.stack([new_src, new_rel, new_dst], dim=1))
        else:
            output_edge_lists.append(torch.stack([new_src, new_dst], dim=1))

    node_mapping = np.stack([unique_nodes.numpy(), mapped_node_ids.numpy()], axis=1)
    rel_mapping = None
    if has_rels:
        rel_mapping = np.stack([unique_rels.numpy(), mapped_rel_ids.numpy()], axis=1)

    return output_edge_lists, node_mapping, rel_mapping


def split_edges(edges, splits):
    train_edges_tens = None
    valid_edges_tens = None
    test_edges_tens = None

    total_split_edges = int(sum(splits) * edges.shape[0])

    num_total_edges = edges.shape[0]
    rand_perm = torch.randperm(num_total_edges)

    if len(splits) == 3:
        train_split = splits[0]
        valid_split = splits[1]
        test_split = splits[2]

        print("Splitting into: {}/{}/{} fractions".format(train_split, valid_split, test_split))

        num_train = int(num_total_edges * train_split)
        num_valid = int(num_total_edges * valid_split)

        train_edges_tens = edges[rand_perm[:num_train]]
        valid_edges_tens = edges[rand_perm[num_train : num_train + num_valid]]
        test_edges_tens = edges[rand_perm[num_train + num_valid : total_split_edges]]
    elif len(splits) == 2:
        train_split = splits[0]
        test_split = splits[1]

        print("Splitting into: {}/{} fractions".format(train_split, test_split))

        num_train = int(num_total_edges * train_split)

        train_edges_tens = edges[rand_perm[:num_train]]
        test_edges_tens = edges[rand_perm[num_train:total_split_edges]]
    else:
        raise RuntimeError("Splits must be length 2 or 3")

    return train_edges_tens, valid_edges_tens, test_edges_tens


class TorchEdgeListConverter(object):
    def __init__(
        self,
        output_dir: Path,
        train_edges: Path,
        valid_edges: Path = None,
        test_edges: Path = None,
        splits: list = None,
        format: str = "csv",
        columns: list = [0, 1, 2],
        header_length: int = 0,
        delim: str = "\t",
        dtype: str = "int32",
        num_partitions: int = 1,
        partitioned_evaluation: bool = False,
        remap_ids: bool = True,
        sequential_train_nodes: bool = False,
        sequential_deg_nodes: int = 0,
        num_nodes: int = None,
        num_rels: int = None,
        known_node_ids: list = None,
    ):
        """
        This converter is used to preprocess input edge lists which fit in memory. Pandas, numpy and pytorch are used to convert input edge lists that are
        stored as delimited files, numpy arrays, or pytorch tensors into the input format required by Marius.

        Steps of conversion process:
        1. Read in input dataset and convert to a pytorch tensor
        2. Remap node and relation ids to randomly assigned integer ids (optional). Write mappings to the output directory.
        3. Perform data set splitting into train/valid/test sets (optional)
        4. Reorder/partition edge list(s) according to their edge buckets (optional)
        5. Write contents of the edge list(s) tensors to a file in the specified output directory

        Output format:
        The output format is as follows

        <output_dir>/
            edges/
                train_edges.bin                                 Binary file of size num_train * 2 * sizeof(dtype) or num_train * 3 * sizeof(dtype)
                train_partition_offsets.txt     (optional)      List of training edge bucket sizes in sequential order (0, 0), (0, 1) ... (1, 0), ... (n-1, n-1)
                valid_edges.bin                 (optional)      Binary file of size num_valid * 2 * sizeof(dtype) or num_valid * 3 * sizeof(dtype)
                valid_partition_offsets.txt     (optional)      List of validation edge bucket sizes in sequential order (0, 0), (0, 1) ... (1, 0), ... (n-1, n-1)
                test_edges.bin                  (optional)      Binary file of size num_test * 2 * sizeof(dtype) or num_test * 3 * sizeof(dtype)
                test_partition_offsets.txt      (optional)      List of test edge bucket sizes in sequential order (0, 0), (0, 1) ... (1, 0), ... (n-1, n-1)
                relation_mapping.txt            (optional)      Two column CSV containing a mapping of raw relation/edge-type ids (1st column) to randomly assigned integer ids (2nd column).
            nodes/
                node_mapping.txt                (optional)      Two column CSV containing a mapping of raw node ids (1st column) to randomly assigned integer ids (2nd column).
            dataset.yaml                                        Output dataset statistics in YAML format.

        :param output_dir:   (required)         Directory which will contain the preprocessed dataset
        :param train_edges:  (required)         Raw input training edges, can be a delimited file, a numpy array, or pytorch tensor
        :param valid_edges:                     Raw input validation edges, can be a delimited file, a numpy array, or pytorch tensor (optional)
        :param test_edges:                      Raw input test edges, can be a delimited file, a numpy array, or pytorch tensor (optional)
        :param splits:                          Train/valid/test split to use for the input
        :param format:                          Format of the input dataset, can be a delimited file (CSV, TSV, TXT) or a numpy array or a pytorch tensor.
        :param columns:                         List of column ids of input delimited files which denote the src node, edge-type, and dst node of edges.
                                                E.g. columns=[0, 2, 1] means that the source nodes are found in the first column, the edge-types are found in
                                                in the third column, and the destination nodes are found in the second column.
                                                For datasets without edge-types, only two column ids should be specified. E.g. columns=[0, 2]
        :param header_length:                   Length of the header for input delimited files
        :param delim:                           Delimiter of the input delimited files
        :param dtype:                           Datatype of the node ids in the output preprocessed datasets. Unless you have over 2 billion nodes, this should
                                                stay as int32/
        :param num_partitions:                  Number of node partitions which will be used to train the model with the partition buffer. Setting this will
                                                reorder the output edge list(s) according to the num_partitions^2 edge buckets in a sequential order.
                                                E.g. edge bucket (0,0) will be first, then (0, 1), (0, 2) ... (0, n-1), (1, 0) .... (1, n-1), ... (n-1, 0) ... (n-1, n-1).
                                                The sizes of the edge buckets are stored in <output_dir>/edges/<type>_partition_offsets.txt
        :param partitioned_evaluation:          If true, the edge buckets for the validation and test sets will be computed and the edge lists will be reordered.
        :param remap_ids:                       If true, then the raw entity ids of the input edge lists will be remapped to random integer ids. The mapping of
                                                the node ids is stored as a two column CSV in <output_dir>/nodes/node_mapping.txt
        :param sequential_train_nodes           If true, the train nodes will be given ids 0 to num train nodes. Applicable to node classification datasets. If set,
                                                remap_ids must also be set.
        :param sequential_deg_nodes             If greater than zero, this number of the highest degree nodes based on the train edges will be given ids 0 to this number. If
                                                greater than zero, remap_ids must also be set. Can be mixed with sequential_train_nodes, in which case train and high deg nodes
                                                are given ids starting from 0.
        :param num_nodes:                       Number of nodes in the dataset, this is required when remap_ids is set to false.
        :param num_rels:                        Number of nodes in the dataset, this is required when remap_ids is set to false and the dataset has edge_types
        :param known_node_ids:                  List of node id arrays or tensors which contain known node ids for the dataset. Used for generating node id mappings
                                                when some nodes may not be present in the edge list.
        """  # noqa: E501
        self.output_dir = output_dir
        self.num_nodes = num_nodes
        self.num_rels = num_rels

        if format.upper() in SUPPORTED_DELIM_FORMATS:
            assert isinstance(train_edges, str) or isinstance(train_edges, Path)

            self.reader = PandasDelimitedFileReader(train_edges, valid_edges, test_edges, columns, header_length, delim)

        elif format.upper() in SUPPORTED_IN_MEMORY_FORMATS:
            self.reader = None
            if format.upper() == "NUMPY":
                assert isinstance(train_edges, np.ndarray)
                self.train_edges_tens = torch.from_numpy(train_edges)
                self.valid_edges_tens = None
                self.test_edges_tens = None

                if valid_edges is not None:
                    assert isinstance(valid_edges, np.ndarray)
                    self.valid_edges_tens = torch.from_numpy(valid_edges)

                if test_edges is not None:
                    assert isinstance(test_edges, np.ndarray)
                    self.test_edges_tens = torch.from_numpy(test_edges)
            elif format.upper() == "PYTORCH":
                assert isinstance(train_edges, torch.Tensor)
                self.train_edges_tens = train_edges
                self.valid_edges_tens = valid_edges
                self.test_edges_tens = test_edges

                if valid_edges is not None:
                    assert isinstance(valid_edges, torch.Tensor)

                if test_edges is not None:
                    assert isinstance(test_edges, torch.Tensor)
        else:
            raise RuntimeError("Unsupported input format")
        self.num_partitions = num_partitions

        if self.num_partitions > 1:
            self.partitioner = TorchPartitioner(partitioned_evaluation)
        else:
            self.partitioner = None

        self.writer = TorchWriter(self.output_dir, partitioned_evaluation)

        self.splits = splits

        self.has_rels = False
        if len(columns) == 3:
            self.has_rels = True

        if dtype.upper() == "INT32" or dtype.upper() == "INT":
            self.dtype = torch.int32
        elif dtype.upper() == "INT64" or dtype.upper() == "LONG":
            self.dtype = torch.int64
        else:
            raise RuntimeError("Unrecognized datatype")

        self.remap_ids = remap_ids

        if self.num_nodes is None and not self.remap_ids:
            raise RuntimeError(
                "Must specify num_nodes and num_rels (if applicable) to the converter when remap_ids=False"
            )

        if self.num_rels is None and not self.remap_ids and self.has_rels:
            raise RuntimeError(
                "Must specify num_nodes and num_rels (if applicable) to the converter when remap_ids=False"
            )

        self.sequential_train_nodes = sequential_train_nodes

        if self.sequential_train_nodes is True and self.remap_ids is False:
            raise RuntimeError("remap_ids must be true when sequential_train_nodes is true")

        self.sequential_deg_nodes = sequential_deg_nodes

        if self.sequential_deg_nodes > 0 and self.remap_ids is False:
            raise RuntimeError("remap_ids must be true when sequential_deg_nodes is greater than zero")

        if known_node_ids is not None:
            self.known_node_ids = []
            for node_id in known_node_ids:
                if node_id is not None:
                    if isinstance(node_id, np.ndarray):
                        node_id = torch.from_numpy(node_id)

                    assert isinstance(node_id, torch.Tensor)
                    self.known_node_ids.append(node_id)
        else:
            self.known_node_ids = None

    def convert(self):
        train_edges_tens = None
        valid_edges_tens = None
        test_edges_tens = None

        os.makedirs(self.output_dir / Path("nodes"), exist_ok=True)
        os.makedirs(self.output_dir / Path("edges"), exist_ok=True)

        print("Reading edges")
        if self.reader is not None:
            train_edges_df, valid_edges_df, test_edges_df = self.reader.read()

            if self.remap_ids:
                edge_lists, node_mapping, rel_mapping = map_edge_lists(
                    [train_edges_df, valid_edges_df, test_edges_df],
                    known_node_ids=self.known_node_ids,
                    sequential_train_nodes=self.sequential_train_nodes,
                    sequential_deg_nodes=self.sequential_deg_nodes,
                )

                self.num_nodes = node_mapping.shape[0]

                if rel_mapping is None:
                    self.num_rels = 1
                else:
                    self.num_rels = rel_mapping.shape[0]

                train_edges_tens = edge_lists[0]
                if len(edge_lists) == 2:
                    test_edges_tens = edge_lists[1]
                elif len(edge_lists) == 3:
                    valid_edges_tens = edge_lists[1]
                    test_edges_tens = edge_lists[2]

                print(
                    "Node mapping written to: {}".format(
                        (self.output_dir / Path(PathConstants.node_mapping_path)).__str__()
                    )
                )
                np.savetxt(
                    (self.output_dir / Path(PathConstants.node_mapping_path)).__str__(),
                    node_mapping,
                    fmt="%s",
                    delimiter=",",
                )

                if self.num_rels > 1:
                    print(
                        "Relation mapping written to: {}".format(
                            (self.output_dir / Path(PathConstants.relation_mapping_path)).__str__()
                        )
                    )
                    np.savetxt(
                        (self.output_dir / Path(PathConstants.relation_mapping_path)).__str__(),
                        rel_mapping,
                        fmt="%s",
                        delimiter=",",
                    )
            else:
                train_edges_tens = dataframe_to_tensor(train_edges_df)

                if valid_edges_df is not None:
                    valid_edges_tens = dataframe_to_tensor(valid_edges_df)

                if test_edges_df is not None:
                    test_edges_tens = dataframe_to_tensor(test_edges_df)
        else:
            train_edges_tens = self.train_edges_tens
            valid_edges_tens = self.valid_edges_tens
            test_edges_tens = self.test_edges_tens

            if self.remap_ids:
                edge_lists, node_mapping, rel_mapping = map_edge_lists(
                    [train_edges_tens, valid_edges_tens, test_edges_tens],
                    known_node_ids=self.known_node_ids,
                    sequential_train_nodes=self.sequential_train_nodes,
                    sequential_deg_nodes=self.sequential_deg_nodes,
                )

                self.num_nodes = node_mapping.shape[0]

                if rel_mapping is None:
                    self.num_rels = 1
                else:
                    self.num_rels = rel_mapping.shape[0]

                train_edges_tens = edge_lists[0]
                if len(edge_lists) == 2:
                    test_edges_tens = edge_lists[1]
                elif len(edge_lists) == 3:
                    valid_edges_tens = edge_lists[1]
                    test_edges_tens = edge_lists[2]

                print(
                    "Node mapping written to: {}".format(
                        (self.output_dir / Path(PathConstants.node_mapping_path)).__str__()
                    )
                )
                np.savetxt(
                    (self.output_dir / Path(PathConstants.node_mapping_path)).__str__(),
                    node_mapping,
                    fmt="%s",
                    delimiter=",",
                )

                if self.num_rels > 1:
                    print(
                        "Relation mapping written to: {}".format(
                            (self.output_dir / Path(PathConstants.relation_mapping_path)).__str__()
                        )
                    )
                    np.savetxt(
                        (self.output_dir / Path(PathConstants.relation_mapping_path)).__str__(),
                        rel_mapping,
                        fmt="%s",
                        delimiter=",",
                    )

        train_edges_tens = train_edges_tens.to(self.dtype)
        if valid_edges_tens is not None:
            valid_edges_tens = valid_edges_tens.to(self.dtype)
        if test_edges_tens is not None:
            test_edges_tens = test_edges_tens.to(self.dtype)

        if self.splits is not None:
            train_edges_tens, valid_edges_tens, test_edges_tens = split_edges(train_edges_tens, self.splits)

        if self.partitioner is not None:
            print("Partition nodes into {} partitions".format(self.num_partitions))
            (
                train_edges_tens,
                train_edges_offsets,
                valid_edges_tens,
                valid_edges_offsets,
                test_edges_tens,
                test_edges_offsets,
            ) = self.partitioner.partition_edges(
                train_edges_tens, valid_edges_tens, test_edges_tens, self.num_nodes, self.num_partitions
            )

            return self.writer.write_to_binary(
                train_edges_tens,
                valid_edges_tens,
                test_edges_tens,
                self.num_nodes,
                self.num_rels,
                self.num_partitions,
                train_edges_offsets,
                valid_edges_offsets,
                test_edges_offsets,
            )
        else:
            return self.writer.write_to_binary(
                train_edges_tens, valid_edges_tens, test_edges_tens, self.num_nodes, self.num_rels, self.num_partitions
            )
