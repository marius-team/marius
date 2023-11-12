import os
from pathlib import Path

import numpy as np
import pandas as pd

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.partitioners.torch_partitioner import TorchPartitioner
from marius.tools.preprocess.converters.readers.pandas_readers import PandasDelimitedFileReader
from marius.tools.preprocess.converters.torch_constants import TorchConverterColumnKeys as ColNames
from marius.tools.preprocess.converters.writers.torch_writer import TorchWriter

import torch  # isort:skip

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


def extract_tensors_from_df(df, column_mappings):
    if df is None:
        return None, None

    edge_weight_tensor = None
    edge_weight_column_num = column_mappings[ColNames.EDGE_WEIGHT_COL]
    edge_weight_column_name = ColNames.EDGE_WEIGHT_COL.value

    if edge_weight_column_num is not None:
        assert edge_weight_column_name in list(df.columns)
        edge_weight_tensor = torch.tensor(df[edge_weight_column_name].values)
        df = df.drop(columns=[edge_weight_column_name])

    edges_tensor = dataframe_to_tensor(df)
    return edges_tensor, edge_weight_tensor


def map_edge_list_dfs(
    edge_lists: list,
    known_node_ids=None,
    sequential_train_nodes=False,
    sequential_deg_nodes=0,
    column_mappings: dict = {},
):
    if sequential_train_nodes or sequential_deg_nodes > 0:
        raise RuntimeError("sequential_train_nodes not yet supported for map_edge_list_dfs")

    # Combine all the non null dfs
    combined_dfs = []
    has_rels = column_mappings[ColNames.EDGE_TYPE_COL] is not None
    for edge_df in edge_lists:
        if edge_df is not None:
            # Convert all columns to str
            edge_df[ColNames.SRC_COL.value] = edge_df[ColNames.SRC_COL.value].astype(str)
            edge_df[ColNames.DST_COL.value] = edge_df[ColNames.DST_COL.value].astype(str)
            if has_rels:
                edge_df[ColNames.EDGE_TYPE_COL.value] = edge_df[ColNames.EDGE_TYPE_COL.value].astype(str)
            combined_dfs.append(edge_df)

    # Get the unique nodes
    all_edges_df = pd.concat(combined_dfs)
    unique_src = all_edges_df[ColNames.SRC_COL.value].unique().astype(str)
    unique_dst = all_edges_df[ColNames.DST_COL.value].unique().astype(str)

    unique_list = [unique_src, unique_dst]
    if known_node_ids is not None:
        for n in known_node_ids:
            unique_list.append(n.numpy().astype(str))

    unique_nodes = np.unique(np.concatenate(unique_list, axis=None))
    num_nodes = unique_nodes.shape[0]
    mapped_node_ids = np.random.permutation(num_nodes)
    nodes_dict = dict(zip(list(unique_nodes), list(mapped_node_ids)))

    unique_rels = torch.empty([0])
    mapped_rel_ids = torch.empty([0])
    rels_dict = None

    if has_rels:
        unique_rels = all_edges_df[ColNames.EDGE_TYPE_COL.value].unique()
        num_rels = unique_rels.shape[0]
        mapped_rel_ids = np.random.permutation(num_rels)
        rels_dict = dict(zip(list(unique_rels), list(mapped_rel_ids)))

    output_edge_lists, output_edge_weights = [], []
    for edge_list in edge_lists:
        if edge_list is None:
            output_edge_lists.append(None)
            output_edge_weights.append(None)
            continue

        # Map the src and dst values
        edge_list[ColNames.SRC_COL.value] = edge_list[ColNames.SRC_COL.value].map(nodes_dict)
        assert edge_list[ColNames.SRC_COL.value].isna().sum() == 0

        edge_list[ColNames.DST_COL.value] = edge_list[ColNames.DST_COL.value].map(nodes_dict)
        assert edge_list[ColNames.DST_COL.value].isna().sum() == 0

        if has_rels:
            edge_list[ColNames.EDGE_TYPE_COL.value] = edge_list[ColNames.EDGE_TYPE_COL.value].map(rels_dict)
            assert edge_list[ColNames.EDGE_TYPE_COL.value].isna().sum() == 0

        edge_tensor, edge_weights = extract_tensors_from_df(edge_list, column_mappings)
        output_edge_lists.append(edge_tensor)
        output_edge_weights.append(edge_weights)

    node_mapping = np.stack([unique_nodes, mapped_node_ids], axis=1)
    rel_mapping = None
    if has_rels:
        rel_mapping = np.stack([unique_rels, mapped_rel_ids], axis=1)

    return output_edge_lists, node_mapping, rel_mapping, output_edge_weights


def extract_tensor_from_tens(edges_tensor, column_mappings):
    if edges_tensor is None:
        return None, None

    edge_weights_column = column_mappings[ColNames.EDGE_WEIGHT_COL]
    cols_to_keep = [column_mappings[ColNames.SRC_COL], column_mappings[ColNames.DST_COL]]
    if column_mappings[ColNames.EDGE_TYPE_COL] is not None:
        cols_to_keep.insert(len(cols_to_keep) - 1, column_mappings[ColNames.EDGE_TYPE_COL])

    converted_tensor = edges_tensor[:, cols_to_keep]
    converted_weights = None
    if edge_weights_column is not None:
        converted_weights = edges_tensor[:, edge_weights_column]

    return converted_tensor, converted_weights


def map_edge_lists(
    edge_lists: list,
    perform_unique=True,
    known_node_ids=None,
    sequential_train_nodes=False,
    sequential_deg_nodes=0,
    column_mappings: dict = {},
):
    print("Remapping node ids")

    # Ensure that we extract the edge weights as well that edge_lists are in [src, dst] or in [src, type, dst] order
    edge_weights_list = [None] * len(edge_lists)
    has_rels = column_mappings[ColNames.EDGE_TYPE_COL] is not None
    all_edges = []
    if isinstance(edge_lists[0], pd.DataFrame):
        first_df = edge_lists[0]
        if any(col_dtype != np.number for col_dtype in first_df.dtypes):
            # need to take uniques using pandas for string datatypes, since torch doesn't support strings
            return map_edge_list_dfs(
                edge_lists,
                known_node_ids,
                sequential_train_nodes,
                sequential_deg_nodes,
                column_mappings=column_mappings,
            )

        for idx in range(len(edge_lists)):
            edge_tensors, edge_weights = extract_tensors_from_df(edge_lists[idx], column_mappings)
            edge_lists[idx] = edge_tensors
            edge_weights_list[idx] = edge_weights
            if edge_tensors is not None:
                all_edges.append(edge_tensors)
    else:
        # Determine the order of tensors to keep
        for idx in range(len(edge_lists)):
            curr_edges = edge_lists[idx]
            if curr_edges is None:
                continue

            converted_edges, converted_weights = extract_tensor_from_tens(curr_edges, column_mappings)
            edge_lists[idx] = converted_edges
            all_edges.append(converted_edges)
            edge_weights_list[idx] = converted_weights

    all_edges = torch.cat(all_edges)
    num_rels = 1
    unique_rels = torch.empty([0])
    mapped_rel_ids = torch.empty([0])
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

    if has_rels:
        min_rel_val = unique_rels[0].to(torch.int64)

    if sequential_train_nodes or sequential_deg_nodes > 0:
        print("inside sequential mode because", sequential_train_nodes, sequential_deg_nodes)
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
    for idx, edge_list in enumerate(edge_lists):
        if edge_list is None:
            output_edge_lists.append(None)
            continue

        new_src = extended_map[edge_list[:, 0].to(torch.int64)]
        new_dst = extended_map[edge_list[:, -1].to(torch.int64)]
        curr_row = [new_src, new_dst]

        if has_rels:
            new_rel = mapped_rel_ids[edge_list[:, 1].to(torch.int64) - min_rel_val]
            curr_row.insert(len(curr_row) - 1, new_rel)
        output_edge_lists.append(torch.stack(curr_row, dim=1))

    node_mapping = np.stack([unique_nodes.numpy(), mapped_node_ids.numpy()], axis=1)
    rel_mapping = None
    if has_rels:
        rel_mapping = np.stack([unique_rels.numpy(), mapped_rel_ids.numpy()], axis=1)

    return output_edge_lists, node_mapping, rel_mapping, edge_weights_list


def split_edges(edges, edges_weights, splits):
    train_edges_tens, train_edges_weights = None, None
    valid_edges_tens, valid_edges_weights = None, None
    test_edges_tens, test_edges_weights = None, None

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

        if edges_weights is not None:
            train_edges_weights = edges_weights[rand_perm[:num_train]]
            valid_edges_weights = edges_weights[rand_perm[num_train : num_train + num_valid]]
            test_edges_weights = edges_weights[rand_perm[num_train + num_valid : total_split_edges]]

    elif len(splits) == 2:
        train_split = splits[0]
        test_split = splits[1]
        print("Splitting into: {}/{} fractions".format(train_split, test_split))

        num_train = int(num_total_edges * train_split)

        train_edges_tens = edges[rand_perm[:num_train]]
        test_edges_tens = edges[rand_perm[num_train:total_split_edges]]

        if edges_weights is not None:
            train_edges_weights = edges_weights[rand_perm[:num_train]]
            test_edges_weights = edges_weights[rand_perm[num_train:total_split_edges]]

    else:
        raise RuntimeError("Splits must be length 2 or 3")

    return (
        train_edges_tens,
        train_edges_weights,
        valid_edges_tens,
        valid_edges_weights,
        test_edges_tens,
        test_edges_weights,
    )


class TorchEdgeListConverter(object):
    def __init__(
        self,
        output_dir: Path,
        train_edges: Path,
        valid_edges: Path = None,
        test_edges: Path = None,
        splits: list = None,
        format: str = "csv",
        header_length: int = 0,
        delim: str = "\t",
        dtype: str = "int32",
        num_partitions: int = 1,
        partitioned_evaluation: bool = False,
        src_column: int = None,
        dst_column: int = None,
        edge_type_column: int = None,
        edge_weight_column: int = None,
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
                valid_edges.bin                 (optional)      Binary file of size num_valid * 2 * sizeof(dtype) or num_valid * 3 * sizeof(dtype) or num_valid * 3 * sizeof(dtype)
                                                                The ordering of the data is as as follows based on dataset breakdown:
                                                                    Both edge weights and edge types present: [src, type, weight, dst]
                                                                    Neither edge weight or edge type present: [src, dst]
                                                                    Only edge weight present: [src, weight, dst]
                                                                    Only edge type present: [src, type, dst]
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
        :param src_column:                      The column storing the src nodes.
        :param dst_column:                      The column storing the dst nodes.
        :param edge_type_column:                The column storing the edge type.
        :param edge_weight_column:              The column storing the edge weights.
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

        # Read in the src and dst column
        if src_column is None:
            raise ValueError("Src column must be specified with a non None value")

        if dst_column is None:
            raise ValueError("Dst column must be specified with a non None value")

        # Save these variables
        self.output_dir = output_dir
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.column_mappings = {
            ColNames.SRC_COL: src_column,
            ColNames.DST_COL: dst_column,
            ColNames.EDGE_TYPE_COL: edge_type_column,
            ColNames.EDGE_WEIGHT_COL: edge_weight_column,
        }

        if format.upper() in SUPPORTED_DELIM_FORMATS:
            assert isinstance(train_edges, str) or isinstance(train_edges, Path)
            self.reader = PandasDelimitedFileReader(
                train_edges, valid_edges, test_edges, self.column_mappings, header_length, delim
            )

        elif format.upper() in SUPPORTED_IN_MEMORY_FORMATS:
            self.reader = None
            if format.upper() == "NUMPY" or format.upper() == "NP":
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

            elif format.upper() == "PYTORCH" or format.upper() == "TORCH":
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

        # Determine if this has edge types
        self.has_rels = self.column_mappings[ColNames.EDGE_TYPE_COL] is not None
        if dtype.upper() == "INT32" or dtype.upper() == "INT":
            self.dtype = torch.int32
            self.weight_dtype = torch.float32
        elif dtype.upper() == "INT64" or dtype.upper() == "LONG":
            self.dtype = torch.int64
            self.weight_dtype = torch.float64
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

    # flake8: noqa: C901
    def convert(self):
        train_edges_tens, train_edge_weights = None, None
        valid_edges_tens, valid_edge_weights = None, None
        test_edges_tens, test_edge_weights = None, None

        os.makedirs(self.output_dir / Path("nodes"), exist_ok=True)
        os.makedirs(self.output_dir / Path("edges"), exist_ok=True)

        if self.reader is not None:
            print("Reading edges")
            train_edges_df, valid_edges_df, test_edges_df = self.reader.read()

            if self.remap_ids:
                all_edge_lists, node_mapping, rel_mapping, all_edge_weights = map_edge_lists(
                    [train_edges_df, valid_edges_df, test_edges_df],
                    known_node_ids=self.known_node_ids,
                    sequential_train_nodes=self.sequential_train_nodes,
                    sequential_deg_nodes=self.sequential_deg_nodes,
                    column_mappings=self.column_mappings,
                )

                self.num_nodes = node_mapping.shape[0]

                if rel_mapping is None:
                    self.num_rels = 1
                else:
                    self.num_rels = rel_mapping.shape[0]

                train_edges_tens = all_edge_lists[0]
                if len(all_edge_lists) == 2:
                    test_edges_tens = all_edge_lists[1]
                elif len(all_edge_lists) == 3:
                    valid_edges_tens = all_edge_lists[1]
                    test_edges_tens = all_edge_lists[2]

                train_edge_weights = all_edge_weights[0]
                valid_edge_weights = all_edge_weights[1]
                test_edge_weights = all_edge_weights[2]

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
                # Determine which columns to keep
                print("Not remapping node ids")

                # Extract all the tensors and weights
                train_edges_tens, train_edge_weights = extract_tensors_from_df(train_edges_df, self.column_mappings)
                valid_edges_tens, valid_edge_weights = extract_tensors_from_df(valid_edges_df, self.column_mappings)
                test_edges_tens, test_edge_weights = extract_tensors_from_df(test_edges_df, self.column_mappings)

        else:
            print("Using in memory data")
            train_edges_tens = self.train_edges_tens
            valid_edges_tens = self.valid_edges_tens
            test_edges_tens = self.test_edges_tens

            if self.remap_ids:
                all_edges_list, node_mapping, rel_mapping, all_edge_weights = map_edge_lists(
                    [train_edges_tens, valid_edges_tens, test_edges_tens],
                    known_node_ids=self.known_node_ids,
                    sequential_train_nodes=self.sequential_train_nodes,
                    sequential_deg_nodes=self.sequential_deg_nodes,
                    column_mappings=self.column_mappings,
                )

                self.num_nodes = node_mapping.shape[0]
                if rel_mapping is None:
                    self.num_rels = 1
                else:
                    self.num_rels = rel_mapping.shape[0]

                train_edges_tens = all_edges_list[0]
                if len(all_edges_list) == 2:
                    test_edges_tens = all_edges_list[1]
                elif len(all_edges_list) == 3:
                    valid_edges_tens = all_edges_list[1]
                    test_edges_tens = all_edges_list[2]

                train_edge_weights = all_edge_weights[0]
                valid_edge_weights = all_edge_weights[1]
                test_edge_weights = all_edge_weights[2]

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
                train_edges_tens, train_edge_weights = extract_tensor_from_tens(train_edges_tens, self.column_mappings)
                test_edges_tens, test_edge_weights = extract_tensor_from_tens(test_edges_tens, self.column_mappings)
                valid_edges_tens, valid_edge_weights = extract_tensor_from_tens(valid_edges_tens, self.column_mappings)

        # Split the edges
        if self.splits is not None:
            (
                train_edges_tens,
                train_edge_weights,
                valid_edges_tens,
                valid_edge_weights,
                test_edges_tens,
                test_edge_weights,
            ) = split_edges(train_edges_tens, train_edge_weights, self.splits)

        # Cast to the correct dtype
        def perform_cast(edge_tensor, weights_tensor, edge_dtype, weights_dtype):
            if edge_tensor is None:
                return edge_tensor, weights_tensor

            if weights_tensor is not None:
                weights_tensor = weights_tensor.to(weights_dtype)
            return edge_tensor.to(edge_dtype), weights_tensor

        train_edges_tens, train_edge_weights = perform_cast(
            train_edges_tens, train_edge_weights, self.dtype, self.weight_dtype
        )
        valid_edges_tens, valid_edge_weights = perform_cast(
            valid_edges_tens, valid_edge_weights, self.dtype, self.weight_dtype
        )
        test_edges_tens, test_edge_weights = perform_cast(
            test_edges_tens, test_edge_weights, self.dtype, self.weight_dtype
        )

        # Resolve all the null counts
        if self.num_nodes is None:
            combined_nodes = [train_edges_tens[:, [0, -1]]]
            if test_edges_tens is not None:
                combined_nodes.append(test_edges_tens[:, [0, -1]])
            if valid_edges_tens is not None:
                combined_nodes.append(valid_edges_tens[:, [0, -1]])

            combined_tensor = torch.unique(combined_nodes, sorted=False)
            self.num_nodes = torch.numel(combined_tensor)

        if self.num_rels is None:
            self.num_rels = 1

        all_edge_weights = [train_edge_weights, valid_edge_weights, test_edge_weights]
        if self.partitioner is not None:
            print("Partition nodes into {} partitions".format(self.num_partitions))
            (
                train_edges_tens,
                train_edges_offsets,
                valid_edges_tens,
                valid_edges_offsets,
                test_edges_tens,
                test_edges_offsets,
                all_edge_weights,
            ) = self.partitioner.partition_edges(
                train_edges_tens,
                valid_edges_tens,
                test_edges_tens,
                self.num_nodes,
                self.num_partitions,
                edge_weights=all_edge_weights,
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
                edge_weights=all_edge_weights,
            )
        else:
            return self.writer.write_to_binary(
                train_edges_tens,
                valid_edges_tens,
                test_edges_tens,
                self.num_nodes,
                self.num_rels,
                self.num_partitions,
                edge_weights=all_edge_weights,
            )
