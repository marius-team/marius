import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TESTING_DATA_DIR, TMP_TEST_DIR

import numpy as np
import pandas as pd
from omegaconf import MISSING, OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.configuration.marius_config import DatasetConfig
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter

import torch  # isort:skip

test_files = ["train_edges.txt", "train_edges_weights.txt", "valid_edges.txt", "test_edges.txt"]


def validate_partitioned_output_dir(
    output_dir: Path,
    expected_stats: DatasetConfig,
    num_partitions,
    dtype=np.int32,
    weight_dtype=np.float32,
    partitioned_eval=False,
    has_weights=False,
):
    print("Validate partioned called with value", has_weights)
    validate_output_dir(output_dir, expected_stats, dtype, remap_ids=True)

    train_edges_path = output_dir / Path(PathConstants.train_edges_path)
    train_edge_buckets_path = output_dir / Path(PathConstants.train_edge_buckets_path)

    assert train_edge_buckets_path.exists()
    with open(train_edge_buckets_path, "r") as f:
        train_buckets_sizes = f.readlines()
        assert len(train_buckets_sizes) == num_partitions**2

    train_edges = np.fromfile(train_edges_path, dtype).reshape(expected_stats.num_train, -1)

    offset = 0
    node_partition_size = np.ceil(expected_stats.num_nodes / num_partitions)
    for i in range(num_partitions):
        src_lower_bound = i * node_partition_size
        src_upper_bound = src_lower_bound + node_partition_size

        for j in range(num_partitions):
            bucket_size = int(train_buckets_sizes[i * num_partitions + j])

            if bucket_size != 0:
                edge_bucket = train_edges[offset : offset + bucket_size]

                dst_lower_bound = j * node_partition_size
                dst_upper_bound = dst_lower_bound + node_partition_size

                src_col = edge_bucket[:, 0]
                dst_col = edge_bucket[:, -1]

                assert np.all((src_col >= src_lower_bound) & (src_col < src_upper_bound))
                assert np.all((dst_col >= dst_lower_bound) & (dst_col < dst_upper_bound))

                offset += bucket_size

    assert offset == expected_stats.num_train

    print("Checking with has_weight of", has_weights)
    if has_weights:
        weights_file_path = output_dir / Path(PathConstants.train_edges_weights_path)
        assert weights_file_path.exists()
        values = np.fromfile(weights_file_path, dtype=weight_dtype)
        values = np.sort(values)
        for i in range(len(values)):
            assert values[i] == float(i)


def validate_output_dir(
    output_dir: Path,
    expected_stats: DatasetConfig,
    dtype=np.int32,
    remap_ids=True,
    has_weights=False,
    weight_dtype=np.float32,
):
    assert output_dir.exists()
    assert (output_dir / Path("edges")).exists()
    assert (output_dir / Path("nodes")).exists()

    dataset_stats = OmegaConf.load(output_dir / Path("dataset.yaml"))

    assert Path(dataset_stats.dataset_dir).absolute().__str__() == Path(expected_stats.dataset_dir).absolute().__str__()
    assert dataset_stats.num_edges == expected_stats.num_edges
    assert dataset_stats.num_relations == expected_stats.num_relations
    assert dataset_stats.num_nodes == expected_stats.num_nodes
    assert dataset_stats.num_train == expected_stats.num_train
    assert dataset_stats.get("num_valid", MISSING) == expected_stats.num_valid
    assert dataset_stats.get("num_test", MISSING) == expected_stats.num_test

    num_columns = 3
    if dataset_stats.num_relations == 1:
        num_columns = 2

    train_edges_path = output_dir / Path(PathConstants.train_edges_path)
    valid_edges_path = output_dir / Path(PathConstants.valid_edges_path)
    test_edges_path = output_dir / Path(PathConstants.test_edges_path)

    dtype_size = 4
    if dtype == np.int64:
        dtype_size = 8

    assert train_edges_path.exists()
    assert os.path.getsize(train_edges_path) == dataset_stats.num_train * num_columns * dtype_size
    train_edges = np.fromfile(train_edges_path, dtype)
    assert train_edges.reshape(dataset_stats.num_train, -1).shape[1] == num_columns

    if dataset_stats.get("num_valid", MISSING) != MISSING and dataset_stats.get("num_valid", MISSING) != -1:
        assert valid_edges_path.exists()
        valid_edges = np.fromfile(valid_edges_path, dtype)
        assert valid_edges.reshape(dataset_stats.num_valid, -1).shape[1] == num_columns
    else:
        assert not valid_edges_path.exists()

    if dataset_stats.get("num_test", MISSING) != MISSING and dataset_stats.get("num_test", MISSING) != -1:
        assert test_edges_path.exists()
        test_edges = np.fromfile(test_edges_path, dtype)
        assert test_edges.reshape(dataset_stats.num_test, -1).shape[1] == num_columns
    else:
        assert not test_edges_path.exists()

    node_mapping_path = output_dir / Path(PathConstants.node_mapping_path)
    relation_mapping_path = output_dir / Path(PathConstants.relation_mapping_path)
    if remap_ids:
        assert node_mapping_path.exists()
        node_mapping_df = pd.read_csv(node_mapping_path, sep=",", header=None)
        assert node_mapping_df.shape[0] == dataset_stats.num_nodes
        if num_columns == 3:
            assert relation_mapping_path.exists()
            relation_mapping_df = pd.read_csv(relation_mapping_path, sep=",", header=None)
            assert relation_mapping_df.shape[0] == dataset_stats.num_relations
        else:
            assert not relation_mapping_path.exists()
    else:
        assert not node_mapping_path.exists()
        assert not relation_mapping_path.exists()

    print("Checking with has_weight of", has_weights)
    if has_weights:
        weights_file_path = output_dir / Path(PathConstants.train_edges_weights_path)
        assert weights_file_path.exists()
        values = np.fromfile(weights_file_path, dtype=weight_dtype)
        for i in range(len(values)):
            assert values[i] == float(i)


class TestTorchConverter(unittest.TestCase):
    """
    Tests for the general preprocessor
    """

    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

        for test_file in test_files:
            shutil.copy(str(Path(TESTING_DATA_DIR) / Path(test_file)), str(Path(TMP_TEST_DIR) / Path(test_file)))
        pass

    @classmethod
    def tearDown(self):
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    def test_delimited_defaults(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_delim_default")
        output_dir.mkdir()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_delimited_str_ids(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_delim_str_ids")
        output_dir.mkdir()

        tmp = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")

        tmp[0] = tmp[0].map(str) + "_test"
        tmp[1] = tmp[1].map(str) + "_test"
        tmp[2] = tmp[2].map(str) + "_test"

        tmp.to_csv(Path(TMP_TEST_DIR) / Path("str_train_edges.txt"), header=None, sep=" ", index=False)

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("str_train_edges.txt"),
            delim=" ",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_numpy_defaults(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_numpy_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")

        train_edges = train_edges_df.to_numpy()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            format="numpy",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_pytorch_defaults(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")

        train_edges = torch.tensor(train_edges_df.to_numpy())

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            format="pytorch",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_splits(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_splits")
        output_dir.mkdir()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            splits=[0.9, 0.05, 0.05],
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 900
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 900
        expected_stats.num_valid = 50
        expected_stats.num_test = 50

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_columns(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_columns")
        output_dir.mkdir()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            src_column=0,
            dst_column=2,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_header(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_header")
        output_dir.mkdir()

        tmp = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        tmp.to_csv(
            Path(TMP_TEST_DIR) / Path("header_train_edges.txt"), header=["src", "rel", "dst"], sep=" ", index=False
        )

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("header_train_edges.txt"),
            delim=" ",
            header_length=1,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_delim(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_delim")
        output_dir.mkdir()

        tmp = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        tmp.to_csv(Path(TMP_TEST_DIR) / Path("delim_train_edges.txt"), header=None, sep=",", index=False)

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("delim_train_edges.txt"),
            delim=",",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_dtype(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_dtype")
        output_dir.mkdir()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            dtype="int64",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir,
            expected_stats=expected_stats,
            dtype=np.int64,
            weight_dtype=np.float64,
            remap_ids=True,
        )

    def test_partitions(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_partitions")
        output_dir.mkdir()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            num_partitions=10,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_partitioned_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, num_partitions=10
        )

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            num_partitions=100,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        validate_partitioned_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, num_partitions=100
        )

    def test_no_remap(self):
        output_dir = Path(TMP_TEST_DIR) / Path("test_dtype")
        output_dir.mkdir()

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            remap_ids=False,
            num_nodes=100,
            num_rels=10,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=False)

    def test_torch_no_relation_no_remap(self):
        remap_val = False
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))
        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            num_nodes=100,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val)

    def test_pandas_no_relation_no_remap(self):
        remap_val = False
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            num_nodes=100,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val)

    def test_torch_no_relation_remap(self):
        remap_val = True
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            num_nodes=100,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val)

    def test_pandas_no_relation_remap(self):
        remap_val = True
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            num_nodes=100,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val)

    def test_torch_only_weights_no_remap(self):
        remap_val = False
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_weight_column=3,
            num_nodes=100,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_pandas_only_weights_no_remap(self):
        remap_val = False
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_weight_column=3,
            num_nodes=100,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_torch_only_weights_remap(self):
        remap_val = True
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_weight_column=3,
            num_nodes=100,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_pandas_only_weights_remap(self):
        remap_val = True
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_weight_column=3,
            num_nodes=100,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 1
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_torch_relationship_weights_no_remap(self):
        remap_val = False
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            edge_weight_column=3,
            num_nodes=100,
            num_rels=10,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_pandas_relationship_weights_no_remap(self):
        remap_val = False
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            edge_weight_column=3,
            num_nodes=100,
            num_rels=10,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_torch_relationship_weights_remap(self):
        remap_val = True
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            edge_weight_column=3,
            num_nodes=100,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_pandas_relationship_weights_remap(self):
        remap_val = True
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            remap_ids=remap_val,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            edge_weight_column=3,
            num_nodes=100,
            num_rels=10,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=remap_val, has_weights=True
        )

    def test_torch_relationship_weights_remap_partioned(self):
        num_paritions = 10
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_df = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        train_edges = torch.tensor(train_edges_df.to_numpy())

        num_rows = train_edges.size(0)
        train_edges = torch.column_stack((train_edges, torch.arange(num_rows)))

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            edge_weight_column=3,
            num_partitions=num_paritions,
            format="pytorch",
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_partitioned_output_dir(
            output_dir=output_dir,
            expected_stats=expected_stats,
            dtype=np.int32,
            num_partitions=num_paritions,
            has_weights=True,
        )

    def test_pandas_relationship_weights_remap_partioned(self):
        num_paritions = 10
        output_dir = Path(TMP_TEST_DIR) / Path("test_torch_defaults")
        output_dir.mkdir()

        train_edges_file = Path(TMP_TEST_DIR) / Path("train_edges_weights.txt")

        converter = TorchEdgeListConverter(
            output_dir=output_dir,
            train_edges=train_edges_file,
            delim=" ",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            edge_weight_column=3,
            num_partitions=num_paritions,
        )
        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_partitioned_output_dir(
            output_dir=output_dir,
            expected_stats=expected_stats,
            dtype=np.int32,
            num_partitions=num_paritions,
            has_weights=True,
        )
