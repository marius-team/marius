import shutil
import unittest
from pathlib import Path
from test.python.constants import TESTING_DATA_DIR, TMP_TEST_DIR
from test.python.preprocessing.test_torch_converter import validate_output_dir, validate_partitioned_output_dir

import numpy as np
import pandas as pd
import pytest

test_files = ["train_edges.txt", "valid_edges.txt", "test_edges.txt"]

try:
    from marius.tools.configuration.marius_config import DatasetConfig
    from marius.tools.preprocess.converters.spark_converter import SparkEdgeListConverter

    pyspark_imported = True
except ImportError:
    pyspark_imported = False


@pytest.mark.skipif(not pyspark_imported, reason="Pyspark must be installed to run these tests.")
class TestSparkConverter(unittest.TestCase):
    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

        for test_file in test_files:
            shutil.copy(str(Path(TESTING_DATA_DIR) / Path(test_file)), str(Path(TMP_TEST_DIR) / Path(test_file)))
        pass

    @classmethod
    def tearDown(self):
        pass
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    def make_directory_tree(self, dir_path):
        output_dir = Path(TMP_TEST_DIR) / Path(dir_path)
        output_dir.mkdir()
        nodes_out_dir = output_dir / Path("nodes")
        edges_out_dir = output_dir / Path("edges")
        nodes_out_dir.mkdir()
        edges_out_dir.mkdir()
        return output_dir

    def test_delimited_defaults(self):
        output_dir = self.make_directory_tree("test_delim_default")

        converter = SparkEdgeListConverter(
            output_dir=output_dir, train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"), delim=" "
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
        output_dir = self.make_directory_tree("test_delimited_str_ids")

        tmp = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")

        tmp[0] = tmp[0].map(str) + "_test"
        tmp[1] = tmp[1].map(str) + "_test"
        tmp[2] = tmp[2].map(str) + "_test"

        tmp.to_csv(Path(TMP_TEST_DIR) / Path("str_train_edges.txt"), header=None, sep=" ", index=False)

        converter = SparkEdgeListConverter(
            output_dir=output_dir, train_edges=Path(TMP_TEST_DIR) / Path("str_train_edges.txt"), delim=" "
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    # randomSplit doesn't split the df in the exact ratio, outputs a close one though. skippping this one.

    def test_columns(self):
        output_dir = self.make_directory_tree("test_columns")

        converter = SparkEdgeListConverter(
            output_dir=output_dir, train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"), delim=" ", columns=[0, 2]
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
        output_dir = self.make_directory_tree("test_header")

        tmp = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        tmp.to_csv(
            Path(TMP_TEST_DIR) / Path("header_train_edges.txt"), header=["src", "rel", "dst"], sep=" ", index=False
        )

        converter = SparkEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("header_train_edges.txt"),
            delim=" ",
            header_length=1,
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
        output_dir = self.make_directory_tree("test_delim")

        tmp = pd.read_csv(Path(TMP_TEST_DIR) / Path("train_edges.txt"), header=None, sep=" ")
        tmp.to_csv(Path(TMP_TEST_DIR) / Path("delim_train_edges.txt"), header=None, sep=",", index=False)

        converter = SparkEdgeListConverter(
            output_dir=output_dir, train_edges=Path(TMP_TEST_DIR) / Path("delim_train_edges.txt"), delim=","
        )

        converter.convert()

        expected_stats = DatasetConfig()
        expected_stats.dataset_dir = output_dir.__str__()
        expected_stats.num_edges = 1000
        expected_stats.num_nodes = 100
        expected_stats.num_relations = 10
        expected_stats.num_train = 1000

        validate_output_dir(output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, remap_ids=True)

    def test_partitions(self):
        output_dir = self.make_directory_tree("test_partitions")

        converter = SparkEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            num_partitions=10,
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

        converter = SparkEdgeListConverter(
            output_dir=output_dir,
            train_edges=Path(TMP_TEST_DIR) / Path("train_edges.txt"),
            delim=" ",
            num_partitions=100,
        )

        converter.convert()

        validate_partitioned_output_dir(
            output_dir=output_dir, expected_stats=expected_stats, dtype=np.int32, num_partitions=100
        )
