import unittest
from pathlib import Path
import shutil
import os
from marius.tools.csv_converter import general_parser


TEST_DIR = "./output_dir"

class TestGeneralParser(unittest.TestCase):
    """
    Tests for the general preprocessor
    """

    @classmethod
    def tearDown(self):
        if Path(TEST_DIR).exists():
            shutil.rmtree(Path(TEST_DIR))

    def test_basic(self):
        """
        Check the preprocessor executes on the test data without error
        """

        input_dir = "test/test_data/"
        train_file = "train_edges.txt"
        valid_file = "valid_edges.txt"
        test_file = "test_edges.txt"

        output_dir = TEST_DIR

        os.makedirs(output_dir)

        stats, num_nodes, num_rels = general_parser(
            [str(Path(input_dir) / train_file),
             str(Path(input_dir) / valid_file),
             str(Path(input_dir) / test_file)],
            ["srd"], [output_dir], num_partitions=1)

        assert(stats[2] == 1000)
        assert(stats[3] == 100)
        assert(stats[4] == 100)
        assert(stats[0] == 100)
        assert(stats[1] == 10)
