import unittest
from pathlib import Path
from tools.csv_converter import general_parser


class TestGeneralParser(unittest.TestCase):
    """
    Tests for the general preprocessor
    """

    def test_basic(self):
        """
        Check the preprocessor executes on the test data without error
        """

        output_dir = "test/test_data/"

        train_file = "train_edges.txt"
        valid_file = "valid_edges.txt"
        test_file = "test_edges.txt"

        stats = general_parser(
                               [str(Path(output_dir) / train_file),
                                str(Path(output_dir) / valid_file),
                                str(Path(output_dir) / test_file)],
                               ["srd"], [output_dir], num_partitions=1)
        assert(stats[2] == 1000)
        assert(stats[3] == 100)
        assert(stats[4] == 100)
        assert(stats[0] == 100)
        assert(stats[1] == 10)
