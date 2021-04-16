import unittest
import argparse
import subprocess
import shutil
from pathlib import Path
from marius.tools.config_generator import set_args
from marius.tools.config_generator import parse_args
from marius.tools.config_generator import read_template


class TestConfigGeneratorCmdOptParser(unittest.TestCase):
    """
    Tests for functions parsing command line arguments for config generator
    """
    cmd_args = [
        ["./output_dir", "-d", "wn18"],
        ["./output_dir", "-d"],
        ["./output_dir", "-d", "wn1267"],
        ["./output_dir", "-d", "wn18", "-s", "14", "6", "22", "12", "10"],
        ["./output_dir", "-d", "wn18", "-dev", "multi-GPU"],
        ["./output_dir", "-s", "14", "2", "14", "7", "5"],
        ["./output_dir", "-s", "23", "2"],
        ["-d", "live_journal"],
        [],
        ["marius_config_generator", "./output_dir",
         "-d", "wn18"],
        ["./output_dir", "-d", "live_journal",
         "--training.number_of_chunks=32"],
        ["./output_dir", "-d", "fb15k", "--reporting.log_level=all"],
        ["./output_dir", "-d", "wn18", "--data_directory", "./data_dir"],
        ["./output_dir"]
    ]

    def tearDown(self):
        if Path("./output_dir").exists():
            shutil.rmtree(Path("./output_dir"))

    def test_device_default(self):
        """
        Check if default values of -dev and other config opts
            are assigned correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[0])
        config_dict = parse_args(args)
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("model.embedding_size") == "128")
        self.assertTrue(config_dict.get("general.random_seed") is None)
        self.assertTrue(str(config_dict.get("num_train")) == "141442")
        self.assertTrue(str(config_dict.get("num_nodes")) == "40943")
        self.assertTrue(str(config_dict.get("num_relations")) == "18")
        self.assertTrue(str(config_dict.get("num_valid")) == "5000")
        self.assertTrue(str(config_dict.get("num_test")) == "5000")
        self.assertTrue(config_dict.get("data_directory") is None)
        self.assertTrue(config_dict.get("output_directory") == "./output_dir")

    def test_missing_dataset(self):
        """
        Check if exception is thrown when dataset name is missing
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[1])

    def test_unrecognized_dataset(self):
        """
        Check if exception is thrown with incorrect dataset name given
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[2])
        with self.assertRaises(RuntimeError):
            config_dict = parse_args(args)

    def test_exclusive_args(self):
        """
        Check if exception is thrown when dataset and stats are both specified
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[3])

    def test_multi_gpu_opt(self):
        """
        Check if multi_gpu can be parsed by -dev option
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[4])
        config_dict = parse_args(args)
        self.assertTrue(config_dict.get("general.device") == "multi-GPU")

    def test_stats_parsing(self):
        """
        Check if stats opt can be parsed correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[5])
        config_dict = parse_args(args)
        print(config_dict.get("num_nodes"))
        self.assertTrue(str(config_dict.get("num_nodes")) == "14")
        self.assertTrue(str(config_dict.get("num_relations")) == "2")
        self.assertTrue(str(config_dict.get("num_train")) == "14")
        self.assertTrue(str(config_dict.get("num_valid")) == "7")
        self.assertTrue(str(config_dict.get("num_test")) == "5")

    def test_incomplete_stats(self):
        """
        Check if exception is thrown when incomplete stats is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[6])

    def test_missing_arg(self):
        """
        Check if exception is thrown when missing required arg
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[7])

    def test_empty_arg(self):
        """
        Check if exception is thrown when no arg is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[8])

    def test_output(self):
        """
        Check if config file is generated into the correct directory
        """
        subprocess.run(self.cmd_args[9])
        self.assertTrue(Path("./output_dir/wn18_gpu.ini").exists())

    def test_config_opt_parsing(self):
        """
        Check if config opt can be parsed correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[10])
        config_dict = parse_args(args)
        print(config_dict.get("training.number_of_chunks"))
        self.assertTrue(config_dict.get("training.number_of_chunks") == "32")
        self.assertTrue(config_dict.get("dataset") == "live_journal")
        self.assertTrue(config_dict.get("model.embedding_size") == "128")

    def test_invalid_config_opt(self):
        """
        Check if exception is thrown with invalid config opt
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[11])

    def test_data_path(self):
        """
        Check if data path is set correctly if --data_directory is specified
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[12])
        config_dict = parse_args(args)
        self.assertTrue(str(config_dict.get("data_directory")) ==
                        "./data_dir")
        self.assertTrue(str(config_dict.get("output_directory")) ==
                        "./output_dir")

    def test_missing_mode_arg(self):
        """
        Check if exception is thrown when -d and -s are not specified
        """
        parser, config_dict = set_args()
        with self.assertRaises(RuntimeError):
            args = parser.parse_args(self.cmd_args[13])
            config_dict = parse_args(args)
