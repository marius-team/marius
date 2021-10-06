import unittest
import argparse
import subprocess
import shutil
from pathlib import Path
from marius.tools.preprocess import set_args
from marius.tools.preprocess import parse_args


class TestPreprocessCmdOptParser(unittest.TestCase):
    """
    Tests for functions parsing command line arguments
    """
    cmd_args = [
            ["./output_dir", "--dataset", "wn18", "--generate_template_config",
             "--num_partitions", "5"],
            ["./output_dir", "--dataset", "wn18", "-gc", "GPU"],
            ["./output_dir", "--dataset", "wn18", "-gc", "CPU",
             "--model.embedding_size=400", "--training.batch_size=51200",
             "--training.num_epochs=23"],
            ["./output_dir", "--dataset", "wn18", "-gc", "GPU",
             "--general.embedding_size=400"],
            ["./output_dir", "--dataset", "wn18", "--general.embedding_size=200"],
            ["--dataset", "wn18", "./output_dir"],
            ["--dataset", "wn18"],
            ["./output_dir", "--dataset", "wn18", "CPU"],
            ["./output_dir", "--dataset", "wn18", "--gc", "--model.decoder"],
            [],
            ["./output_dir", "--dataset", "wn18", "multi_cpu"],
            ["./output_dir", "--dataset", "wn18", "--gc",
             "--storage.edge_bucket_ordering=EliminationCus"],
            ["marius_preprocess", "./output_dir", "--dataset", "wn18", "-gc"]
        ]

    @classmethod
    def setUp(self):
        if not Path("./output_dir").exists():
            Path("./output_dir").mkdir()

        if Path("./download_dir").exists():
            shutil.rmtree(Path("./download_dir"))

    @classmethod
    def tearDown(self):
        if Path("./output_dir").exists():
            shutil.rmtree(Path("./output_dir"))

    def test_generate_template_config_default(self):
        """
        Check if default value of --generate_template_config is assigned correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[0])
        config_dict, arg_dict = parse_args(config_dict, args)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(arg_dict.get("num_partitions") == 5)

    def test_gpu(self):
        """
        Check if --gc can parse device choice correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[1])
        config_dict, arg_dict = parse_args(config_dict, args)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(arg_dict.get("output_directory") == "./output_dir")

    def test_cpu_training_config(self):
        """
        Check if training configs can be parsed correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[2])
        config_dict, arg_dict = parse_args(config_dict, args)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "CPU")
        self.assertTrue(arg_dict.get("output_directory") == "./output_dir")
        self.assertTrue(config_dict.get("model.embedding_size") == "400")
        self.assertTrue(config_dict.get("training.batch_size") == "51200")
        self.assertTrue(config_dict.get("training.num_epochs") == "23")

    def test_unmatching_training_config(self):
        """
        Check if exception is thrown when config with unmatching
            section is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[3])

    def test_inconsistent_training_config(self):
        """
        Check if excpetion is thrown if trainig config is specified without
            --generate_template_config being specified
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[4])
            config_dict, arg_dict = parse_args(config_dict, args)

    def test_required_args(self):
        """
        Check if args.generate_template_config is set correctly if --generate_template_config
            is not specified
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[5])
        config_dict, arg_dict = parse_args(config_dict, args)
        self.assertTrue(arg_dict.get("generate_template_config") is None)

    def test_required_arg_omitted(self):
        """
        Check if exception is thrown when output_directory is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[6])

    def test_training_config_missing(self):
        """
        Check if exception is thrown when config name is missing and
            value is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[7])

    def test_training_config_value_missing(self):
        """
        Check if exception is thrown when config value is missing
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[8])

    def test_missing_arg(self):
        """
        Check if exception is thrown when no arg is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[9])

    def test_invalid_arg(self):
        """
        Check if exception is thrown when invalid arg value is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[10])

    def test_invalid_training_config_value(self):
        """
        Check if exception is thrown if invalid config value is given
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[11])

    def test_output(self):
        """
        Check if config file is generated into the correct directory
        """
        subprocess.run(self.cmd_args[12])
        self.assertTrue(Path("./output_dir/wn18_gpu.ini").exists())

    def test_custom_dataset(self):
        """
        Check if custom dataset is processed correctly
        """
        subprocess.run(["python3", "./src/python/tools/preprocess.py",
                        "./output_dir",
                        "--files",
                        "./test/test_data/train_edges.txt",
                        "./test/test_data/valid_edges.txt",
                        "./test/test_data/test_edges.txt",
                        "-gc", "CPU"])
        self.assertTrue(Path("./output_dir/custom_cpu.ini").exists())