import unittest
import argparse
import subprocess
import shutil
from pathlib import Path
from marius.tools.config_generator import output_config
from marius.tools.preprocess import set_args
from marius.tools.preprocess import parse_args


class TestPreprocessCmdOptParser(unittest.TestCase):
    """
    Tests for functions parsing command line arguments
    """
    cmd_args = [
            ["--output_directory", "./output_dir", "--dataset", "wn18",
             "--generate_config", "--num_partitions", "5"],
            ["--output_directory", "./output_dir", "--dataset", "wn18", "-gc",
             "GPU"],
            ["--output_directory", "./output_dir", "--dataset", "wn18", "-gc",
             "CPU",
             "--model.embedding_size=400", "--training.batch_size=51200",
             "--training.num_epochs=23"],
            ["--output_directory", "./output_dir", "--dataset", "wn18", "-gc",
             "GPU", "--general.embedding_size=400"],
            ["--output_directory", "./output_dir", "--dataset", "wn18",
             "--general.embedding_size=200"],
            ["--dataset", "wn18", "--output_directory", "./output_dir"],  # 5
            ["--dataset", "wn18"],
            ["--output_directory", "./output_dir", "--dataset", "wn18", "CPU"],
            ["--output_directory", "./output_dir", "--dataset", "wn18", "--gc",
             "--model.decoder"],
            [],
            ["--output_directory", "./output_dir", "--dataset", "wn18",
             "multi_cpu"], # 10
            ["--output_directory", "./output_dir", "--dataset", "wn18", "--gc",
             "--storage.edge_bucket_ordering=EliminationCus"],
            ["marius_preprocess", "--output_directory", "./output_dir",
             "--dataset", "wn18", "-gc"],
            ["--files",
             "./test/test_data/train_edges.txt",
             "./test/test_data/valid_edges.txt",
             "./test/test_data/test_edges.txt"] # 13
        ]

    dataset_dirs = []

    @classmethod
    def tearDown(self):
        for dir in self.dataset_dirs:
            if Path(dir).exists():
                shutil.rmtree(Path(dir))

    def test_generate_config_default(self):
        """
        Check if default value of --generate_config is assigned correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[0])
        config_dict, arg_dict, output_dir = parse_args(config_dict, args)
        self.dataset_dirs.append(output_dir)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(arg_dict.get("num_partitions") == 5)

    def test_gpu(self):
        """
        Check if --gc can parse device choice correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[1])
        config_dict, arg_dict, output_dir = parse_args(config_dict, args)
        self.dataset_dirs.append(output_dir)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(arg_dict.get("output_directory") == "./output_dir")

    def test_cpu_training_config(self):
        """
        Check if training configs can be parsed correctly
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[2])
        config_dict, arg_dict, output_dir = parse_args(config_dict, args)
        self.dataset_dirs.append(output_dir)
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
            --generate_config being specified
        """
        parser, config_dict = set_args()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[4])
            config_dict, arg_dict, output_dir = parse_args(config_dict, args)
            self.dataset_dirs.append(output_dir)

    def test_required_args(self):
        """
        Check if args.generate_config is set correctly if --generate_config
            is not specified
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[5])
        config_dict, arg_dict, output_dir = parse_args(config_dict, args)
        self.dataset_dirs.append(output_dir)
        self.assertTrue(arg_dict.get("generate_config") is None)

    def test_output_directory_not_specified_build_in_dataset(self):
        """
        Check if exception is thrown when output_directory is not given and 
            if output_directory is set to correct value when built-in dataset
            is specified
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[6])
        config_dict, arg_dict, output_dir = parse_args(config_dict, args)
        self.dataset_dirs.append(output_dir)
        self.assertTrue("wn18_dataset" in str(output_dir))

    def test_output_directory_not_spcified_custom_dataset(self):
        """
        Check if exception is thrown when output_directory is not given and
            if output_directory is set to correct value when preprocessing
            custom datasets
        """
        parser, config_dict = set_args()
        args = parser.parse_args(self.cmd_args[13])
        config_dict, arg_dict, output_dir = parse_args(config_dict, args)
        self.dataset_dirs.append(output_dir)
        self.assertTrue("custom_dataset" in str(output_dir))

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
        with self.assertRaises(RuntimeError):
            args = parser.parse_args(self.cmd_args[9])
            config_dict, arg_dict, output_dir = parse_args(config_dict, args)
            self.dataset_dirs.append(output_dir)

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
        proc = subprocess.run(self.cmd_args[12], capture_output=True)
        
        proc_output = proc.stdout.decode('utf-8')
        output_dir = proc_output.split('\n')[0].split(' ')[-1]
        self.dataset_dirs.append(output_dir)
        self.assertTrue((Path(output_dir) / Path("wn18_gpu.ini")).exists())

    def test_custom_dataset(self):
        """
        Check if custom dataset is processed correctly
        """
        proc = subprocess.run(["python3", "./src/python/tools/preprocess.py",
                        "--output_directory",
                        "./output_dir",
                        "--files",
                        "./test/test_data/train_edges.txt",
                        "./test/test_data/valid_edges.txt",
                        "./test/test_data/test_edges.txt",
                        "-gc", "CPU"], capture_output=True)

        proc_output = proc.stdout.decode('utf-8')
        output_dir = proc_output.split('\n')[0].split(' ')[-1]
        self.dataset_dirs.append(output_dir)
        self.assertTrue((Path(output_dir) / Path("custom_cpu.ini")).exists())