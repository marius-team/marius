import unittest
import argparse
from pathlib import Path
import shutil
from tools.preprocess import setArgs
from tools.preprocess import parseArgs

class TestCmdOptParser(unittest.TestCase):
    """
    Tests for functions parsing command line arguments
    """
    cmd_args = [
            ["wn18", "./output_dir", "--generate_config", "--num_partitions", "5"],
            ["wn18", "./output_dir", "-gc", "GPU"],
            ["wn18", "./output_dir", "-gc", "CPU", "--model.embedding_size=400",
                "--training.batch_size=51200", "--training.num_epochs=23"],
            ["wn18", "./output_dir", "-gc", "GPU", "--general.embedding_size=400"],      
            ["wn18", "./output_dir", "--general.embedding_size=200"],
            ["wn18", "./output_dir"],
            ["wn18"],
            ["wn18", "./output_dir", "CPU"],
            ["wn18", "./output_dir", "--gc", "--model.decoder"],
            [],
            ["wn18", "./output_dir", "multi_cpu"],
            ["wn18", "./output_dir", "--gc", "--storage.edge_bucket_ordering=EliminationCus"]
        ]

    def tearDown(self):
        if Path("./output_dir").exists():
            shutil.rmtree(Path("./output_dir"))

    def test_generate_config_default(self):
        """
        Check if default value of --generate_config is assigned correctly
        """
        parser, config_dict = setArgs()
        args = parser.parse_args(self.cmd_args[0])
        config_dict, arg_dict = parseArgs(config_dict, args)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(arg_dict.get("num_partitions") == 5)

    def test_gpu(self):
        """
        Check if --gc can parse device choice correctly
        """
        parser, config_dict = setArgs()
        args = parser.parse_args(self.cmd_args[1])
        config_dict, arg_dict = parseArgs(config_dict, args)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "GPU")
        self.assertTrue(arg_dict.get("output_directory") == "./output_dir")

    def test_cpu_training_config(self):
        """
        Check if training configs can be parsed correctly
        """
        parser, config_dict = setArgs()
        args = parser.parse_args(self.cmd_args[2])
        config_dict, arg_dict = parseArgs(config_dict, args)
        self.assertTrue(config_dict.get("dataset") == "wn18")
        self.assertTrue(config_dict.get("device") == "CPU")
        self.assertTrue(arg_dict.get("output_directory") == "./output_dir")
        self.assertTrue(config_dict.get("model.embedding_size") == "400")
        self.assertTrue(config_dict.get("training.batch_size") == "51200")
        self.assertTrue(config_dict.get("training.num_epochs") == "23")

    def test_unmatching_training_config(self):
        """
        Check if exception is thrown when config with unmatching section is given
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[3])
    
    def test_inconsistent_training_config(self):
        """
        Check if excpetion is thrown if trainig config is specified without 
            --generate_config being specified
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[4])
            config_dict, arg_dict = parseArgs(config_dict, args)

    def test_required_args(self):
        """
        Check if args.generate_config is set correctly if --generate_config 
            is not specified
        """
        parser, config_dict = setArgs()
        args = parser.parse_args(self.cmd_args[5])
        config_dict, arg_dict = parseArgs(config_dict, args)
        self.assertTrue(arg_dict.get("generate_config") == None)
    
    def test_required_arg_omitted(self):
        """
        Check if exception is thrown when output_directory is given
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[6])
        
    def test_training_config_missing(self):
        """
        Check if exception is thrown when config name is missing and 
            value is given
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[7])

    def test_training_config_value_missing(self):
        """
        Check if exception is thrown when config value is missing
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[8])

    def test_missing_arg(self):
        """
        Check if exception is thrown when no arg is given
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[9])

    def test_invalid_arg(self):
        """
        Check if exception is thrown when invalid arg value is given
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[10])
    
    def test_invalid_training_config_value(self):
        """
        Check if exception is thrown if invalid config value is given
        """
        parser, config_dict = setArgs()
        with self.assertRaises(SystemExit):
            args = parser.parse_args(self.cmd_args[11])


    # def test_parseArgs(self):
    #     """
    #     Check if command line arguments are parsed correctly
    #     """
        

    #     args = []
    #     for i in range(len(cmd_args)):
    #         if i > len(cmd_args) - 4: continue
    #         a = cmd_args[i]
    #         arg = parser.parse_args(a)
    #         args.append(arg)
        
    #     for i in range(len(cmd_args)):
    #         devices_idx = -1
    #         if i < 4:
    #             devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
    #             arg_dict = vars(args[i])
            
    #         if i == 0:
    #             self.assertTrue(arg_dict.get("dataset") == "wn18")
    #             self.assertTrue(devices_idx == 1)
    #             self.assertTrue(args[i].num_partitions  == 5)
    #             self.assertTrue(arg_dict.get("config_dir") == "./config_output_dir")
    #         elif i == 1:
    #             self.assertTrue(devices_idx == 2)
    #         elif i == 2:
    #             self.assertTrue(devices_idx == 2)
    #             self.assertTrue(arg_dict.get("general.embedding_size") == "400")
    #         elif i == 3:
    #             self.assertTrue(devices_idx == 0)
    #             self.assertTrue(arg_dict.get("model.embedding_size") == "400")
    #             self.assertTrue(arg_dict.get("training.batch_size") == "51200")
    #             self.assertTrue(arg_dict.get("training.num_epochs") == "23")
    #         elif i == 4:
    #             with self.assertRaises(RuntimeError):
    #                 devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
    #         elif i == 5:
    #             with self.assertRaises(AssertionError):
    #                 devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
    #         elif i == 6:
    #             with self.assertRaises(RuntimeError):
    #                 devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
    #         elif i == 7:
    #             with self.assertRaises(SystemExit):
    #                 arg = parser.parse_args(cmd_args[i])
    #         elif i == 8:
    #             with self.assertRaises(SystemExit):
    #                 arg = parser.parse_args(cmd_args[i])
    #         elif i == 9:
    #             with self.assertRaises(SystemExit):
    #                 arg = parser.parse_args(cmd_args[i])
    #