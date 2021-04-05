import unittest
import argparse
from tools.preprocess import setArgs
from tools.preprocess import parseArgs

class TestCmdOptParser(unittest.TestCase):
    """
    Tests for functions parsing command line arguments
    """


    def test_parseArgs(self):
        """
        Check if command line arguments are parsed correctly
        """
        parser, opts, cpu_dict, gpu_dict, mgpu_dict = setArgs()
        devices = ["CPU", "GPU", "Multi_GPU"]
        dicts = [cpu_dict, gpu_dict, mgpu_dict]
        cmd_args = [
            ["wn18", "./output_dir", "--generate_config", "--num_partitions", "5",
            "-cfd", "./config_output_dir"],
            ["wn18", "./output_dir", "-gc", "Multi_GPU"],
            ["wn18", "./output_dir", "-gc", "Multi_GPU", "--general.embedding_size=400"],
            ["wn18", "./output_dir", "-gc", "CPU", "--model.embedding_size=400",
            "--training.batch_size=51200", "--training.num_epochs=23"],            
            ["wn18", "./output", "--general.embedding_size=200"],
            ["wn18", "./output", "-cfd", "./dir"],
            ["wn18", "./output", "-gc", "CPU", "--general.embedding_size=200"],
            ["wn18", "./output", "CPU"],
            ["wn18"],
            []
        ]

        args = []
        for i in range(len(cmd_args)):
            if i > len(cmd_args) - 4: continue
            a = cmd_args[i]
            arg = parser.parse_args(a)
            args.append(arg)
        
        for i in range(len(cmd_args)):
            devices_idx = -1
            if i < 4:
                devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
                arg_dict = vars(args[i])
            
            if i == 0:
                self.assertTrue(arg_dict.get("dataset") == "wn18")
                self.assertTrue(devices_idx == 1)
                self.assertTrue(args[i].num_partitions  == 5)
                self.assertTrue(arg_dict.get("config_dir") == "./config_output_dir")
            elif i == 1:
                self.assertTrue(devices_idx == 2)
            elif i == 2:
                self.assertTrue(devices_idx == 2)
                self.assertTrue(arg_dict.get("general.embedding_size") == "400")
            elif i == 3:
                self.assertTrue(devices_idx == 0)
                self.assertTrue(arg_dict.get("model.embedding_size") == "400")
                self.assertTrue(arg_dict.get("training.batch_size") == "51200")
                self.assertTrue(arg_dict.get("training.num_epochs") == "23")
            elif i == 4:
                with self.assertRaises(RuntimeError):
                    devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
            elif i == 5:
                with self.assertRaises(AssertionError):
                    devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
            elif i == 6:
                with self.assertRaises(RuntimeError):
                    devices_idx, dicts, arg_dict = parseArgs(devices, dicts, args[i], opts)
            elif i == 7:
                with self.assertRaises(SystemExit):
                    arg = parser.parse_args(cmd_args[i])
            elif i == 8:
                with self.assertRaises(SystemExit):
                    arg = parser.parse_args(cmd_args[i])
            elif i == 9:
                with self.assertRaises(SystemExit):
                    arg = parser.parse_args(cmd_args[i])