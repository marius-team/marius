import os
import shutil
import unittest
import subprocess
from pathlib import Path
import random
import numpy as np
from marius.tools.csv_converter import general_parser

TEST_DIR = "./output_dir"
test_data_dir = "./test/test_data/"
input_dir = "./test/test_data/test_data_copy/"
train_file = "train_edges.txt"
valid_file = "valid_edges.txt"
test_file = "test_edges.txt"
output_dir = TEST_DIR

class TestGeneralParser(unittest.TestCase):
    """
    Tests for the general preprocessor
    """
    @classmethod
    def setUp(self):
        if not Path(TEST_DIR).exists():
            Path(TEST_DIR).mkdir()

        if Path(input_dir).exists():
            shutil.rmtree(Path(input_dir))
        
        Path(input_dir).mkdir()
        shutil.copy(str(Path(test_data_dir) / Path(train_file)),
                    str(Path(input_dir) / Path(train_file)))
        shutil.copy(str(Path(test_data_dir) / Path(valid_file)),
                    str(Path(input_dir) / Path(valid_file)))
        shutil.copy(str(Path(test_data_dir) / Path(test_file)),
                    str(Path(input_dir) / Path(test_file)))


    @classmethod
    def tearDown(self):
        if Path(TEST_DIR).exists():
            shutil.rmtree(Path(TEST_DIR))
        
        if Path(input_dir).exists():
            shutil.rmtree(Path(input_dir))


    def dataset_generator(self, train_len=1000, valid_len=100, test_len=100,
            delim="\t", start_col=0, num_line_skip=0):
        with open(str(Path(input_dir) / Path(train_file)), "w") as f:
            for i in range(num_line_skip):
                f.write("This is a line needs to be skipped.\n")
            for i in range(train_len):
                src = random.randint(1, 100)
                dst = random.randint(1, 100)
                rel = random.randint(101, 110)
                for j in range(start_col):
                    f.write("col_" + str(j) + delim)
                f.write(str(src) + delim + str(rel) + delim + str(dst) + "\n")
        f.close()

        with open(str(Path(input_dir) / Path(valid_file)), "w") as f:
            for i in range(num_line_skip):
                f.write("This is a line needs to be skipped.\n")
            for i in range(valid_len):
                src = random.randint(1, 100)
                dst = random.randint(1, 100)
                rel = random.randint(101, 110)
                for j in range(start_col):
                    f.write("col_" + str(j) + delim)
                f.write(str(src) + delim + str(rel) + delim + str(dst) + "\n")
        f.close()

        with open(str(Path(input_dir) / Path(test_file)), "w") as f:
            for i in range(num_line_skip):
                f.write("This is a line needs to be skipped.\n")
            for i in range(test_len):
                src = random.randint(1, 100)
                dst = random.randint(1, 100)
                rel = random.randint(101, 110)
                for j in range(start_col):
                    f.write("col_" + str(j) + delim)
                f.write(str(src) + delim + str(rel) + delim + str(dst) + "\n")
        f.close()


    def test_basic(self):
        """
        Check the preprocessor executes on the test data without error
        """
        stats = general_parser(
            [str(Path(input_dir) / Path(train_file)),
             str(Path(input_dir) / Path(valid_file)),
             str(Path(input_dir) / Path(test_file))],
            ["srd"], [output_dir], num_partitions=1)

        assert (stats[2] == 1000) # num_train
        assert (stats[3] == 100) # num_valid
        assert (stats[4] == 100) # num_test
        assert (stats[0] == 100) # num_nodes
        assert (stats[1] == 10) # num_relations


    def test_num_files_multi_files(self):
        """
        Check if different numbers of input files can be processed
        Expected behavior: if #files > 3, all files will be merged into 1 file
            when dataset_split is not specified
        """
        stats = general_parser(
                    [str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(train_file))],
                     ["srd"], [output_dir])

        self.assertTrue(stats[0] == 100)
        self.assertTrue(stats[1] == 10)
        self.assertTrue(stats[2] == 3300)
    
    def test_num_files_zero_file(self):
        """
        Check if exception is thrown if 0 file is passed
        """
        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [],
                    ["srd"], [output_dir], dataset_split = (0.1, 0.1))
    
    def test_format_invalid_format(self):
        """
        Check if exception is thrown if format is specified incorrectly
        Expected format should be only composed of "s", "r", and "d" or "s" and
        "d"
        """
        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file))],
                    ["srdx"], [output_dir], dataset_split = (0.1, 0.1))
        
        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file))],
                    ["sr"], [output_dir], dataset_split = (0.1, 0.1))

        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file))],
                    ["srx"], [output_dir], dataset_split = (0.1, 0.1))

        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file))],
                    ["x"], [output_dir], dataset_split = (0.1, 0.1))

        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file))],
                    ["s"], [output_dir], dataset_split = (0.1, 0.1))


    def test_delim_basic(self):
        """
        Check if csv_converter can detect delimiter in basic case
        """
        proc = subprocess.run(
            ["python3", "./src/python/tools/csv_converter.py",
             str(Path(input_dir) / Path(train_file)),
             str(Path(input_dir) / Path(valid_file)),
             str(Path(input_dir) / Path(test_file)),
             "srd", output_dir
            ], capture_output=True)

        proc_output = str(proc.stdout)
        delim_idx = proc_output.find('~')
        delim = proc_output[delim_idx+1]
        self.assertEqual(delim, " ")


    def test_delim_complex(self):
        """
        Check if csv_converter can detect delimiter in more complex case
            Note: csv_converter uses python Sniffer to detect delimiter. Its
            capability of detecting delimiter is limited. This test ensures
            future changes will not have worse performance
        """
        self.dataset_generator(delim='\t', num_line_skip=2, start_col=2)

        proc = subprocess.run(
            ["python3", "./src/python/tools/csv_converter.py",
             str(Path(input_dir) / Path(train_file)),
             str(Path(input_dir) / Path(valid_file)),
             str(Path(input_dir) / Path(test_file)),
             "srd", output_dir, "--start_col=2", "--num_line_skip=2"
            ], capture_output=True)

        proc_output = str(proc.stdout)
        print(proc_output)
        delim_idx = proc_output.find('~')
        delim = proc_output[delim_idx+1: delim_idx+3]
        self.assertEqual(delim, '\\t')


    def test_num_line_skip(self):
        """
        Check if csv_converter can process data files correctly with 
            num_line_skip specified
        """
        self.dataset_generator(delim='\t', num_line_skip=5)
        proc = subprocess.run(
            ["python3", "./src/python/tools/csv_converter.py",
             str(Path(input_dir) / Path(train_file)),
             str(Path(input_dir) / Path(valid_file)),
             str(Path(input_dir) / Path(test_file)),
             "srd", output_dir, "--num_line_skip=5"
            ], capture_output=True)
        
        proc_output = str(proc.stdout)
        print(proc_output)
        delim_idx = proc_output.find('~')
        delim = proc_output[delim_idx+1: delim_idx+3]
        self.assertEqual(delim, '\\t')


    def test_num_line_skip_incorrect_value(self):
        """
        Check if exception is thrown when num_line_skip is not
            specified correctly
        """
        self.dataset_generator(delim='\t', num_line_skip=5)
        with self.assertRaises(AssertionError):
            general_parser([str(Path(input_dir) / Path(train_file)),
                            str(Path(input_dir) / Path(valid_file)),
                            str(Path(input_dir) / Path(test_file))],
                           ["srd"], [output_dir], num_line_skip=1)
   

    def test_start_col_exception(self):
        """
        Check if exception is thrown if start_col is specified but
            num_line_skip is not
            Note: csv_converter currently requires num_line_skip to be
            specified if start_col is specified. If future improvment is made,
            this test can be removed
        """
        self.dataset_generator(delim='\t', start_col=2)
        with self.assertRaises(AssertionError):
            general_parser([str(Path(input_dir) / Path(train_file)),
                            str(Path(input_dir) / Path(valid_file)),
                            str(Path(input_dir) / Path(test_file))],
                           ["srd"], [output_dir], start_col=2)


    def test_dataset_split_single_file(self):
        """
        Check if dataset_split works properly
        Expected behavior: if dataset_split is set to non-zero tuple and number
            of files is 1, the data file will be redistributed into train, 
            valid, and test sets according to the fraction specified by the 
            dataset_split tuple
        """
        stats = general_parser(
                    [str(Path(input_dir) / Path(train_file))],
                    ["srd"], [output_dir], dataset_split = (0.1, 0.1))
        
        self.assertTrue(stats[0] == 100)
        self.assertTrue(stats[1] == 10)
        self.assertTrue(stats[2] == 800)
        self.assertTrue(stats[3] == 100)
        self.assertTrue(stats[4] == 100)


    def test_dataset_split_multi_files(self):
        """
        Check if dataset_split works properly
        Expected behavior: if dataset_split is set to non-zero tuple and number
            of files is greater than 3, the data files will be merged into 1 
            file and then redistributed into train, valid, and test sets
            accroding to the fraction specified by the dataset_split tuple
        """
        stats = general_parser(
                    [str(Path(input_dir) / train_file),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(test_file)),
                     str(Path(input_dir) / Path(valid_file)),
                     str(Path(input_dir) / Path(train_file)),
                     str(Path(input_dir) / Path(train_file))],
                     ["srd"], [output_dir], dataset_split = (0.1, 0.1)) 
        
        self.assertTrue(stats[0] == 100)
        self.assertTrue(stats[1] == 10)
        self.assertTrue(stats[2] == 2640)
        self.assertTrue(stats[3] == 330)
        self.assertTrue(stats[4] == 330)


    def test_dataset_split_invalid_fraction(self):
        """
        Check if dataset_split throws exception with invalid fraction
        """
        with self.assertRaises(AssertionError):
            stats = general_parser(
                    [str(Path(input_dir) / Path(train_file))],
                    ["srd"], [output_dir], dataset_split = (0.6, 0.7))


    def test_remap_ids_false(self):
        """
        Check if processed data has sequential ids if remap_ids is set to False
        """
        general_parser([str(Path(input_dir) / Path(train_file)),
                        str(Path(input_dir) / Path(valid_file)),
                        str(Path(input_dir) / Path(test_file))],
                       ["srd"], [output_dir], remap_ids=False)

        internal_node_ids = np.fromfile(str(Path(output_dir)) /
                                        Path("node_mapping.bin"), dtype=int)
        internal_rel_ids = np.fromfile(str(Path(output_dir)) /
                                       Path("rel_mapping.bin"), dtype=int)
        
        for i in range(len(internal_node_ids) - 1):
            self.assertEqual((internal_node_ids[i+1] - internal_node_ids[i]), 1)
        
        for i in range(len(internal_rel_ids) - 1):
            self.assertEqual((internal_rel_ids[i+1] - internal_rel_ids[i]), 1)
        
    
    def test_remap_ids_true(self):
        """
        Check if processed data has non-sequential ids if remap_ids is set 
            to True
        """
        general_parser([str(Path(input_dir) / Path(train_file)),
                        str(Path(input_dir) / Path(valid_file)),
                        str(Path(input_dir) / Path(test_file))],
                       ["srd"], [output_dir], remap_ids=True)

        internal_node_ids = np.fromfile(str(Path(output_dir)) /
                                        Path("node_mapping.bin"), dtype=int)
        internal_rel_ids = np.fromfile(str(Path(output_dir)) /
                                       Path("rel_mapping.bin"), dtype=int)
        
        delta_sum = 0
        for i in range(len(internal_node_ids) - 1):
            delta_sum = (internal_node_ids[i+1] - internal_node_ids[i])\
                                                                    + delta_sum
            self.assertTrue(delta_sum < (len(internal_node_ids) - 1))
        
        delta_sum = 0
        for i in range(len(internal_rel_ids) - 1):
            delta_sum = (internal_rel_ids[i+1] - internal_rel_ids[i])\
                                                                + delta_sum
            self.assertTrue(delta_sum < (len(internal_rel_ids) - 1))


    def test_num_partitions(self):
        """
        Check if correct number of partitions are made as num_partitions is
            specified
        """
        general_parser([str(Path(input_dir) / Path(train_file)),
                        str(Path(input_dir) / Path(valid_file)),
                        str(Path(input_dir) / Path(test_file))],
                       ["srd"], [output_dir], num_partitions=3)
        
        with open(str(Path(output_dir) /
                      Path("train_edges_partitions.txt")), 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 9)
    

    def test_num_partitions_invalid(self):
        """
        Check if exception is thrown if invalid number of partitions is set
        """
        with self.assertRaises(AssertionError):
            general_parser([str(Path(input_dir) / Path(train_file)),
                            str(Path(input_dir) / Path(valid_file)),
                            str(Path(input_dir) / Path(test_file))],
                           ["srd"], [output_dir], num_partitions=-1)
        
        with self.assertRaises(AssertionError):
            general_parser([str(Path(input_dir) / Path(train_file)),
                            str(Path(input_dir) / Path(valid_file)),
                            str(Path(input_dir) / Path(test_file))],
                           ["srd"], [output_dir], num_partitions=0)
        

    def test_dtype_int32(self):
        """
        Check if processed data is stored in int32 as specified
        """
        stats = general_parser([str(Path(input_dir) / Path(train_file)),
                            str(Path(input_dir) / Path(valid_file)),
                            str(Path(input_dir) / Path(test_file))],
                           ["srd"], [output_dir], dtype=np.int32)
        
        self.assertEqual((Path(output_dir) /
                          Path("train_edges.pt")).stat().st_size, stats[2]*3*4)
        self.assertEqual((Path(output_dir) /
                          Path("valid_edges.pt")).stat().st_size, stats[3]*3*4)
        self.assertEqual((Path(output_dir) /
                          Path("test_edges.pt")).stat().st_size, stats[4]*3*4)


    def test_dtype_int64(self):
        """
        Check if processed data is stored in int64 as specified
        """
        stats = general_parser([str(Path(input_dir) / Path(train_file)),
                            str(Path(input_dir) / Path(valid_file)),
                            str(Path(input_dir) / Path(test_file))],
                           ["srd"], [output_dir], dtype=np.int64)
        
        self.assertEqual((Path(output_dir) /
                          Path("train_edges.pt")).stat().st_size, stats[2]*3*8)
        self.assertEqual((Path(output_dir) /
                          Path("valid_edges.pt")).stat().st_size, stats[3]*3*8)
        self.assertEqual((Path(output_dir) /
                          Path("test_edges.pt")).stat().st_size, stats[4]*3*8)

    def test_dtype_cmd_opt(self):
        """
        Check if dtype opt works from command line arguments
        """
        proc = subprocess.run(
            ["python3", "./src/python/tools/csv_converter.py",
             str(Path(input_dir) / Path(train_file)),
             str(Path(input_dir) / Path(valid_file)),
             str(Path(input_dir) / Path(test_file)),
             "srd", output_dir, "--dtype=int32"
            ], capture_output=True)

        self.assertEqual((Path(output_dir) /
                          Path("train_edges.pt")).stat().st_size, 1000*3*4)
        self.assertEqual((Path(output_dir) /
                          Path("valid_edges.pt")).stat().st_size, 100*3*4)
        self.assertEqual((Path(output_dir) /
                          Path("test_edges.pt")).stat().st_size, 100*3*4)