import os
import shutil
import unittest
from pathlib import Path

from marius.tools.csv_converter import general_parser

TEST_DIR = "./output_dir"
input_dir = "test/test_data/"
train_file = "train_edges.txt"
valid_file = "valid_edges.txt"
test_file = "test_edges.txt"

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
        output_dir = TEST_DIR

        os.makedirs(output_dir)

        stats = general_parser(
            [str(Path(input_dir) / train_file),
             str(Path(input_dir) / valid_file),
             str(Path(input_dir) / test_file)],
            ["srd"], [output_dir], num_partitions=1)

        assert (stats[2] == 1000) # num_train
        assert (stats[3] == 100) # num_valid
        assert (stats[4] == 100) # num_test
        assert (stats[0] == 100) # num_nodes
        assert (stats[1] == 10) # num_relations

    def test_num_files(self):
        """
        Check if different numbers of input files can be processed
        Expected behavior: if #files > 3, all files will be merged into 1 file
            when dataset_split is not specified
        """
        output_dir = TEST_DIR

        stats = general_parser(
                    [str(Path(input_dir) / train_file), # 1000
                     str(Path(input_dir) / valid_file), # 100
                     str(Path(input_dir) / test_file), # 100
                     str(Path(input_dir) / valid_file), # 100
                     str(Path(input_dir) / train_file), # 1000
                     str(Path(input_dir) / train_file)], # 1000
                     ["srd"], [output_dir])
        
        self.assertTrue(len(stats) == 3)
        self.assertTrue(stats[0] == 100) # num_nodes
        self.assertTrue(stats[1] == 10) # num_relations
        self.assertTrue(stats[2] == 3300) # num_train
    
    def test_dataset_split_single_file(self):
        """
        Check if dataset_split works properly
        Expected behavior: if dataset_split is set to non-zero tuple and number
            of files is 1, the data file will be redistributed into train, 
            valid, and test sets according to the fraction specified by the 
            dataset_split tuple
        """
        output_dir = TEST_DIR

        stats = general_parser(
                    [str(Path(input_dir) / train_file)],
                    ["srd"], [output_dir], dataset_split = (0.1, 0.1))
        
        self.assertTrue(len(stats) == 5)
        self.assertTrue(stats[0] == 100)
        self.assertTrue(stats[1] == 10)
        self.assertTrue(stats[2] == 2400)
        self.assertTrue(stats[3] == 300)
        self.assertTrue(stats[4] == 300)
        

    def test_dataset_split_multi_files(self):
        """
        Check if dataset_split works properly
        Expected behavior: if dataset_split is set to non-zero tuple and number
            of files is greater than 3, the data files will be merged into 1 
            file and then redistributed into train, valid, and test sets
            accroding to the fraction specified by the dataset_split tuple
        """
        output_dir = TEST_DIR

        stats = general_parser(
                    [str(Path(input_dir) / train_file), # 1000
                     str(Path(input_dir) / valid_file), # 100
                     str(Path(input_dir) / test_file), # 100
                     str(Path(input_dir) / valid_file), # 100
                     str(Path(input_dir) / train_file), # 1000
                     str(Path(input_dir) / train_file)], # 1000
                     ["srd"], [output_dir], dataset_split = (0.1, 0.1)) 
        
        self.assertTrue(len(stats) == 5)
        self.assertTrue(stats[0] == 100)
        self.assertTrue(stats[1] == 10)
        self.assertTrue(stats[2] == 2640)
        self.assertTrue(stats[3] == 330)
        self.assertTrue(stats[4] == 330)


    