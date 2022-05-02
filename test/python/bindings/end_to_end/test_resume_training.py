import unittest
import shutil
from pathlib import Path
import pytest
import os
import marius as m
import torch

from test.python.constants import TMP_TEST_DIR, TESTING_DATA_DIR
from test.test_data.generate import generate_random_dataset
from test.test_configs.generate_test_configs import generate_configs_for_dataset

def replace_string_in_file(filepath, before, after):
    os.system("sed -i -E 's@{}@{}@g' {}".format(before, after, filepath))

def get_line_in_file(filepath, line_num):
    return os.popen("sed '{}!d' {}".format(line_num, filepath)).read().lstrip()

def run_config(config_file):
    config = m.config.loadConfig(config_file, True)
    m.manager.marius_train(config)

class TestResumeTraining(unittest.TestCase):
    base_dir = None
    config_file = None
    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()
        self.base_dir = TMP_TEST_DIR

    @classmethod
    def tearDown(self):
        pass
        # if Path(TMP_TEST_DIR).exists():
        #     shutil.rmtree(Path(TMP_TEST_DIR))

    def init_dataset_dir(self, name):
        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        generate_random_dataset(output_dir=Path(self.base_dir) / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                task="lp")
        
        generate_configs_for_dataset(Path(self.base_dir) / Path(name),
                                     model_names=["distmult"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="lp")
        
        for filename in os.listdir(Path(self.base_dir) / Path(name)):
            if filename.startswith("M-"):
                self.config_file = Path(self.base_dir) / Path(name) / Path(filename)
                config = m.config.loadConfig(self.config_file.__str__(), True)
                m.manager.marius_train(config)


    def test_resume_training_without_model_dir(self):
        self.init_dataset_dir("without_model_dir")
        
        config = m.config.loadConfig(self.config_file.__str__(), False)
        metadata_file_path = Path(config.storage.model_dir) / Path("metadata.csv")

        trained_epochs = int(get_line_in_file(metadata_file_path.__str__(), 2))
        assert trained_epochs == 2, "Expected to see trained epochs as {} in {}, but found {}".format(2, str(metadata_file_path), trained_epochs)

        full_config_path = Path(config.storage.model_dir) / Path("full_config.yaml")
        replace_string_in_file(full_config_path.__str__(), 'resume_training: false', 'resume_training: true')
        replace_string_in_file(full_config_path.__str__(), 'model_dir:.*', '')
        run_config(full_config_path.__str__())
        
        # overwrites the model_0 directory with new model data
        trained_epochs = int(get_line_in_file(metadata_file_path.__str__(), 2))
        assert trained_epochs == 4, "Expected to see trained epochs as {} in {}, but found {}".format(4, str(metadata_file_path), trained_epochs)
    
    def test_resume_training_with_checkpoint_dir(self):
        self.init_dataset_dir("with_checkpoint_dir")
        
        config = m.config.loadConfig(self.config_file.__str__(), False)
        metadata_file_path = Path(config.storage.model_dir) / Path("metadata.csv")

        trained_epochs = int(get_line_in_file(metadata_file_path.__str__(), 2))
        assert trained_epochs == 2, "Expected to see trained epochs as {} in {}, but found {}".format(2, str(metadata_file_path), trained_epochs)
        
        full_config_path = Path(config.storage.model_dir) / Path("full_config.yaml")
        replace_string_in_file(full_config_path.__str__() , 'resume_training: false', 'resume_training: true')
        replace_string_in_file(full_config_path.__str__() , 'model_dir:.*', '')
        replace_string_in_file(full_config_path.__str__() , 'resume_from_checkpoint:.*', "resume_from_checkpoint: {}".format(config.storage.model_dir))
        
        # creates model_1 directory with model data
        run_config(full_config_path.__str__())
        
        config = m.config.loadConfig(full_config_path.__str__(), False)
        metadata_file_path = Path(config.storage.model_dir) / Path("metadata.csv")
        trained_epochs = int(get_line_in_file(metadata_file_path.__str__(), 2))
        assert trained_epochs == 4, "Expected to see trained epochs as {} in {}, but found {}".format(4, str(metadata_file_path), trained_epochs)