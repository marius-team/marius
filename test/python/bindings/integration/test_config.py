import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TMP_TEST_DIR
from test.test_configs.generate_test_configs import generate_configs_for_dataset

from omegaconf import OmegaConf

import marius.tools.configuration.marius_config
from marius.config import loadConfig


class TestConfig(unittest.TestCase):
    """
    Basic tests for loadConfig and the returned MariusConfig object.
    """

    output_dir = TMP_TEST_DIR / Path("config")

    ds_config = marius.tools.configuration.marius_config.DatasetConfig()
    ds_config.dataset_dir = output_dir.__str__()
    ds_config.num_edges = 1000
    ds_config.num_nodes = 100
    ds_config.num_relations = 1
    ds_config.num_train = 100
    ds_config.num_valid = 10
    ds_config.num_test = 10
    ds_config.initialized = False

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        OmegaConf.save(self.ds_config, self.output_dir / Path("dataset.yaml"))

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_missing_config(self):
        try:
            loadConfig("foo.yaml")
            raise RuntimeError("Exception not thrown")
        except Exception as e:
            assert "No such file or directory" in e.__str__()

    def test_missing_dataset_yaml(self):
        generate_configs_for_dataset(
            self.output_dir,
            model_names=["distmult"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        os.system("rm {}".format(self.output_dir / Path("dataset.yaml")))
        for filename in os.listdir(self.output_dir):
            if filename.startswith("M-"):
                try:
                    config_file = self.output_dir / Path(filename)
                    _ = loadConfig(config_file.__str__(), save=True)
                    raise RuntimeError("Exception not thrown")
                except Exception as e:
                    assert "expected to see dataset.yaml file" in e.__str__()

        shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        OmegaConf.save(self.ds_config, self.output_dir / Path("dataset.yaml"))

        generate_configs_for_dataset(
            self.output_dir,
            model_names=["gs_1_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        os.system("rm {}".format(self.output_dir / Path("dataset.yaml")))
        for filename in os.listdir(self.output_dir):
            if filename.startswith("M-"):
                try:
                    config_file = self.output_dir / Path(filename)
                    _ = loadConfig(config_file.__str__(), save=True)
                    raise RuntimeError("Exception not thrown")
                except Exception as e:
                    assert "expected to see dataset.yaml file" in e.__str__()

    def test_load_config(self):
        generate_configs_for_dataset(
            self.output_dir,
            model_names=["distmult, gs_1_layer, gs_3_layer, gat_1_layer, gat_3_layer"],
            storage_names=["in_memory, part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        # check that each generated config can be parsed and it's members accessed.
        for filename in os.listdir(self.output_dir):
            if filename.startswith("M-"):
                config_file = self.output_dir / Path(filename)

                config = loadConfig(config_file.__str__(), save=True)
                loaded_full_config = loadConfig((config.storage.model_dir / Path("full_config.yaml")).__str__())
                assert loaded_full_config.model.random_seed == config.model.random_seed

                assert config.model is not None
                assert config.storage is not None
                assert config.training is not None
                assert config.evaluation is not None

                assert config.model.encoder is not None
                assert config.model.decoder is not None

                assert config.storage.dataset.dataset_dir.rstrip("/") == self.output_dir.__str__()
                assert config.storage.dataset.num_edges == 1000
                assert config.storage.dataset.num_nodes == 100
                assert config.storage.dataset.num_relations == 1
                assert config.storage.dataset.num_train == 100
                assert config.storage.dataset.num_valid == 10
                assert config.storage.dataset.num_test == 10

                assert config.training is not None
                assert config.evaluation is not None

                config.model.random_seed = 0
                assert config.model.random_seed == 0

        # reset directory
        shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        OmegaConf.save(self.ds_config, self.output_dir / Path("dataset.yaml"))

        generate_configs_for_dataset(
            self.output_dir,
            model_names=["gs_1_layer", "gs_3_layer", "gat_1_layer", "gat_3_layer"],
            storage_names=["in_memory", "part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        # check that each generated config can be parsed and it's members accessed.
        for filename in os.listdir(self.output_dir):
            if filename.startswith("M-"):
                config_file = self.output_dir / Path(filename)

                config = loadConfig(config_file.__str__(), save=True)
                loaded_full_config = loadConfig((config.storage.model_dir / Path("full_config.yaml")).__str__())
                assert loaded_full_config.model.random_seed == config.model.random_seed

                assert config.model is not None
                assert config.storage is not None
                assert config.training is not None
                assert config.evaluation is not None

                assert config.model.encoder is not None
                assert config.model.decoder is not None

                assert config.storage.dataset.dataset_dir.rstrip("/") == self.output_dir.__str__()
                assert config.storage.dataset.num_edges == 1000
                assert config.storage.dataset.num_nodes == 100
                assert config.storage.dataset.num_relations == 1
                assert config.storage.dataset.num_train == 100
                assert config.storage.dataset.num_valid == 10
                assert config.storage.dataset.num_test == 10

                assert config.training is not None
                assert config.evaluation is not None

                config.model.random_seed = 0
                assert config.model.random_seed == 0
