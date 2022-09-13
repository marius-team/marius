import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TMP_TEST_DIR
from test.test_configs.generate_test_configs import generate_configs_for_dataset
from test.test_data.generate import generate_random_dataset

import marius as m


def replace_string_in_file(filepath, before, after):
    os.system("sed -i -E 's@{}@{}@g' {}".format(before, after, filepath.__str__()))


def get_line_in_file(filepath, line_num):
    return os.popen("sed '{}!d' {}".format(line_num, filepath.__str__())).read().lstrip()


def run_config(config_file, enable_checkpointing, checkpoint_interval, save_state):
    config = m.config.loadConfig(config_file.__str__(), True)
    config.training.num_epochs = 6
    if enable_checkpointing:
        config.training.checkpoint.interval = checkpoint_interval
        config.training.checkpoint.save_state = save_state
    m.manager.marius_train(config)


class TestIntervalCheckpointing(unittest.TestCase):
    base_dir = None
    config_file = None

    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()
        self.base_dir = TMP_TEST_DIR

    @classmethod
    def tearDown(self):
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    def init_dataset_dir(self, name):
        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        generate_random_dataset(
            output_dir=Path(self.base_dir) / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            task="lp",
        )

        generate_configs_for_dataset(
            Path(self.base_dir) / Path(name),
            model_names=["distmult"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        for filename in os.listdir(Path(self.base_dir) / Path(name)):
            if filename.startswith("M-"):
                self.config_file = Path(self.base_dir) / Path(name) / Path(filename)

    def test_checkpointing_with_state(self):
        name = "test_checkpointing_with_state"
        self.init_dataset_dir(name)

        # runs for a total of 6 epochs, checkpoints every 2 epochs. so checkpoint_2 & checkpoint_4 should exist
        # checkpoint 6 shouldn't exist
        run_config(self.config_file, True, 2, True)

        config = m.config.loadConfig(self.config_file.__str__(), False)
        checkpoint_2_path = Path(config.storage.model_dir) / Path("checkpoint_2")
        checkpoint_4_path = Path(config.storage.model_dir) / Path("checkpoint_4")
        checkpoint_6_path = Path(config.storage.model_dir) / Path("checkpoint_6")
        assert checkpoint_2_path.exists(), "Expected to see checkpointed model and params in {}, but not found".format(
            str(checkpoint_2_path)
        )
        assert checkpoint_4_path.exists(), "Expected to see checkpointed model and params in {}, but not found".format(
            str(checkpoint_4_path)
        )
        assert not checkpoint_6_path.exists(), "{} shouldn't have been created".format(str(checkpoint_6_path))

        checkpoint_files = ["model.pt", "model_state.pt", "embeddings.bin", "embeddings_state.bin"]
        for checkpoint_id in ["checkpoint_2", "checkpoint_4"]:
            for f in checkpoint_files:
                file_path_ = Path(config.storage.model_dir) / Path(checkpoint_id) / Path(f)
                assert file_path_.exists(), "Expected to see checkpointed file {}, but not found".format(
                    str(file_path_)
                )

        # resume training from checkpoint_4 and further train 5 epochs with checkpoint disabled.
        # so the model stored would have ideally been trained for 9 epochs.
        full_config_path = Path(config.storage.model_dir) / Path("full_config.yaml")
        replace_string_in_file(
            full_config_path,
            "resume_from_checkpoint:.*",
            "resume_from_checkpoint: {}/checkpoint_4".format(config.storage.model_dir),
        )
        replace_string_in_file(full_config_path, "model_dir:.*", "")
        run_config(full_config_path, False, -1, False)

        config = m.config.loadConfig(self.config_file.__str__(), False)
        metadata_file_path = Path(config.storage.model_dir) / Path("metadata.csv")

        trained_epochs = int(get_line_in_file(metadata_file_path, 2))
        assert trained_epochs == 10, "Expected to see trained epochs as {} in {}, but found {}".format(
            10, str(metadata_file_path), trained_epochs
        )

    def test_checkpointing_wo_state(self):
        name = "test_checkpointing_wo_state"
        self.init_dataset_dir(name)

        # runs for a total of 6 epochs, checkpoints every 3 epochs. so checkpoint_3 alone should exist
        run_config(self.config_file, True, 3, False)

        config = m.config.loadConfig(self.config_file.__str__(), False)
        checkpoint_3_path = Path(config.storage.model_dir) / Path("checkpoint_3")
        checkpoint_6_path = Path(config.storage.model_dir) / Path("checkpoint_6")
        assert checkpoint_3_path.exists(), "Expected to see checkpointed model and params in {}, but not found".format(
            str(checkpoint_3_path)
        )
        assert not checkpoint_6_path.exists(), "{} shouldn't have been created".format(str(checkpoint_6_path))

        checkpoint_files = ["model.pt", "model_state.pt", "embeddings.bin"]
        for checkpoint_id in ["checkpoint_3"]:
            for f in checkpoint_files:
                file_path_ = Path(config.storage.model_dir) / Path(checkpoint_id) / Path(f)
                assert file_path_.exists(), "Expected to see checkpointed file {}, but not found".format(
                    str(file_path_)
                )
