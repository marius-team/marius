import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TMP_TEST_DIR
from test.test_configs.generate_test_configs import generate_configs_for_dataset
from test.test_data.generate import generate_random_dataset

import pytest

import marius as m


def run_configs(directory, partitioned_eval=False):
    for filename in os.listdir(directory):
        if filename.startswith("M-"):
            config_file = directory / Path(filename)
            print("|||||||||||||||| RUNNING CONFIG ||||||||||||||||")
            print(config_file)
            config = m.config.loadConfig(config_file.__str__(), True)

            if partitioned_eval:
                config.storage.full_graph_evaluation = False

            m.manager.marius_train(config)


class TestLPStorage(unittest.TestCase):
    output_dir = TMP_TEST_DIR / Path("storage")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid(self):
        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "no_valid"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.1],
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train(self):
        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "only_train"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid_no_relations(self):
        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "no_valid_no_relations"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.1],
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train_no_relations(self):
        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "only_train_no_relations"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid_buffer(self):
        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "no_valid_buffer"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.1],
            num_partitions=8,
            partitioned_eval=True,
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train_buffer(self):
        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "only_train_buffer"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            num_partitions=8,
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid_buffer_no_relations(self):
        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "no_valid_buffer_no_relations"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.1],
            num_partitions=8,
            partitioned_eval=True,
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train_buffer_no_relations(self):
        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "only_train_buffer_no_relations"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            num_partitions=8,
            task="lp",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))
