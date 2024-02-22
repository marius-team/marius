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


class TestLP(unittest.TestCase):
    output_dir = TMP_TEST_DIR / Path("relations")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "test_graph"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            task="lp",
        )

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_dm(self):
        name = "dm"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

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
    def test_gs(self):
        name = "gs"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer", "gs_3_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs_uniform(self):
        name = "gs_uniform"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_uniform", "gs_3_layer_uniform"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skip("GAT only supported for GPU")
    def test_gat(self):
        name = "gat"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gat_1_layer", "gat_3_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_sync_training(self):
        name = "sync_training"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync_deg", "sync_filtered"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Async test currently flakey.")
    def test_async_training(self):
        name = "async_training"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["async", "async_deg", "async_filtered"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_sync_eval(self):
        name = "sync_eval"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync", "sync_deg", "sync_filtered"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Async test currently flakey.")
    def test_async_eval(self):
        name = "async_eval"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["async", "async_deg", "async_filtered"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))


class TestLPNoRelations(unittest.TestCase):
    output_dir = TMP_TEST_DIR / Path("no_relations")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "test_graph"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            task="lp",
        )

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_dm(self):
        name = "dm"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

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
    def test_gs(self):
        name = "gs"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer", "gs_3_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs_uniform(self):
        name = "gs_uniform"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_uniform", "gs_3_layer_uniform"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skip("GAT only supported for GPU")
    def test_gat(self):
        name = "gat"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gat_1_layer", "gat_3_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_sync_training(self):
        name = "sync_training"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync_deg", "sync_filtered"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Async test currently flakey.")
    def test_async_training(self):
        name = "async_training"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["async", "async_deg", "async_filtered"],
            evaluation_names=["sync"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_sync_eval(self):
        name = "sync_eval"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync", "sync_deg", "sync_filtered"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Async test currently flakey.")
    def test_async_eval(self):
        name = "async_eval"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["distmult", "gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["async", "async_deg", "async_filtered"],
            task="lp",
        )

        run_configs(self.output_dir / Path(name))
