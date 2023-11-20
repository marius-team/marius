import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TMP_TEST_DIR
from test.test_configs.generate_test_configs import generate_configs_for_dataset
from test.test_data.generate import generate_random_dataset

import marius as m
import pytest


def run_configs(directory, partitioned_eval=False, sequential_train_nodes=False):
    for filename in os.listdir(directory):
        if filename.startswith("M-"):
            config_file = directory / Path(filename)
            print("|||||||||||||||| RUNNING CONFIG ||||||||||||||||")
            print(config_file)
            config = m.config.loadConfig(config_file.__str__(), True)

            if partitioned_eval:
                config.storage.full_graph_evaluation = False

            if sequential_train_nodes:
                config.storage.embeddings.options.node_partition_ordering = m.config.NodePartitionOrdering.SEQUENTIAL
                config.storage.features.options.node_partition_ordering = m.config.NodePartitionOrdering.SEQUENTIAL

            m.manager.marius_train(config)


class TestNCBuffer(unittest.TestCase):
    output_dir = TMP_TEST_DIR / Path("buffer")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "test_graph"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            num_partitions=8,
            feature_dim=10,
            task="nc",
        )

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs(self):
        name = "gs"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer", "gs_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs_uniform(self):
        name = "gs_uniform"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_uniform", "gs_3_layer_uniform"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skip("GAT only supported for GPU")
    def test_gat(self):
        name = "gat"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gat_1_layer", "gat_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Async test currently flakey.")
    def test_async(self):
        name = "async"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer"],
            storage_names=["part_buffer"],
            training_names=["async"],
            evaluation_names=["async"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_emb(self):
        name = "emb"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_emb", "gs_3_layer_emb"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_partitioned_eval(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "partitioned_eval"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            num_partitions=8,
            partitioned_eval=True,
            feature_dim=10,
            task="nc",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_emb", "gs_3_layer_emb", "gs_1_layer", "gs_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Sequential ordering tests currently flakey at small scale")
    def test_sequential(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "sequential_ordering"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.1, 0.05, 0.05],
            num_partitions=8,
            partitioned_eval=True,
            sequential_train_nodes=True,
            feature_dim=10,
            task="nc",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_emb", "gs_3_layer_emb", "gs_1_layer", "gs_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name), partitioned_eval=True, sequential_train_nodes=True)
        run_configs(self.output_dir / Path(name), partitioned_eval=False, sequential_train_nodes=True)


class TestNCBufferNoRelations(unittest.TestCase):
    output_dir = TMP_TEST_DIR / Path("buffer_no_relations")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "test_graph"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            num_partitions=8,
            feature_dim=10,
            task="nc",
        )

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs(self):
        name = "gs"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer", "gs_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs_uniform(self):
        name = "gs_uniform"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_uniform", "gs_3_layer_uniform"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skip("GAT only supported for GPU")
    def test_gat(self):
        name = "gat"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gat_1_layer", "gat_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Async test currently flakey.")
    def test_async(self):
        name = "async"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer"],
            storage_names=["part_buffer"],
            training_names=["async"],
            evaluation_names=["async"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_emb(self):
        name = "emb"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_emb", "gs_3_layer_emb"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_partitioned_eval(self):
        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "partitioned_eval"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            num_partitions=8,
            partitioned_eval=True,
            feature_dim=10,
            task="nc",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_emb", "gs_3_layer_emb"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Sequential ordering tests currently flakey at small scale")
    def test_sequential(self):
        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "sequential_ordering"
        generate_random_dataset(
            output_dir=self.output_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.1, 0.05, 0.05],
            num_partitions=8,
            partitioned_eval=True,
            sequential_train_nodes=True,
            feature_dim=10,
            task="nc",
        )

        generate_configs_for_dataset(
            self.output_dir / Path(name),
            model_names=["gs_1_layer_emb", "gs_3_layer_emb", "gs_1_layer", "gs_3_layer"],
            storage_names=["part_buffer"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        run_configs(self.output_dir / Path(name), partitioned_eval=True, sequential_train_nodes=True)
        run_configs(self.output_dir / Path(name), partitioned_eval=False, sequential_train_nodes=True)
