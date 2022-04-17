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


def run_configs(directory, model_dir=None, partitioned_eval=False, sequential_train_nodes=False):
    for filename in os.listdir(directory):
        if filename.startswith("M-"):
            config_file = directory / Path(filename)
            print("|||||||||||||||| RUNNING CONFIG ||||||||||||||||")
            print(config_file)
            config = m.config.loadConfig(config_file.__str__(), True)

            if model_dir is not None:
                config.storage.model_dir = model_dir + "/"
                relation_mapping_filepath = Path(config.storage.dataset.base_directory) / Path("edges") / Path("relation_mapping.txt")
                if relation_mapping_filepath.exists():
                    shutil.copy(str(relation_mapping_filepath), "{}/{}".format(config.storage.model_dir, "relation_mapping.txt"))

                node_mapping_filepath = Path(config.storage.dataset.base_directory) / Path("nodes") / Path("node_mapping.txt")
                if node_mapping_filepath.exists():
                    shutil.copy(str(node_mapping_filepath), "{}/{}".format(config.storage.model_dir, "node_mapping.txt"))
            
            if partitioned_eval:
                config.storage.full_graph_evaluation = False

            if sequential_train_nodes:
                config.storage.embeddings.options.node_partition_ordering = m.config.NodePartitionOrdering.SEQUENTIAL
                config.storage.features.options.node_partition_ordering = m.config.NodePartitionOrdering.SEQUENTIAL

            m.manager.marius_train(config)

def has_model_params(model_dir_path, task="lp", has_embeddings=False, has_relations=True):
    if not model_dir_path.exists():
        return False, "{} directory with model params not found".format(model_dir_path)
    
    model_file = model_dir_path / Path("model.pt")
    if not model_file.exists():
        return False, "{} not found".format(model_file)
    
    model_state_file = model_dir_path / Path("model_state.pt")
    if not model_state_file.exists():
        return False, "{} not found".format(model_state_file)
    
    node_mapping_file = model_dir_path / Path("node_mapping.txt")
    if not node_mapping_file.exists():
        return False, "{} not found".format(node_mapping_file)
    
    if has_relations:
        relation_mapping_file = model_dir_path / Path("node_mapping.txt")
        if not relation_mapping_file.exists():
            return False, "{} not found".format(relation_mapping_file)

    if task == "lp" or has_embeddings:
        embeddings_file = model_dir_path / Path("embeddings.bin")
        if not embeddings_file.exists():
            return False, "{} not found".format(embeddings_file)
        
        embeddings_state_file = model_dir_path / Path("embeddings_state.bin")
        if not embeddings_state_file.exists():
            return False, "{} not found".format(embeddings_state_file)
        
        rel_mapping_file = model_dir_path / Path("relation_mapping.txt")
        if not node_mapping_file.exists():
            return False, "{} not found".format(node_mapping_file)
    
    return True, ""


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
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                task="lp")

    @classmethod
    def tearDown(self):
        # pass
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_dm(self):
        name = "dm"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["distmult"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="lp")

        run_configs(self.output_dir / Path(name))
        model_dir_path = self.output_dir / Path("test_graph") / Path("model_0")
        ret, err = has_model_params(model_dir_path)
        assert ret == True, err

        run_configs(self.output_dir / Path(name))
        model_dir_path = self.output_dir / Path("test_graph") / Path("model_1")
        ret, err = has_model_params(model_dir_path)
        assert ret == True, err

        for i in range(2, 11):
            model_dir_path = self.output_dir / Path("test_graph") / Path("model_{}".format(i))
            model_dir_path.mkdir(parents=True, exist_ok=True)
        
        model_dir_path = self.output_dir / Path("test_graph") / Path("model_10")
        ret, err = has_model_params(model_dir_path)
        assert ret == False, err

        run_configs(self.output_dir / Path(name))
        ret, err = has_model_params(model_dir_path)
        assert ret == True, err

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path)
        assert ret == True, err

class TestNC(unittest.TestCase):

    output_dir = TMP_TEST_DIR / Path("relations")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "test_graph"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                feature_dim=10,
                                task="nc")

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs(self):
        name = "gs"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer", "gs_3_layer"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))
        
        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc")
        assert ret == True, err

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_async(self):
        name = "async"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["in_memory"],
                                     training_names=["async"],
                                     evaluation_names=["async"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc")
        assert ret == True, err

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_emb(self):
        name = "emb"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer_emb", "gs_3_layer_emb"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc", True)
        assert ret == True, err

class TestLPBufferNoRelations(unittest.TestCase):

    output_dir = TMP_TEST_DIR / Path("buffer_no_relations")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "test_graph"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                num_partitions=8,
                                splits=[.9, .05, .05],
                                task="lp")

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_dm(self):
        name = "dm"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["distmult"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="lp")

        run_configs(self.output_dir / Path(name))

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "lp", False)
        assert ret == True, err

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_partitioned_eval(self):
        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "partitioned_eval"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                num_partitions=8,
                                partitioned_eval=True,
                                task="lp")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["distmult"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync", "async", "async_deg", "async_filtered"],
                                     task="lp")

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "lp", False)
        assert ret == True, err

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
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                num_partitions=8,
                                feature_dim=10,
                                task="nc")

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_gs(self):
        name = "gs"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer", "gs_3_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc")
        assert ret == True, err

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_async(self):
        name = "async"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["async"],
                                     evaluation_names=["async"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc")
        assert ret == True, err

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_emb(self):
        name = "emb"
        shutil.copytree(self.output_dir / Path("test_graph"), self.output_dir / Path(name))

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer_emb", "gs_3_layer_emb"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc", True)
        assert ret == True, err

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_partitioned_eval(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "partitioned_eval"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                num_partitions=8,
                                partitioned_eval=True,
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer_emb", "gs_3_layer_emb", "gs_1_layer", "gs_3_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc", True)
        assert ret == True, err

    # @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    @pytest.mark.skip("Sequential ordering tests currently flakey at small scale")
    def test_sequential(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "sequential_ordering"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.1, .05, .05],
                                num_partitions=8,
                                partitioned_eval=True,
                                sequential_train_nodes=True,
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer_emb", "gs_3_layer_emb", "gs_1_layer", "gs_3_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name), partitioned_eval=True, sequential_train_nodes=True)

        model_dir_path = self.output_dir / Path(name)
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc", True)
        assert ret == True, err

        run_configs(self.output_dir / Path(name), partitioned_eval=False, sequential_train_nodes=True)

        model_dir_path = self.output_dir / Path(name) / Path("_1")
        run_configs(self.output_dir / Path(name), str(model_dir_path))
        ret, err = has_model_params(model_dir_path, "nc", True)
        assert ret == True, err