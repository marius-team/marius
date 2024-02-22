import glob
import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TMP_TEST_DIR
from test.test_configs.generate_test_configs import generate_configs_for_dataset
from test.test_data.generate import generate_random_dataset

import pandas as pd

import marius as m
from marius.tools.postprocess.in_memory_exporter import InMemoryExporter


def check_output(output_dir, fmt, has_rels=False):
    # check created
    assert (output_dir / ("embeddings." + fmt)).exists()
    assert (output_dir / ("encoded_nodes." + fmt)).exists()
    assert (output_dir / "model.pt").exists()

    # check embeddings
    if fmt == "csv":
        base_embeddings_df = pd.read_csv(
            output_dir / ("embeddings." + fmt),
            header=0,
            converters={"embedding": lambda x: x.strip("[]").strip().split()},
        )

        assert base_embeddings_df.shape[0] == 100  # check matches number of nodes
        assert base_embeddings_df.shape[1] == 2  # has two columns
        assert len(base_embeddings_df["embedding"][0]) == 10
    elif fmt == "parquet":
        base_embeddings_df = pd.read_parquet(output_dir / ("embeddings." + fmt))
        assert base_embeddings_df.shape[0] == 100  # check matches number of nodes
        assert base_embeddings_df.shape[1] == 2  # has two columns
        assert len(base_embeddings_df.iloc[0, 1]) == 10  # the second column has list values for the embeddings
    else:
        raise RuntimeError("Unknown format")

    if fmt == "csv":
        encoded_nodes_df = pd.read_csv(
            output_dir / ("encoded_nodes." + fmt),
            header=0,
            converters={"embedding": lambda x: x.strip("[]").strip().split()},
        )

        assert encoded_nodes_df.shape[0] == 100
        assert encoded_nodes_df.shape[1] == 2
        assert len(encoded_nodes_df["embedding"][0]) == 10
    elif fmt == "parquet":
        encoded_nodes_df = pd.read_parquet(output_dir / ("encoded_nodes." + fmt))
        assert encoded_nodes_df.shape[0] == 100
        assert encoded_nodes_df.shape[1] == 2
        assert len(encoded_nodes_df.iloc[0, 1]) == 10

    if has_rels:
        if fmt == "csv":
            rel_embs_df = pd.read_csv(
                output_dir / ("relation_embeddings." + fmt),
                header=0,
                converters={"embedding": lambda x: x.strip("[]").strip().split()},
            )

            assert rel_embs_df.shape[0] == 10
            assert rel_embs_df.shape[1] == 2
            assert len(rel_embs_df["embedding"][0]) == 10

            rel_embs_df = pd.read_csv(
                output_dir / ("inverse_relation_embeddings." + fmt),
                header=0,
                converters={"embedding": lambda x: x.strip("[]").strip().split()},
            )

            assert rel_embs_df.shape[0] == 10
            assert rel_embs_df.shape[1] == 2
            assert len(rel_embs_df["embedding"][0]) == 10
        elif fmt == "parquet":
            rel_embs_df = pd.read_parquet(output_dir / ("relation_embeddings." + fmt))
            assert rel_embs_df.shape[0] == 10
            assert rel_embs_df.shape[1] == 2
            assert len(rel_embs_df.iloc[0, 1]) == 10

            rel_embs_df = pd.read_parquet(output_dir / ("inverse_relation_embeddings." + fmt))
            assert rel_embs_df.shape[0] == 10
            assert rel_embs_df.shape[1] == 2
            assert len(rel_embs_df.iloc[0, 1]) == 10


class TestLP(unittest.TestCase):
    config_file = None

    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

        base_dir = TMP_TEST_DIR

        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "export_lp"
        generate_random_dataset(
            output_dir=base_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            splits=[0.9, 0.05, 0.05],
            task="lp",
        )

        generate_configs_for_dataset(
            base_dir / Path(name),
            model_names=["gs_1_layer"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="lp",
        )

        self.model_dir = Path(base_dir) / name / "model_0"

        for filename in os.listdir(base_dir / Path(name)):
            if filename.startswith("M-"):
                self.config_file = base_dir / Path(name) / Path(filename)

        config = m.config.loadConfig(self.config_file.__str__(), True)
        config.storage.export_encoded_nodes = True
        m.manager.marius_train(config)

    @classmethod
    def tearDown(self):
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    def test_export_csv(self):
        assert self.model_dir.exists()
        exporter = InMemoryExporter(self.model_dir, fmt="csv")

        exporter.export(self.model_dir)
        check_output(self.model_dir, fmt="csv")

        exporter.export(self.model_dir.parent / "model_tmp")
        check_output(self.model_dir.parent / "model_tmp", fmt="csv", has_rels=True)

    def test_export_binary(self):
        assert self.model_dir.exists()
        exporter = InMemoryExporter(self.model_dir, fmt="bin")

        # nothing new should be created since the output dir matches input dir
        exporter.export(self.model_dir)

        # copies full model directory
        exporter.export(self.model_dir.parent / "model_tmp")

        # check contents of input match contents of output
        input_files = glob.glob(self.model_dir.__str__() + "/*")
        output_files = glob.glob((self.model_dir.parent / "model_tmp").__str__() + "/*")
        assert len(input_files) == len(output_files)

    def test_export_parquet(self):
        assert self.model_dir.exists()
        exporter = InMemoryExporter(self.model_dir, fmt="parquet")

        exporter.export(self.model_dir)
        check_output(self.model_dir, fmt="parquet")

        exporter.export(self.model_dir.parent / "model_tmp")
        check_output(self.model_dir.parent / "model_tmp", fmt="parquet", has_rels=True)

    # TODO add testing support for s3. Need a test bucket somewhere
    # def test_export_s3(self):
    #     assert self.model_dir.exists()
    #     exporter = InMemoryExporter(self.model_dir)
    #     exporter.export(s3_path)

    def test_export_no_model(self):
        try:
            InMemoryExporter(Path("TEST_NOT_A_DIR"))
            raise RuntimeError("Exception not thrown")
        except RuntimeError:
            pass

    def test_export_overwrite(self):
        test_dir = self.model_dir.parent / "model_tmp"
        exporter = InMemoryExporter(self.model_dir, fmt="csv", overwrite=False)
        exporter.export(test_dir)
        check_output(test_dir, fmt="csv", has_rels=True)

        try:
            exporter.export(test_dir)
            raise RuntimeError("Exception not thrown")
        except RuntimeError:
            pass

        exporter.overwrite = True
        exporter.export(test_dir)
        check_output(test_dir, fmt="csv", has_rels=True)


class TestNC(unittest.TestCase):
    config_file = None

    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

        base_dir = TMP_TEST_DIR

        num_nodes = 100
        num_rels = 1
        num_edges = 1000

        name = "nc_export"
        generate_random_dataset(
            output_dir=base_dir / Path(name),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_rels=num_rels,
            feature_dim=10,
            splits=[0.9, 0.05, 0.05],
            task="nc",
        )

        generate_configs_for_dataset(
            base_dir / Path(name),
            model_names=["gs_1_layer_emb"],
            storage_names=["in_memory"],
            training_names=["sync"],
            evaluation_names=["sync"],
            task="nc",
        )

        self.model_dir = Path(base_dir) / name / "model_0"

        for filename in os.listdir(base_dir / Path(name)):
            if filename.startswith("M-"):
                self.config_file = base_dir / Path(name) / Path(filename)

        config = m.config.loadConfig(self.config_file.__str__(), True)
        config.storage.export_encoded_nodes = True
        m.manager.marius_train(config)

    @classmethod
    def tearDown(self):
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    def test_export_csv(self):
        assert self.model_dir.exists()
        exporter = InMemoryExporter(self.model_dir, fmt="csv")

        exporter.export(self.model_dir)
        check_output(self.model_dir, fmt="csv")

        exporter.export(self.model_dir.parent / "model_tmp")
        check_output(self.model_dir.parent / "model_tmp", fmt="csv")

    def test_export_binary(self):
        assert self.model_dir.exists()
        exporter = InMemoryExporter(self.model_dir, fmt="bin")

        # nothing new should be created since the output dir matches input dir
        exporter.export(self.model_dir)

        # copies full model directory
        exporter.export(self.model_dir.parent / "model_tmp")

        # check contents of input match contents of output
        input_files = glob.glob(self.model_dir.__str__() + "/*")
        output_files = glob.glob((self.model_dir.parent / "model_tmp").__str__() + "/*")
        assert len(input_files) == len(output_files)

    def test_export_parquet(self):
        assert self.model_dir.exists()
        exporter = InMemoryExporter(self.model_dir, fmt="parquet")

        exporter.export(self.model_dir)
        check_output(self.model_dir, fmt="parquet")

        exporter.export(self.model_dir.parent / "model_tmp")
        check_output(self.model_dir.parent / "model_tmp", fmt="parquet")

    # TODO add testing support for s3. Need a test bucket somewhere
    # def test_export_s3(self):
    #     assert self.model_dir.exists()
    #     exporter = InMemoryExporter(self.model_dir)
    #     exporter.export(s3_path)

    def test_export_no_model(self):
        try:
            InMemoryExporter(Path("TEST_NOT_A_DIR"))
            raise RuntimeError("Exception not thrown")
        except RuntimeError:
            pass

    def test_export_overwrite(self):
        test_dir = self.model_dir.parent / "model_tmp"
        exporter = InMemoryExporter(self.model_dir, fmt="csv", overwrite=False)
        exporter.export(test_dir)
        check_output(test_dir, fmt="csv")

        try:
            exporter.export(test_dir)
            raise RuntimeError("Exception not thrown")
        except RuntimeError:
            pass

        exporter.overwrite = True
        exporter.export(test_dir)
        check_output(test_dir, fmt="csv")
