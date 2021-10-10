import unittest
from pathlib import Path
from marius.tools.postprocess import get_embs
from marius.tools.postprocess import output_embeddings
from marius.tools.preprocess import wn18
from marius.tools.config_generator import set_up_files
import subprocess
import shutil
import numpy as np
import torch

class TestPostprocess(unittest.TestCase):
    """
    Tests for postprocess
    """
    dataset_dir = Path("./output_dir")
    data_dir = Path("./data/")
    node_mapping_file = Path(dataset_dir) / Path("node_mapping.txt")
    rel_mapping_file = Path(dataset_dir) / Path("rel_mapping.txt")
    node_embs_file = Path(data_dir) / Path("marius/embeddings/embeddings.bin")
    lhs_embs_file = Path(data_dir) / Path("marius/relations/src_relations.bin")
    rhs_embs_file = Path(data_dir) / Path("marius/relations/dst_relations.bin")

    @classmethod
    def setUpClass(self):
        self.dataset_dir = set_up_files(self.dataset_dir)
        wn18(str(self.dataset_dir))
        
        if not Path("./data/marius/embeddings").exists():
            Path("./data/marius/embeddings").mkdir(parents=True)
        if not Path("./data/marius/relations").exists():
            Path("./data/marius/relations").mkdir(parents=True)
        
        node_embs = np.random.rand(409430,).astype(np.float32).flatten().tofile(self.node_embs_file)
        lhs_embs = np.random.rand(180,).astype(np.float32).flatten().tofile(self.lhs_embs_file)
        rhs_embs = np.random.rand(180,).astype(np.float32).flatten().tofile(self.rhs_embs_file)


    @classmethod
    def tearDownClass(self):
        if Path("./data").exists():
            shutil.rmtree(Path("./data"))
        
        if Path(self.dataset_dir).exists():
            shutil.rmtree(Path(self.dataset_dir))

    def test_get_embs(self):
        """
        Check if embeddings are returned in correct size
        """
        node_embs = get_embs(self.node_mapping_file, self.node_embs_file)
        lhs_embs = get_embs(self.rel_mapping_file, self.lhs_embs_file)
        rhs_embs = get_embs(self.rel_mapping_file, self.rhs_embs_file)

        self.assertEqual(node_embs.shape, (40943, 10))
        self.assertEqual(lhs_embs.shape, (18, 10))
        self.assertEqual(rhs_embs.shape, (18, 10))

    def test_output_embeddings(self):
        """
        Check if correct file format is created
        """
        fmts = ["CSV", "TSV", "TXT", "Tensor"]
        file_appendix = ["csv", "tsv", "txt", "pt"]
        
        node_embs, lhs_embs, rhs_embs = output_embeddings(self.data_dir, self.dataset_dir, Path("./output_dir"), fmts[0])
        self.assertTrue((self.dataset_dir / Path("node_embeddings." + file_appendix[0])).exists())
        self.assertTrue((self.dataset_dir / Path("src_relations_embeddings." + file_appendix[0])).exists())
        self.assertTrue((self.dataset_dir / Path("dst_relations_embeddings." + file_appendix[0])).exists())
        self.assertEqual(node_embs.shape, (40943, 10))
        self.assertEqual(lhs_embs.shape, (18, 10))
        self.assertEqual(rhs_embs.shape, (18, 10))

        node_embs, lhs_embs, rhs_embs = output_embeddings(self.data_dir, self.dataset_dir, Path("./output_dir"), fmts[1])
        self.assertTrue((self.dataset_dir / Path("node_embeddings." + file_appendix[1])).exists())
        self.assertTrue((self.dataset_dir / Path("src_relations_embeddings." + file_appendix[1])).exists())
        self.assertTrue((self.dataset_dir / Path("dst_relations_embeddings." + file_appendix[1])).exists())
        self.assertEqual(node_embs.shape, (40943, 10))
        self.assertEqual(lhs_embs.shape, (18, 10))
        self.assertEqual(rhs_embs.shape, (18, 10))

        node_embs, lhs_embs, rhs_embs = output_embeddings(self.data_dir, self.dataset_dir, Path("./output_dir"), fmts[2])
        self.assertTrue((self.dataset_dir / Path("node_embeddings." + file_appendix[2])).exists())
        self.assertTrue((self.dataset_dir / Path("src_relations_embeddings." + file_appendix[2])).exists())
        self.assertTrue((self.dataset_dir / Path("dst_relations_embeddings." + file_appendix[2])).exists())
        self.assertEqual(node_embs.shape, (40943, 10))
        self.assertEqual(lhs_embs.shape, (18, 10))
        self.assertEqual(rhs_embs.shape, (18, 10))

        node_embs, lhs_embs, rhs_embs = output_embeddings(self.data_dir, self.dataset_dir, Path("./output_dir"), fmts[3])
        self.assertTrue((self.dataset_dir / Path("node_embeddings." + file_appendix[3])).exists())
        self.assertTrue((self.dataset_dir / Path("src_relations_embeddings." + file_appendix[3])).exists())
        self.assertTrue((self.dataset_dir / Path("dst_relations_embeddings." + file_appendix[3])).exists())
        self.assertEqual(node_embs.shape, (40943, 10))
        self.assertEqual(lhs_embs.shape, (18, 10))
        self.assertEqual(rhs_embs.shape, (18, 10))

    def test_default_output(self):
        """
        Check if the default output is in csv format and in default location
        """
        subprocess.run(["python3", "./src/python/tools/postprocess.py", self.data_dir,
                        self.dataset_dir])
        self.assertTrue((self.dataset_dir / Path("node_embeddings.csv")).exists())
        self.assertTrue((self.dataset_dir / Path("src_relations_embeddings.csv")).exists())
        self.assertTrue((self.dataset_dir / Path("dst_relations_embeddings.csv")).exists())


        