import unittest
from pathlib import Path
import shutil
import subprocess
import numpy as np
import argparse
from marius.tools.preprocess import wn18
from marius.tools.predict import parse_infer_list
from marius.tools.predict import set_args
from marius.tools.predict import perform_link_prediction

class TestPredict(unittest.TestCase):
    """
    Tests for predict
    """
    dataset_dir = Path("./output_dir")
    data_dir = Path("./data/marius/")
    node_mapping_file = Path(dataset_dir) / Path("node_mapping.txt")
    rel_mapping_file = Path(dataset_dir) / Path("rel_mapping.txt")
    node_embs_file = Path(data_dir) / Path("embeddings/embeddings.bin")
    lhs_embs_file = Path(data_dir) / Path("relations/src_relations.bin")
    rhs_embs_file = Path(data_dir) / Path("relations/dst_relations.bin")

    @classmethod
    def setUpClass(self):
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
        
        if Path("./output_dir").exists():
            shutil.rmtree(Path("./output_dir"))

    def test_cmd_line_infer_list(self):
        """
        Check if inference can be extracted from command line correctly
        """
        parser = set_args()
        args = parser.parse_args(["./data/marius", "./output_dir", "3", 
                                  "-s", "s1", "-r", "r1"])
        args_dict = vars(args)
        self.assertEqual(args_dict.get("src"), "s1")
        self.assertEqual(args_dict.get("rel"), "r1")
        self.assertEqual(args_dict.get("decoder"), "DisMult")

        self.assertEqual(parse_infer_list(args), [["s1", "r1", ""]])

    def test_file_infer_list(self):
        """
        Check if inference can be extracted from file correctly
        """
        infer_list = [["01371092","_hypernym",""],
                      ["03902220","_part_of",""],
                      ["","_hypernym","08102555"],
                      ["08621598","_hypernym",""]]
        np.savetxt("./infer_list", infer_list, delimiter=",", fmt="%s")
        parser = set_args()
        args = parser.parse_args(["./data/marius", "./output_dir", "3", 
                    "-f", "./infer_list", "-dc", "TransE"])
        args_dict = vars(args)
        self.assertEqual(args_dict.get("decoder"), "TransE")
        self.assertTrue(np.array_equal(parse_infer_list(args), infer_list))

        Path("./infer_list").unlink()
    
    def test_file_cmd_line_infer_list(self):
        """
        Check if inference can be extracted from both cmd line and file
            correctly
        """
        infer_list = [["01371092","_hypernym",""],
                      ["03902220","_part_of",""],
                      ["","_hypernym","08102555"],
                      ["08621598","_hypernym",""]]
        result = [["01371092","_hypernym",""],
                  ["03902220","_part_of",""],
                  ["","_hypernym","08102555"],
                  ["08621598","_hypernym",""],
                  ["s1", "r1", ""]]
        
        np.savetxt("./infer_list", infer_list, delimiter=",", fmt="%s")
        parser = set_args()
        args = parser.parse_args(["./data/marius/", "./output_dir", "3", 
                    "-f", "./infer_list", "-dc", "TransE", 
                    "-s", "s1", "-r", "r1"])
        args_dict = vars(args)

        self.assertEqual(args_dict.get("decoder"), "TransE")
        self.assertTrue(np.array_equal(np.array(parse_infer_list(args)), 
                    np.array(result)))

        Path("./infer_list").unlink()

    def test_top_k(self):
        """
        Check if top_nodes_list returned has correct shape for all 3 
            decoders
        """
        infer_list = [["01371092","_hypernym",""],
                      ["03902220","_part_of",""],
                      ["","_hypernym","08102555"],
                      ["08621598","_hypernym",""]]
        
        decoders = ["DisMult", "TransE", "ComplEx"]
        ks = [4,6,8]

        for i in range(len(decoders)):
            top_nodes_list = np.array(perform_link_prediction(self.data_dir, 
                    self.dataset_dir, infer_list, ks[i], decoders[i]))
            self.assertEqual(top_nodes_list.shape[0], 4)
            self.assertEqual(top_nodes_list.shape[1], ks[i])

    
        
