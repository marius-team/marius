from pathlib import Path
import sys

import numpy as np
import pandas as pd
from preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file

import torch  # isort:skip

class OGBLCollab(LinkPredictionDataset):
    """
    Open Graph Benchmark: collab

    The ogbl-collab dataset is a weighted directed graph, representing a subset of the collaboration network
    between authors indexed by MAG. Each node represents an author and edges indicate the collaboration between 
    authors. All nodes come with 128-dimensional features, obtained by averaging the word embeddings of papers 
    that are published by the authors. All edges are associated with two meta-information: the year and the 
    edge weight, representing the number of co-authored papers published in that year. The graph can be viewed 
    as a dynamic multi-graph since there can be multiple edges between two nodes if they collaborate in more 
    than one year.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbl_citation2"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/linkproppred/collab.zip"
        self.node_ids = None

    def download(self, overwrite=False):
        self.input_train_edges_file = self.output_directory / Path("train.pt")
        self.input_valid_edges_file = self.output_directory / Path("valid.pt")
        self.input_test_edges_file = self.output_directory / Path("test.pt")

        download = False
        if overwrite:
            download = True
        elif not self.input_train_edges_file.exists():
            download = True
        elif not self.input_valid_edges_file.exists():
            download = True
        elif not self.input_test_edges_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            for file in (self.output_directory / Path("collab/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))
        
        # Read in the nodes
        nodes_path = Path(self.output_directory).joinpath("collab", "raw", "num-node-list.csv.gz")
        df = pd.read_csv(nodes_path, compression = 'gzip', header = None)
        self.num_nodes = df.iloc[0][0]

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, 
        sequential_train_nodes=False, partitioned_eval=False):  
        # Read in the training data
        train_idx = torch.load(self.input_train_edges_file)
        train_list = train_idx.get("edge")
        train_weights = train_idx.get("weight").reshape(-1, 1)
        train_edges = np.hstack([train_list, train_weights])

        # Read in the valid data
        valid_idx = torch.load(self.input_valid_edges_file)
        valid_list = valid_idx.get("edge")
        valid_weights = valid_idx.get("weight").reshape(-1, 1)
        valid_edges = np.hstack([valid_list, valid_weights])

        # Read in the test data
        test_idx = torch.load(self.input_test_edges_file)
        test_list = test_idx.get("edge")
        test_weights = test_idx.get("weight").reshape(-1, 1)
        test_edges = np.hstack([test_list, test_weights])

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=train_edges,
            valid_edges=valid_edges,
            test_edges=test_edges,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            known_node_ids=[ torch.arange(self.num_nodes) ], 
            format="numpy",
            edge_weight_column=2,
            columns = [0, 2, 1],
            partitioned_evaluation=partitioned_eval,
        )

        return converter.convert()