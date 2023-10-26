from pathlib import Path
import sys

import numpy as np
import os
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

    def __init__(self, output_directory: Path, spark=False, include_edge_type = False, include_edge_weight = False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbl_citation2"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/linkproppred/collab.zip"
        self.node_ids = None
        self.include_edge_type = include_edge_type
        self.include_edge_weight = include_edge_weight

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
        
        for edge_type in [False, True]:
            for edge_weight in [False, True]:
                self.include_edge_type = edge_type
                self.include_edge_weight = edge_weight
                save_dir = os.path.join(self.output_directory, "type_" + str(self.include_edge_type) + "_weight_" + str(self.include_edge_weight))
                os.makedirs(save_dir, exist_ok = True)

                # Read in the training data
                train_idx = torch.load(self.input_train_edges_file)
                train_edges = train_idx.get("edge")

                # Read in the valid data
                valid_idx = torch.load(self.input_valid_edges_file)
                valid_edges = valid_idx.get("edge")

                # Read in the test data
                test_idx = torch.load(self.input_test_edges_file)
                test_edges = test_idx.get("edge")

                weights_col_id = -1
                col_ids = [0, 1]
                if self.include_edge_weight:
                    # Add in the weights
                    train_weights = train_idx.get("weight").reshape(-1, 1)
                    train_edges = np.hstack([train_edges, train_weights])

                    valid_weights = valid_idx.get("weight").reshape(-1, 1)
                    valid_edges = np.hstack([valid_edges, valid_weights])

                    test_weights = test_idx.get("weight").reshape(-1, 1)
                    test_edges = np.hstack([test_edges, test_weights])
                    
                    weights_col_id = 2
                    col_ids.insert(1, weights_col_id)

                if self.include_edge_type:
                    # Added in the year information
                    train_edges = np.hstack([train_edges, train_idx.get("year").reshape(-1, 1)])
                    valid_edges = np.hstack([valid_edges, valid_idx.get("year").reshape(-1, 1)])
                    test_edges = np.hstack([test_edges, test_idx.get("year").reshape(-1, 1)])

                    # Normalize the edge types
                    min_year = min([np.min(train_edges[ : , -1]), np.min(valid_edges[ : , -1]), np.min(test_edges[ : , -1])])
                    train_edges[ : , -1] = train_edges[ : , -1] - min_year
                    valid_edges[ : , -1] = valid_edges[ : , -1] - min_year
                    test_edges[ : , -1] = test_edges[ : , -1] - min_year

                    # Added in edge type column id
                    last_col_id = train_edges.shape[1] - 1
                    if len(col_ids) == 3:
                        col_ids[1] = last_col_id
                    else:
                        col_ids.insert(1, last_col_id)        
                
                # Add in the edge type information
                converter = TorchEdgeListConverter(
                    output_dir = save_dir,
                    train_edges = train_edges,
                    valid_edges = valid_edges,
                    test_edges = test_edges,
                    num_partitions = num_partitions,
                    remap_ids = remap_ids,
                    known_node_ids = [ torch.arange(self.num_nodes) ], 
                    format = "numpy",
                    edge_weight_column = weights_col_id,
                    columns = col_ids,
                    partitioned_evaluation=partitioned_eval,
                )

                converter.convert()