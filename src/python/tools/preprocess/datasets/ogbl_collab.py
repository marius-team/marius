from pathlib import Path

import pandas as pd

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
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

    def __init__(self, output_directory: Path, spark=False, include_edge_type=True, include_edge_weight=True):
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
        df = pd.read_csv(nodes_path, compression="gzip", header=None)
        self.num_nodes = df.iloc[0][0]

    def preprocess(
        self,
        num_partitions=1,
        remap_ids=True,
        splits=None,
        sequential_train_nodes=False,
        partitioned_eval=False,
    ):
        # Read in the training data
        train_idx = torch.load(self.input_train_edges_file)
        train_edges = torch.from_numpy(train_idx.get("edge"))

        # Read in the valid data
        valid_idx = torch.load(self.input_valid_edges_file)
        valid_edges = torch.from_numpy(valid_idx.get("edge"))

        # Read in the test data
        test_idx = torch.load(self.input_test_edges_file)
        test_edges = torch.from_numpy(test_idx.get("edge"))

        edge_type_column, edge_weight_column = None, None
        if self.include_edge_type:
            # Added in the year information
            train_year = torch.from_numpy(train_idx.get("year").reshape(-1, 1))
            train_edges = torch.cat((train_edges, train_year), dim=1)

            valid_year = torch.from_numpy(valid_idx.get("year").reshape(-1, 1))
            valid_edges = torch.cat((valid_edges, valid_year), dim=1)

            test_year = torch.from_numpy(test_idx.get("year").reshape(-1, 1))
            test_edges = torch.cat((test_edges, test_year), dim=1)

            edge_type_column = 2

        if self.include_edge_weight:
            # Add in the weights
            train_weight = torch.from_numpy(train_idx.get("weight").reshape(-1, 1))
            train_edges = torch.cat((train_edges, train_weight), dim=1)

            valid_weight = torch.from_numpy(valid_idx.get("weight").reshape(-1, 1))
            valid_edges = torch.cat((valid_edges, valid_weight), dim=1)

            test_weight = torch.from_numpy(test_idx.get("weight").reshape(-1, 1))
            test_edges = torch.cat((test_edges, test_weight), dim=1)

            edge_weight_column = 3

        # Add in the edge type information
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=train_edges,
            valid_edges=valid_edges,
            test_edges=test_edges,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            known_node_ids=[torch.arange(self.num_nodes)],
            format="pytorch",
            splits=splits,
            sequential_train_nodes=sequential_train_nodes,
            src_column=0,
            dst_column=1,
            edge_type_column=edge_type_column,
            edge_weight_column=edge_weight_column,
            partitioned_evaluation=partitioned_eval,
        )

        converter.convert()
