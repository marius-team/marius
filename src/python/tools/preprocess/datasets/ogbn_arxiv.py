from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes
from marius.tools.preprocess.utils import download_url, extract_file


class OGBNArxiv(NodeClassificationDataset):
    """
    Open Graph Benchmark: arxiv

    The ogbn-arxiv dataset is a directed graph,
    representing the citation network between all Computer Science (CS) arXiv papers indexed by MAG.
    Each node is an arXiv paper and each directed edge indicates that one paper cites another one.
    Each paper comes with a 128-dimensional feature vector obtained by averaging the embeddings of words
    in its title and abstract.
    The embeddings of individual words are computed by running the skip-gram model over the MAG corpus.
    We also provide the mapping from MAG paper IDs into the raw texts of titles and abstracts here.
    In addition, all papers are also associated with the year that the corresponding paper was published.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbn_arxiv"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"

    def download(self, overwrite=False):
        self.input_edge_list_file = self.output_directory / Path("edge.csv")
        self.input_node_feature_file = self.output_directory / Path("node-feat.csv")
        self.input_node_label_file = self.output_directory / Path("node-label.csv")
        self.input_train_nodes_file = self.output_directory / Path("train.csv")
        self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
        self.input_test_nodes_file = self.output_directory / Path("test.csv")

        download = False
        if not self.input_edge_list_file.exists():
            download = True
        if not self.input_node_feature_file.exists():
            download = True
        if not self.input_node_label_file.exists():
            download = True
        if not self.input_train_nodes_file.exists():
            download = True
        if not self.input_valid_nodes_file.exists():
            download = True
        if not self.input_test_nodes_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            extract_file(self.output_directory / Path("arxiv/raw/edge.csv.gz"))
            extract_file(self.output_directory / Path("arxiv/raw/node-feat.csv.gz"))
            extract_file(self.output_directory / Path("arxiv/raw/node-label.csv.gz"))

            (self.output_directory / Path("arxiv/raw/edge.csv")).rename(self.input_edge_list_file)
            (self.output_directory / Path("arxiv/raw/node-feat.csv")).rename(self.input_node_feature_file)
            (self.output_directory / Path("arxiv/raw/node-label.csv")).rename(self.input_node_label_file)

            for file in (self.output_directory / Path("arxiv/split/time")).iterdir():
                extract_file(file)

            for file in (self.output_directory / Path("arxiv/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        train_nodes = np.genfromtxt(self.input_train_nodes_file, delimiter=",").astype(np.int32)
        valid_nodes = np.genfromtxt(self.input_valid_nodes_file, delimiter=",").astype(np.int32)
        test_nodes = np.genfromtxt(self.input_test_nodes_file, delimiter=",").astype(np.int32)

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_edge_list_file,
            num_partitions=num_partitions,
            src_column=0,
            dst_column=1,
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            delim=",",
            known_node_ids=[train_nodes, valid_nodes, test_nodes],
            partitioned_evaluation=partitioned_eval,
        )
        dataset_stats = converter.convert()

        features = np.genfromtxt(self.input_node_feature_file, delimiter=",").astype(np.float32)
        labels = np.genfromtxt(self.input_node_label_file, delimiter=",").astype(np.int32)

        if remap_ids:
            node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            train_nodes, valid_nodes, test_nodes, features, labels = remap_nodes(
                node_mapping, train_nodes, valid_nodes, test_nodes, features, labels
            )

        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.node_labels_file, "wb") as f:
            f.write(bytes(labels))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.node_feature_dim = features.shape[1]
        dataset_stats.num_classes = 40

        dataset_stats.num_nodes = dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_test

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats
