import os
from abc import ABC, abstractmethod
from pathlib import Path

from marius.tools.configuration.constants import PathConstants
from marius.tools.configuration.marius_config import DatasetConfig


class Dataset(ABC):
    """
    Abstract dataset class
    """

    edge_list_file: Path
    edge_features_file: Path

    node_mapping_file: Path
    node_features_file: Path

    relation_mapping_file: Path
    relation_features_file: Path

    node_type_file: Path
    node_features_file: Path

    dataset_name: str
    dataset_url: str
    output_directory: Path

    spark: bool

    def __init__(self, output_directory, spark=False):
        self.output_directory = output_directory
        self.spark = spark
        os.makedirs(self.output_directory / Path(PathConstants.edges_directory), exist_ok=True)
        os.makedirs(self.output_directory / Path(PathConstants.nodes_directory), exist_ok=True)

        self.edge_list_file = self.output_directory / Path(PathConstants.train_edges_path)
        self.edge_buckets_file = self.output_directory / Path(PathConstants.train_edge_buckets_path)

        self.node_features_file = self.output_directory / Path(PathConstants.node_features_path)
        self.relation_features_file = self.output_directory / Path(PathConstants.relation_features_path)

    @abstractmethod
    def download(self, overwrite=False):
        pass

    @abstractmethod
    def preprocess(self) -> DatasetConfig:
        pass


class NodeClassificationDataset(Dataset):
    def __init__(self, output_directory, spark):
        super().__init__(output_directory, spark)

        self.train_nodes_file = output_directory / Path(PathConstants.train_nodes_path)
        self.valid_nodes_file = output_directory / Path(PathConstants.valid_nodes_path)
        self.test_nodes_file = output_directory / Path(PathConstants.test_nodes_path)

        self.node_labels_file = output_directory / Path(PathConstants.labels_path)


class LinkPredictionDataset(Dataset):
    def __init__(self, output_directory, spark):
        super().__init__(output_directory, spark)

        self.train_edges_file = output_directory / Path(PathConstants.train_edges_path)
        self.train_edge_buckets_file = self.output_directory / Path(PathConstants.train_edge_buckets_path)

        self.valid_edges_file = output_directory / Path(PathConstants.valid_edges_path)
        self.valid_edge_buckets_file = self.output_directory / Path(PathConstants.valid_edge_buckets_path)

        self.test_edges_file = output_directory / Path(PathConstants.test_edges_path)
        self.test_edge_buckets_file = self.output_directory / Path(PathConstants.test_edge_buckets_path)


class GraphClassificationDataset(Dataset):
    pass
