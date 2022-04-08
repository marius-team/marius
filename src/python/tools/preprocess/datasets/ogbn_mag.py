from pathlib import Path
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.utils import download_url, extract_file
import numpy as np
import pandas as pd
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.converters.spark_converter import SparkEdgeListConverter
from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.datasets.ogb_helpers import remap_ogbn

from omegaconf import OmegaConf


class OGBNMag(NodeClassificationDataset):

    def __init__(self, output_directory: Path, spark=False):

        super().__init__(output_directory, spark)

        self.dataset_name = "ogbn_mag"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip"

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

            extract_file(self.output_directory / Path("mag/raw/edge.csv.gz"))
            extract_file(self.output_directory / Path("mag/raw/node-feat.csv.gz"))
            extract_file(self.output_directory / Path("mag/raw/node-label.csv.gz"))

            (self.output_directory / Path("mag/raw/edge.csv")).rename(self.input_edge_list_file)
            (self.output_directory / Path("mag/raw/node-feat.csv")).rename(self.input_node_feature_file)
            (self.output_directory / Path("mag/raw/node-label.csv")).rename(self.input_node_label_file)

            for file in (self.output_directory / Path("mag/split/time")).iterdir():
                extract_file(file)

            for file in (self.output_directory / Path("mag/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False):

        train_nodes = np.genfromtxt(self.input_train_nodes_file, delimiter=",").astype(np.int32)
        valid_nodes = np.genfromtxt(self.input_valid_nodes_file, delimiter=",").astype(np.int32)
        test_nodes = np.genfromtxt(self.input_test_nodes_file, delimiter=",").astype(np.int32)

        converter = SparkEdgeListConverter if self.spark else TorchEdgeListConverter
        converter = converter(
            output_dir=self.output_directory,
            train_edges=self.input_edge_list_file,
            num_partitions=num_partitions,
            columns=[0, 1],
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            delim=",",
            known_node_ids=[train_nodes, valid_nodes, test_nodes],
            partitioned_evaluation=partitioned_eval
        )

        dataset_stats = converter.convert()

        features = np.genfromtxt(self.input_node_feature_file, delimiter=",").astype(np.float32)
        labels = np.genfromtxt(self.input_node_label_file, delimiter=",").astype(np.int32)

        if remap_ids:
            node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            train_nodes, valid_nodes, test_nodes, features, labels = remap_ogbn(node_mapping, train_nodes, valid_nodes, test_nodes, features, labels)

        with open(self.output_dir / self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.output_dir / self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.output_dir / self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.output_dir / self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.output_dir / self.node_labels_file, "wb") as f:
            f.write(bytes(labels))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.node_feature_dim = features.shape[1]
        dataset_stats.num_classes = 349

        dataset_stats.num_nodes = dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_test

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats
