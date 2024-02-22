from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes
from marius.tools.preprocess.utils import download_url, extract_file

import torch  # isort:skip


class OGBNPapers100M(NodeClassificationDataset):
    """
    Open Graph Benchmark: ogbn-papers100m

    Directed citation graph of 111 million papers indexed by MAG.
    Its graph structure and node features are constructed in the same way as ogbn-arxiv.
    Among its node set, approximately 1.5 million of them are arXiv papers,
    each of which is manually labeled with one of arXiv’s subject areas.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbn_papers100M"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"

    def download(self, overwrite=False):
        self.input_edge_list_file = self.output_directory / Path("data.npz")  # key: edge_index
        self.input_node_feature_file = self.output_directory / Path("data.npz")  # key: node_feat
        self.input_node_label_file = self.output_directory / Path("node-label.npz")
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

            (self.output_directory / Path("papers100M-bin/raw/data.npz")).rename(self.input_node_feature_file)
            (self.output_directory / Path("papers100M-bin/raw/node-label.npz")).rename(self.input_node_label_file)

            for file in (self.output_directory / Path("papers100M-bin/split/time")).iterdir():
                extract_file(file)

            for file in (self.output_directory / Path("papers100M-bin/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        data_dict = np.load(self.input_edge_list_file)

        input_edges = torch.from_numpy(data_dict["edge_index"].astype(np.int32).transpose())
        train_nodes = np.genfromtxt(self.input_train_nodes_file, delimiter=",").astype(np.int32)
        valid_nodes = np.genfromtxt(self.input_valid_nodes_file, delimiter=",").astype(np.int32)
        test_nodes = np.genfromtxt(self.input_test_nodes_file, delimiter=",").astype(np.int32)

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=input_edges,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            format="pytorch",
            known_node_ids=[train_nodes, valid_nodes, test_nodes],
            partitioned_evaluation=partitioned_eval,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        dataset_stats = converter.convert()

        features = data_dict["node_feat"].astype(np.float32)
        labels = np.load(self.input_node_label_file)["node_label"].astype(np.int32)
        labels[np.isnan(labels)] = -1

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
        dataset_stats.num_classes = 172

        dataset_stats.num_nodes = labels.shape[0]

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats
