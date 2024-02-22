import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes
from marius.tools.preprocess.utils import download_url, extract_file

import torch  # isort:skip


class OGBMag240M(NodeClassificationDataset):
    """
    Open Graph Benchmark: mag

    The ogbn-mag dataset is a heterogeneous network composed of a subset of the Microsoft Academic Graph (MAG).
    It contains four types of entities—papers, authors, institutions,
    and fields of study—as well as four types of directed relations connecting two types of entities—an author
    is “affiliated with” an institution, an author “writes” a paper, a paper “cites” a paper,
    and a paper “has a topic of” a field of study. Similar to ogbn-arxiv,
    each paper is associated with a 128-dimensional word2vec feature vector,
    and all the other types of entities are not associated with input node features.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogb_mag240m"
        self.dataset_url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/mag240m_kddcup2021.zip"

    def download(self, overwrite=False):
        self.input_cites_edge_list_file = self.output_directory / Path("cites_edge_index.npy")
        self.input_splits_file = self.output_directory / Path("split_dict.pt")
        self.input_node_feature_file = self.output_directory / Path("node_feat.npy")
        self.input_node_label_file = self.output_directory / Path("node_label.npy")

        download = False
        if not self.input_cites_edge_list_file.exists():
            download = True
        if not self.input_splits_file.exists():
            download = True
        if not self.input_node_feature_file.exists():
            download = True
        if not self.input_node_label_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            (self.output_directory / Path("mag240m_kddcup2021/processed/paper___cites___paper/edge_index.npy")).rename(
                self.input_cites_edge_list_file
            )
            (self.output_directory / Path("mag240m_kddcup2021/split_dict.pt")).rename(self.input_splits_file)
            (self.output_directory / Path("mag240m_kddcup2021/processed/paper/node_feat.npy")).rename(
                self.input_node_feature_file
            )
            (self.output_directory / Path("mag240m_kddcup2021/processed/paper/node_label.npy")).rename(
                self.input_node_label_file
            )

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        citation_edges = np.load(self.input_cites_edge_list_file).astype(np.int32).transpose()

        split_dict = torch.load(self.input_splits_file)

        train_nodes = split_dict["train"].astype(np.int32)
        valid_nodes = split_dict["valid"].astype(np.int32)
        # test_nodes = split_dict['test'].astype(np.int32)
        test_nodes = valid_nodes

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=citation_edges,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            format="numpy",
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            known_node_ids=[
                train_nodes,
                valid_nodes,
                test_nodes,
                np.arange(121751666, dtype=np.int32),
            ],  # not all nodes appear in the edges
            num_nodes=121751666,
            num_rels=1,
            partitioned_evaluation=partitioned_eval,
        )

        dataset_stats = converter.convert()

        features = np.load(self.input_node_feature_file)
        labels = np.load(self.input_node_label_file)
        labels[np.isnan(labels)] = -1
        labels = labels.astype(np.int32)

        if remap_ids:
            node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            train_nodes, valid_nodes, test_nodes, features, labels = remap_nodes(
                node_mapping, train_nodes, valid_nodes, test_nodes, features, labels
            )

        # convert to float32 in chunks, tested on ~500 GB RAM, need at least ~375GB minimum for float32 features
        num_nodes = features.shape[0]
        feat_dim = features.shape[1]
        np.save(self.output_directory / Path("temp.npy"), features)
        features = np.zeros((num_nodes, feat_dim), np.float32)
        chunk_size = int(2e7)
        start = 0
        while start < num_nodes:
            float16_features = np.load(self.output_directory / Path("temp.npy"), mmap_mode="r")[
                start : start + chunk_size
            ]
            features[start : start + chunk_size] = float16_features.astype(np.float32)
            start += chunk_size
        os.remove(self.output_directory / Path("temp.npy"))

        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            chunk_size = int(1e7)
            start = 0
            while start < num_nodes:
                f.write(bytes(features[start : start + chunk_size]))
                start += chunk_size
        with open(self.node_labels_file, "wb") as f:
            f.write(bytes(labels))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.feature_dim = features.shape[1]
        dataset_stats.num_classes = 153

        dataset_stats.num_nodes = labels.shape[0]

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats
