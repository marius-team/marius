from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file


class OGBWikiKG90Mv2(LinkPredictionDataset):
    """
    Open Graph Benchmark: wikikg2

    The ogbl-wikikg2 dataset is a Knowledge Graph (KG) extracted from the Wikidata knowledge base.
    It contains a set of triplet edges (head, relation, tail),
    capturing the different types of relations between entities in the world, e.g.,
    (Canada, citizen, Hinton). We retrieve all the relational statements in Wikidata and filter out rare entities.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogb_wikikg90mv2"
        self.dataset_url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/wikikg90m-v2.zip"

    def download(self, overwrite=False):
        self.input_train_edges_file = self.output_directory / Path("train_hrt.npy")
        self.input_valid_edges_sr_file = self.output_directory / Path("val_hr.npy")
        self.input_valid_edges_d_file = self.output_directory / Path("val_t.npy")
        # self.input_test_edges_file = self.output_directory / Path("test-dev_hr.npy")
        # self.input_test_edges_file = self.output_directory / Path("test-challenge_hr.npy")

        self.input_node_feature_file = self.output_directory / Path("entity_feat.npy")
        self.input_rel_feature_file = self.output_directory / Path("relation_feat.npy")

        download = False
        if not self.input_train_edges_file.exists():
            download = True
        if not self.input_valid_edges_sr_file.exists():
            download = True
        if not self.input_valid_edges_d_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)

            for file in (self.output_directory / Path("wikikg90m-v2/processed/")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        train_edges = np.load(self.input_train_edges_file).astype(np.int32)
        valid_edges_sr = np.load(self.input_valid_edges_sr_file)
        valid_edges_d = np.load(self.input_valid_edges_d_file)

        valid_edges = np.concatenate((valid_edges_sr, np.reshape(valid_edges_d, (-1, 1))), axis=1).astype(np.int32)

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=train_edges,
            valid_edges=valid_edges,
            test_edges=valid_edges,
            num_partitions=num_partitions,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            format="numpy",
            partitioned_evaluation=partitioned_eval,
        )

        dataset_stats = converter.convert()

        node_features = np.load(self.input_node_feature_file).astype(np.float32)
        rel_features = np.load(self.input_rel_feature_file).astype(np.float32)

        if remap_ids:
            node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            random_node_map = node_mapping[:, 1].astype(np.int32)
            random_node_map_argsort = np.argsort(random_node_map)

            with open(self.node_features_file, "wb") as f:
                chunk_size = 1e7
                num_chunks = np.ceil(node_mapping.shape[0] / chunk_size)

                offset = 0

                for chunk_id in range(num_chunks):
                    if offset + chunk_size >= node_mapping.shape[0]:
                        chunk_size = node_mapping.shape[0] - offset
                    f.write(bytes(node_features[random_node_map_argsort[offset : offset + chunk_size]]))

            rel_mapping = np.genfromtxt(
                self.output_directory / Path(PathConstants.relation_mapping_path), delimiter=","
            )
            random_rel_map = rel_mapping[:, 1].astype(np.int32)
            random_rel_map_argsort = np.argsort(random_rel_map)
            rel_features = rel_features[random_rel_map_argsort]
        else:
            with open(self.node_features_file, "wb") as f:
                f.write(bytes(node_features))

        with open(self.relation_features_file, "wb") as f:
            f.write(bytes(rel_features))

        # update dataset yaml
        dataset_stats.node_feature_dim = node_features.shape[1]
        dataset_stats.rel_feature_dim = rel_features.shape[1]

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats
