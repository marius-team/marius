from pathlib import Path

import numpy as np

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file

import torch  # isort:skip


class OGBLCitation2(LinkPredictionDataset):
    """
    Open Graph Benchmark: citation2

    The ogbl-citation2 dataset is a directed graph, representing the citation network between
    a subset of papers extracted from MAG. Each node is a paper with 128-dimensional
    word2vec features that summarizes its title and abstract, and each directed edge
    indicates that one paper cites another. All nodes also come with meta-information
    indicating the year the corresponding paper was published.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbl_citation2"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip"

    def download(self, overwrite=False):
        self.input_train_edges_file = self.output_directory / Path("train.pt")
        self.input_valid_edges_file = self.output_directory / Path("valid.pt")
        self.input_test_edges_file = self.output_directory / Path("test.pt")

        download = False
        if not self.input_train_edges_file.exists():
            download = True
        if not self.input_valid_edges_file.exists():
            download = True
        if not self.input_test_edges_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            for file in (self.output_directory / Path("citation-v2/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        train_idx = torch.load(self.input_train_edges_file)
        valid_idx = torch.load(self.input_valid_edges_file)
        test_idx = torch.load(self.input_test_edges_file)

        train_list = np.array([train_idx.get("source_node"), train_idx.get("target_node")]).T
        valid_list = np.array([valid_idx.get("source_node"), valid_idx.get("target_node")]).T
        test_list = np.array([test_idx.get("source_node"), test_idx.get("target_node")]).T

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=train_list,
            valid_edges=valid_list,
            test_edges=test_list,
            num_partitions=num_partitions,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            remap_ids=remap_ids,
            known_node_ids=[
                torch.arange(2927963)
            ],  # not all nodes appear in the edges, need to supply all node ids for the mapping to be correct
            format="numpy",
            partitioned_evaluation=partitioned_eval,
        )

        return converter.convert()
