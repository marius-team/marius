from pathlib import Path

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file

import torch  # isort:skip


class OGBLPpa(LinkPredictionDataset):
    """
    Open Graph Benchmark: ppa

    The ogbl-ppa dataset is an undirected, unweighted graph.
    Nodes represent proteins from 58 different species, and edges indicate biologically meaningful
    associations between proteins, e.g., physical interactions, co-expression, homology or genomic neighborhood.
    Each node contains a 58-dimensional one-hot feature vector that indicates the species that
    the corresponding protein comes from.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "ogbl_ppa"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/linkproppred/ppassoc.zip"

    def download(self, overwrite=False, remap_ids=True):
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

            for file in (self.output_directory / Path("ppassoc/split/throughput")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        train_idx = torch.load(self.input_train_edges_file).get("edge")
        valid_idx = torch.load(self.input_valid_edges_file).get("edge")
        test_idx = torch.load(self.input_test_edges_file).get("edge")

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=train_idx,
            valid_edges=valid_idx,
            test_edges=test_idx,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            format="numpy",
            partitioned_evaluation=partitioned_eval,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
        )

        return converter.convert()
