from pathlib import Path
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file
import numpy as np
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.converters.spark_converter import SparkEdgeListConverter
import torch


class OGBLWikiKG2(LinkPredictionDataset):

    def __init__(self, output_directory: Path, spark=False):

        super().__init__(output_directory, spark)

        self.dataset_name = "ogbl_wikikg2"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/linkproppred/wikikg-v2.zip"

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

            for file in (self.output_directory / Path("wikikg-v2/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False):
        train_idx = torch.load(self.input_train_edges_file)
        valid_idx = torch.load(self.input_valid_edges_file)
        test_idx = torch.load(self.input_test_edges_file)

        train_list = np.array([train_idx.get("head"),
                               train_idx.get("relation"),
                               train_idx.get("tail")]).T
        valid_list = np.array([valid_idx.get("head"),
                               valid_idx.get("relation"),
                               valid_idx.get("tail")]).T
        test_list = np.array([test_idx.get("head"),
                              test_idx.get("relation"),
                              test_idx.get("tail")]).T

        converter = SparkEdgeListConverter if self.spark else TorchEdgeListConverter
        converter = converter(
            output_dir=self.output_directory,
            train_edges=train_list.astype("int32"),
            valid_edges=valid_list.astype("int32"),
            test_edges=test_list.astype("int32"),
            num_partitions=num_partitions,
            format="numpy",
            remap_ids=remap_ids,
            partitioned_evaluation=partitioned_eval
        )

        return converter.convert()
