from pathlib import Path

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file


class Freebase86m(LinkPredictionDataset):
    """
    Freebase

    The full Freebase dataset. 86054151 nodes, 338586276 edges, 14824 relations.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "freebase86m"
        self.dataset_url = "https://data.dgl.ai/dataset/Freebase.zip"

    def download(self, overwrite=False):
        self.input_train_edges_file = self.output_directory / Path("train.txt")
        self.input_valid_edges_file = self.output_directory / Path("valid.txt")
        self.input_test_edges_file = self.output_directory / Path("test.txt")

        download = False
        if not self.input_train_edges_file.exists():
            download = True
        if not self.input_valid_edges_file.exists():
            download = True
        if not self.input_test_edges_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)

            for file in (self.output_directory / Path("Freebase")).iterdir():
                file.rename(self.output_directory / Path(file.name))

            (self.output_directory / Path("Freebase")).rmdir()

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_train_edges_file,
            valid_edges=self.input_valid_edges_file,
            test_edges=self.input_test_edges_file,
            num_partitions=num_partitions,
            src_column=0,
            dst_column=1,
            edge_type_column=2,
            remap_ids=remap_ids,
            partitioned_evaluation=partitioned_eval,
        )

        return converter.convert()
