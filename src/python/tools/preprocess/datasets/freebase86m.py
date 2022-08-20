from pathlib import Path
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter


class Freebase86m(LinkPredictionDataset):

    def __init__(self, output_directory: Path):
        super().__init__(output_directory)

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

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False):
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_train_edges_file,
            valid_edges=self.input_valid_edges_file,
            test_edges=self.input_test_edges_file,
            num_partitions=num_partitions,
            columns=[0, 2, 1],
            remap_ids=remap_ids
        )

        return converter.convert()
