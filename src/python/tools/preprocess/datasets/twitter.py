from pathlib import Path
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.converters.spark_converter import SparkEdgeListConverter


class Twitter(LinkPredictionDataset):

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "twitter"
        self.dataset_url = "https://snap.stanford.edu/data/twitter-2010.txt.gz"

    def download(self, overwrite=False):

        self.input_edges = self.output_directory / Path("twitter-2010.txt")

        if not self.input_edges.exists():
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)


    def preprocess(self,
                   num_partitions=1,
                   remap_ids=True,
                   splits=[.9, .05, .05], sequential_train_nodes=False):

        converter = SparkEdgeListConverter if self.spark else TorchEdgeListConverter
        converter = converter(
            output_dir=self.output_directory,
            train_edges=self.input_edges,
            delim=" ",
            columns=[0, 1],
            num_partitions=num_partitions,
            splits=splits,
            remap_ids=remap_ids
        )

        return converter.convert()
