from pathlib import Path

from marius.tools.preprocess.converters.spark_converter import SparkEdgeListConverter
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset


class CustomLinkPredictionDataset(LinkPredictionDataset):
    def __init__(
        self, output_directory: Path, files: list, delim: str = "\t", dataset_name: str = "custom", spark: bool = False
    ):
        super().__init__(output_directory, spark)

        self.dataset_name = dataset_name
        self.output_directory = output_directory

        if len(files) == 1:
            self.train_edges_file = files[0]
            self.valid_edges_file = None
            self.test_edges_file = None

        if len(files) == 3:
            self.train_edges_file = files[0]
            self.valid_edges_file = files[1]
            self.test_edges_file = files[2]

        self.delim = delim
        self.spark = spark

    def download(self, overwrite=False):
        pass

    def preprocess(
        self,
        num_partitions=1,
        remap_ids=True,
        splits=[0.9, 0.05, 0.05],
        partitioned_eval=False,
        sequential_train_nodes=False,
        columns=[0, 1, 2],
    ):
        converter = SparkEdgeListConverter if self.spark else TorchEdgeListConverter
        converter = converter(
            output_dir=self.output_directory,
            train_edges=self.train_edges_file,
            valid_edges=self.valid_edges_file,
            test_edges=self.test_edges_file,
            delim=self.delim,
            columns=columns,
            num_partitions=num_partitions,
            splits=splits,
            remap_ids=remap_ids,
            partitioned_evaluation=partitioned_eval,
        )

        converter.convert()
