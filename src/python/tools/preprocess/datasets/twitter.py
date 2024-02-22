from pathlib import Path

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file


class Twitter(LinkPredictionDataset):
    """
    Twitter

    467 million Twitter posts from 20 million users covering a 7 month period from
    June 1 2009 to December 31 2009. Estimated 20-30% of all public tweets published
    on Twitter during the particular time frame. For each public tweet the following
    information is available: Author, Time, Content
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "twitter"
        self.dataset_url = "https://snap.stanford.edu/data/twitter-2010.txt.gz"

    def download(self, overwrite=False):
        self.input_edges = self.output_directory / Path("twitter-2010.txt")

        if not self.input_edges.exists():
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)

    def preprocess(
        self,
        num_partitions=1,
        remap_ids=True,
        splits=[0.9, 0.05, 0.05],
        sequential_train_nodes=False,
        partitioned_eval=False,
    ):
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_edges,
            delim=" ",
            src_column=0,
            dst_column=1,
            num_partitions=num_partitions,
            splits=splits,
            remap_ids=remap_ids,
            partitioned_evaluation=partitioned_eval,
        )

        return converter.convert()
