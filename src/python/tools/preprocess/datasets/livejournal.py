from pathlib import Path

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file, strip_header


class Livejournal(LinkPredictionDataset):
    """
    Livejournal

    LiveJournal is a free on-line community with almost 10 million members;
    a significant fraction of these members are highly active.
    (For example, roughly 300,000 update their content in any given 24-hour period.)
    LiveJournal allows members to maintain journals, individual and group blogs,
    and it allows people to declare which other members are their friends they belong.
    4,847,571 nodes, 68,993,773 edges.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "twitter"
        self.dataset_url = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"

    def download(self, overwrite=False):
        self.input_edges = self.output_directory / Path("soc-LiveJournal1.txt")

        if not self.input_edges.exists():
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)
            strip_header(self.input_edges, num_lines=4)

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
            delim="\t",
            src_column=0,
            dst_column=1,
            header_length=0,
            num_partitions=num_partitions,
            splits=splits,
            remap_ids=remap_ids,
            partitioned_evaluation=partitioned_eval,
        )

        return converter.convert()
