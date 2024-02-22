from pathlib import Path

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file, strip_header


class Friendster(LinkPredictionDataset):
    """
    Friendster

    Friendster is an on-line gaming network.
    Before re-launching as a game website, Friendster was a social networking site where
    users can form friendship edge each other. Friendster social network also allows
    users form a group which other members can then join. We consider such user-defined
    groups as ground-truth communities. For the social network, we take the induced subgraph
    of the nodes that either belong to at least one community or are connected to other nodes
    that belong to at least one community. 65,608,366 nodes, 1,806,067,135 edges.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "friendster"
        self.dataset_url = "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"

    def download(self, overwrite=False):
        self.input_edges = self.output_directory / Path("com-friendster.ungraph.txt")

        if not self.input_edges.exists():
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=True)
            strip_header(self.input_edges, num_lines=4)

    def preprocess(
        self,
        num_partitions=1,
        remap_ids=True,
        splits=None,
        sequential_train_nodes=False,
        generate_random_features=False,
        node_feature_dim=32,
        num_classes=50,
        node_splits=[0.1, 0.05, 0.05],
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
