from pathlib import Path

from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file


class FB15K(LinkPredictionDataset):
    """
    Freebase 15k

    The FB15k dataset contains knowledge base relation triples and textual
    mentions of Freebase entity pairs. It has a total of 592,213 triplets
    with 14,951 entities and 1,345 relationships.
    """

    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "fb15k"
        self.dataset_url = "https://dl.fbaipublicfiles.com/starspace/fb15k.tgz"

    def download(self, overwrite=False):
        self.input_train_edges_file = self.output_directory / Path("freebase_mtr100_mte100-train.txt")
        self.input_valid_edges_file = self.output_directory / Path("freebase_mtr100_mte100-valid.txt")
        self.input_test_edges_file = self.output_directory / Path("freebase_mtr100_mte100-test.txt")

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

            for file in (self.output_directory / Path("FB15k")).iterdir():
                file.rename(self.output_directory / Path(file.name))

            (self.output_directory / Path("FB15k")).rmdir()

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_train_edges_file,
            valid_edges=self.input_valid_edges_file,
            test_edges=self.input_test_edges_file,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            src_column=0,
            dst_column=2,
            edge_type_column=1,
            partitioned_evaluation=partitioned_eval,
        )

        return converter.convert()
