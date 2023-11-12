from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.configuration.marius_config import DatasetConfig


class TorchWriter(object):
    def __init__(self, output_dir, partitioned_evaluation):
        super().__init__()

        self.output_dir = output_dir
        self.partitioned_evaluation = partitioned_evaluation

    def write_to_binary(
        self,
        train_edges_tens,
        valid_edges_tens,
        test_edges_tens,
        num_nodes,
        num_rels,
        num_partitions,
        train_edges_offsets=None,
        valid_edges_offsets=None,
        test_edges_offsets=None,
        edge_weights=None,
    ):
        dataset_stats = DatasetConfig()
        dataset_stats.dataset_dir = Path(self.output_dir).absolute().__str__() + "/"

        dataset_stats.num_edges = train_edges_tens.size(0)
        dataset_stats.num_train = train_edges_tens.size(0)

        if valid_edges_tens is not None:
            dataset_stats.num_valid = valid_edges_tens.size(0)
        if test_edges_tens is not None:
            dataset_stats.num_test = test_edges_tens.size(0)

        dataset_stats.num_nodes = num_nodes
        dataset_stats.num_relations = num_rels

        with open(self.output_dir / Path("dataset.yaml"), "w") as f:
            print("Dataset statistics written to: {}".format((self.output_dir / Path("dataset.yaml")).__str__()))
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        # Read the edge weights
        train_edges_weights, valid_edges_weights, test_edges_weights = None, None, None
        if edge_weights is not None:
            train_edges_weights, valid_edges_weights, test_edges_weights = (
                edge_weights[0],
                edge_weights[1],
                edge_weights[2],
            )

        with open(self.output_dir / Path(PathConstants.train_edges_path), "wb") as f:
            print("Train edges written to:", PathConstants.train_edges_path)
            f.write(bytes(train_edges_tens.numpy()))

        if train_edges_weights is not None:
            train_weights_save_path = self.output_dir / Path(PathConstants.train_edges_weights_path)
            print("Train edges weights written to:", train_weights_save_path)
            train_weights_arr = train_edges_weights.numpy().flatten().astype(np.float32)
            train_weights_arr.tofile(train_weights_save_path)

        if valid_edges_tens is not None:
            print("Valid edges written to:", PathConstants.valid_edges_path)
            with open(self.output_dir / Path(PathConstants.valid_edges_path), "wb") as f:
                f.write(bytes(valid_edges_tens.numpy()))

            if valid_edges_weights is not None:
                valid_weights_save_path = self.output_dir / Path(PathConstants.valid_edges_weights_path)
                print("Valid edges weights written to:", PathConstants.valid_edges_weights_path)
                valid_weights_arr = valid_edges_weights.numpy().flatten().astype(np.float32)
                valid_weights_arr.tofile(valid_weights_save_path)

        if test_edges_tens is not None:
            print("Test edges written to:", PathConstants.test_edges_path)
            with open(self.output_dir / Path(PathConstants.test_edges_path), "wb") as f:
                f.write(bytes(test_edges_tens.numpy()))

            if test_edges_weights is not None:
                test_weights_save_path = self.output_dir / Path(PathConstants.test_edges_weights_path)
                print("Test edge weights written to:", PathConstants.test_edges_weights_path)
                test_weights_arr = test_edges_weights.numpy().flatten().astype(np.float32)
                test_weights_arr.tofile(test_weights_save_path)

        if num_partitions > 1:
            with open(self.output_dir / Path(PathConstants.train_edge_buckets_path), "w") as f:
                print("Train partition offsets written to:", PathConstants.train_edge_buckets_path)
                f.writelines([str(o) + "\n" for o in train_edges_offsets])

            if valid_edges_offsets is not None:
                print("Valid partition offsets written to:", PathConstants.valid_edge_buckets_path)
                with open(self.output_dir / Path(PathConstants.valid_edge_buckets_path), "w") as f:
                    f.writelines([str(o) + "\n" for o in valid_edges_offsets])

            if test_edges_offsets is not None:
                print("Test partition offsets written to:", PathConstants.test_edge_buckets_path)
                with open(self.output_dir / Path(PathConstants.test_edge_buckets_path), "w") as f:
                    f.writelines([str(o) + "\n" for o in test_edges_offsets])

        return dataset_stats
