import glob
import os
import re
import sys
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.configuration.marius_config import DatasetConfig
from marius.tools.preprocess.converters.spark_constants import (
    DST_EDGE_BUCKET_COL,
    EDGES_INDEX_COL,
    INDEX_COL,
    REL_INDEX_COL,
    SRC_EDGE_BUCKET_COL,
    TMP_DATA_DIRECTORY,
)
from marius.tools.preprocess.utils import get_df_count


# TODO can this be made faster? Pandas is pretty slow and not parallel
def convert_to_binary(input_filename, output_filename):
    assert input_filename != output_filename
    with open(output_filename, "wb") as output_file:
        for chunk in pd.read_csv(input_filename, header=None, chunksize=10**8, sep="\t", dtype=int):
            chunk_array = chunk.to_numpy(dtype=np.int32)
            output_file.write(bytes(chunk_array))

    os.system("rm {}".format(input_filename))


# TODO we can make this faster by using the cat bash command to combine these files super fast
def merge_csvs(input_directory, output_file):
    all_csvs = []
    for filename in glob.iglob(input_directory + "/**/*.csv", recursive=True):
        all_csvs.append(filename)

    print("Merging CSVs from {} to {}".format(input_directory, output_file))
    os.system("rm -rf {}".format(output_file))
    for source_file in all_csvs:
        os.system("cat {} >> {}".format(source_file, output_file))

    os.system("rm -rf {}".format(input_directory))


def write_df_to_csv(df, output_filename):
    tmp_dir = TMP_DATA_DIRECTORY + str(randint(0, sys.maxsize))
    df.write.csv(tmp_dir, mode="overwrite", sep="\t")
    merge_csvs(tmp_dir, output_filename)


def write_partitioned_df_to_csv(partition_triples, num_partitions, output_filename):
    bucket_counts = partition_triples.groupBy([SRC_EDGE_BUCKET_COL, DST_EDGE_BUCKET_COL]).count()

    print(partition_triples.rdd.getNumPartitions())

    # for edges, the order needs to be maintained. all edges that belong to bucket [i, j]
    # should appear before [i, j+1] and that of [i, j+1] should appear before [i+1, j].
    # repartitionByRange makes sure that all edges belonging to src bucket i, fall in the
    # same partition. Also, this function will output at most `num_partitions` partitions.
    partition_triples.repartitionByRange(num_partitions, SRC_EDGE_BUCKET_COL).sortWithinPartitions(
        SRC_EDGE_BUCKET_COL, DST_EDGE_BUCKET_COL
    ).drop(DST_EDGE_BUCKET_COL, SRC_EDGE_BUCKET_COL).write.csv(
        TMP_DATA_DIRECTORY + "_edges", mode="overwrite", sep="\t"
    )

    # for partition offset counts, the ordering of dst_buckets does not matter since we
    # read the value before setting the offset in line number 92.
    # dst_buckets = counts.iloc[:, 0].values.
    # we make use of partitionBy to parallelize writes.
    bucket_counts.write.partitionBy(SRC_EDGE_BUCKET_COL).csv(TMP_DATA_DIRECTORY + "_counts", mode="overwrite", sep="\t")

    partition_offsets = []

    os.system("rm -rf {}".format(output_filename))
    for i in range(num_partitions):
        # looks like there is no way in glob to restrict to the pattern [0]*{i}- alone.
        # it matches things like part-00004-sdfvf0-sdf.csv when given part-[0]*0-*.csv
        tmp_edges_files = glob.glob("{}/part-[0]*{}-*.csv".format(TMP_DATA_DIRECTORY + "_edges", str(i)))

        tmp_counts_files = glob.glob(
            "{}/{}={}/*.csv".format(TMP_DATA_DIRECTORY + "_counts", SRC_EDGE_BUCKET_COL, str(i))
        )

        edges_bucket_counts = np.zeros(num_partitions, dtype=np.int)
        edge_file_pattern = re.compile(r"{}/part-[0]*{}-.*\.csv".format(TMP_DATA_DIRECTORY + "_edges", str(i)))
        for tmp_edges_file in tmp_edges_files:
            if edge_file_pattern.match(tmp_edges_file):
                os.system("cat {} >> {}".format(tmp_edges_file, output_filename))

        for tmp_counts_file in tmp_counts_files:
            counts = pd.read_csv(tmp_counts_file, sep="\t", header=None)

            dst_buckets = counts.iloc[:, 0].values
            dst_counts = counts.iloc[:, 1].values

            edges_bucket_counts[dst_buckets] = dst_counts

        partition_offsets.append(edges_bucket_counts)

    os.system("rm -rf {}".format(TMP_DATA_DIRECTORY + "_edges"))
    os.system("rm -rf {}".format(TMP_DATA_DIRECTORY + "_counts"))

    return np.concatenate(partition_offsets)


class SparkWriter(object):
    def __init__(self, spark, output_dir, partitioned_evaluation):
        super().__init__()

        self.spark = spark
        self.output_dir = output_dir
        self.partitioned_evaluation = partitioned_evaluation

    def write_to_csv(self, train_edges_df, valid_edges_df, test_edges_df, nodes_df, rels_df, num_partitions):
        dataset_stats = DatasetConfig()
        dataset_stats.dataset_dir = Path(self.output_dir).absolute().__str__()

        dataset_stats.num_edges = get_df_count(train_edges_df, EDGES_INDEX_COL)
        train_edges_df = train_edges_df.drop(EDGES_INDEX_COL)
        dataset_stats.num_train = dataset_stats.num_edges

        if valid_edges_df is not None:
            dataset_stats.num_valid = get_df_count(valid_edges_df, EDGES_INDEX_COL)
            valid_edges_df = valid_edges_df.drop(EDGES_INDEX_COL)
        if test_edges_df is not None:
            dataset_stats.num_test = get_df_count(test_edges_df, EDGES_INDEX_COL)
            test_edges_df = test_edges_df.drop(EDGES_INDEX_COL)

        dataset_stats.num_nodes = get_df_count(nodes_df, INDEX_COL)

        if rels_df is None:
            dataset_stats.num_relations = 1
        else:
            dataset_stats.num_relations = get_df_count(rels_df, REL_INDEX_COL)

        with open(self.output_dir / Path("dataset.yaml"), "w") as f:
            print("Dataset statistics written to: {}".format((self.output_dir / Path("dataset.yaml")).__str__()))
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        write_df_to_csv(nodes_df, self.output_dir / Path(PathConstants.node_mapping_path))

        if rels_df is not None:
            write_df_to_csv(rels_df, self.output_dir / Path(PathConstants.relation_mapping_path))

        if num_partitions > 1:
            offsets = write_partitioned_df_to_csv(
                train_edges_df, num_partitions, self.output_dir / Path(PathConstants.train_edges_path)
            )

            with open(self.output_dir / Path(PathConstants.train_edge_buckets_path), "w") as f:
                f.writelines([str(o) + "\n" for o in offsets])

            if self.partitioned_evaluation:
                if valid_edges_df is not None:
                    offsets = write_partitioned_df_to_csv(
                        valid_edges_df, num_partitions, self.output_dir / Path(PathConstants.valid_edges_path)
                    )

                    with open(self.output_dir / Path(PathConstants.valid_edge_buckets_path), "w") as f:
                        f.writelines([str(o) + "\n" for o in offsets])

                if test_edges_df is not None:
                    offsets = write_partitioned_df_to_csv(
                        test_edges_df, num_partitions, self.output_dir / Path(PathConstants.test_edges_path)
                    )
                    with open(self.output_dir / Path(PathConstants.test_edge_buckets_path), "w") as f:
                        f.writelines([str(o) + "\n" for o in offsets])

            else:
                if valid_edges_df is not None:
                    write_df_to_csv(valid_edges_df, self.output_dir / Path(PathConstants.valid_edges_path))

                if test_edges_df is not None:
                    write_df_to_csv(test_edges_df, self.output_dir / Path(PathConstants.test_edges_path))

        else:
            write_df_to_csv(train_edges_df, self.output_dir / Path(PathConstants.train_edges_path))

            if valid_edges_df is not None:
                write_df_to_csv(valid_edges_df, self.output_dir / Path(PathConstants.valid_edges_path))

            if test_edges_df is not None:
                write_df_to_csv(test_edges_df, self.output_dir / Path(PathConstants.test_edges_path))

        return dataset_stats

    def write_to_binary(self, train_edges_df, valid_edges_df, test_edges_df, nodes_df, rels_df, num_partitions):
        print("Writing to CSV")
        dataset_stats = self.write_to_csv(
            train_edges_df, valid_edges_df, test_edges_df, nodes_df, rels_df, num_partitions
        )

        train_file = self.output_dir / Path(PathConstants.train_edges_path)
        valid_file = self.output_dir / Path(PathConstants.valid_edges_path)
        test_file = self.output_dir / Path(PathConstants.test_edges_path)

        tmp_train_file = TMP_DATA_DIRECTORY + "tmp_train_edges.tmp"
        tmp_valid_file = TMP_DATA_DIRECTORY + "tmp_valid_edges.tmp"
        tmp_test_file = TMP_DATA_DIRECTORY + "tmp_test_edges.tmp"

        print("Converting to binary")
        os.rename(train_file, tmp_train_file)
        convert_to_binary(tmp_train_file, train_file)

        if valid_edges_df is not None:
            os.rename(valid_file, tmp_valid_file)
            convert_to_binary(tmp_valid_file, valid_file)

        if test_edges_df is not None:
            os.rename(test_file, tmp_test_file)
            convert_to_binary(tmp_test_file, test_file)

        return dataset_stats
