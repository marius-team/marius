import glob
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, rand, row_number
from pyspark.sql.window import Window

from marius.tools.preprocess.converters.partitioners.spark_partitioner import SparkPartitioner
from marius.tools.preprocess.converters.readers.spark_readers import SparkDelimitedFileReader
from marius.tools.preprocess.converters.spark_constants import (
    DST_COL,
    EDGES_INDEX_COL,
    INDEX_COL,
    NODE_LABEL,
    REL_COL,
    REL_INDEX_COL,
    RELATION_LABEL,
    SPARK_APP_NAME,
    SRC_COL,
    TMP_DATA_DIRECTORY,
)
from marius.tools.preprocess.converters.writers.spark_writer import SparkWriter

SUPPORTED_DELIM_FORMATS = ["CSV", "TSV", "TXT", "DELIM", "DELIMITED"]
SUPPORTED_NON_DELIM_FILE_FORMATS = ["PARQUET"]


def remap_columns(df, has_rels):
    columns = [SRC_COL, REL_COL, DST_COL]
    if not has_rels:
        columns = [SRC_COL, DST_COL]
    return df.toDF(*columns)


def get_nodes_df(edges_df):
    nodes = (
        edges_df.select(col(SRC_COL).alias(NODE_LABEL))
        .union(edges_df.select(col(DST_COL).alias(NODE_LABEL)))
        .distinct()
        .repartition(1)
        .orderBy(rand())
        .cache()
    )
    nodes = assign_ids(nodes, INDEX_COL)
    return nodes


def get_relations_df(edges_df):
    rels = (
        edges_df.drop(SRC_COL, DST_COL)
        .distinct()
        .repartition(1)
        .orderBy(rand())
        .withColumnRenamed(REL_COL, RELATION_LABEL)
        .cache()
    )
    rels = assign_ids(rels, REL_INDEX_COL)
    return rels


def assign_ids(df, col_id):
    if df is None:
        return None
    return df.withColumn(col_id, row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)


def remap_edges(edges_df, nodes, rels):
    if rels is not None:
        remapped_edges_df = (
            edges_df.join(nodes.hint("merge"), edges_df.src == nodes.node_label)
            .drop(NODE_LABEL, SRC_COL)
            .withColumnRenamed(INDEX_COL, SRC_COL)
            .join(rels.hint("merge"), edges_df.rel == rels.relation_label)
            .drop(RELATION_LABEL, REL_COL)
            .withColumnRenamed(INDEX_COL, REL_COL)
            .join(nodes.hint("merge"), edges_df.dst == nodes.node_label)
            .drop(NODE_LABEL, DST_COL)
            .withColumnRenamed(INDEX_COL, DST_COL)
        )
    else:
        remapped_edges_df = (
            edges_df.join(nodes.hint("merge"), edges_df.src == nodes.node_label)
            .drop(NODE_LABEL, SRC_COL)
            .withColumnRenamed(INDEX_COL, SRC_COL)
            .join(nodes.hint("merge"), edges_df.dst == nodes.node_label)
            .drop(NODE_LABEL, DST_COL)
            .withColumnRenamed(INDEX_COL, DST_COL)
        )

    return remapped_edges_df


def write_df_to_csv(df, output_filename):
    df.write.csv(TMP_DATA_DIRECTORY, mode="overwrite", sep="\t")
    tmp_file = glob.glob("{}/*.csv".format(TMP_DATA_DIRECTORY))[0]
    os.system("mv {} {}".format(tmp_file, output_filename))
    os.system("rm -rf {}".format(TMP_DATA_DIRECTORY))


class SparkEdgeListConverter(object):
    def __init__(
        self,
        output_dir: Path,
        train_edges: Path,
        valid_edges: Path = None,
        test_edges: Path = None,
        columns: list = [0, 1, 2],
        header_length: int = 0,
        format: str = "csv",
        delim: str = "\t",
        dtype: str = "int32",
        num_partitions: int = 1,
        splits: list = None,
        partitioned_evaluation: bool = False,
        remap_ids: bool = True,
        spark_driver_memory: str = "32g",
        spark_executor_memory: str = "4g",
    ):
        self.output_dir = output_dir

        self.spark = (
            SparkSession.builder.appName(SPARK_APP_NAME)
            .config("spark.driver.memory", spark_driver_memory)
            .config("spark.executor.memory", spark_executor_memory)
            .config("spark.logConf", False)
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("OFF")

        if format.upper() in SUPPORTED_DELIM_FORMATS:
            self.reader = SparkDelimitedFileReader(
                self.spark, train_edges, valid_edges, test_edges, columns, header_length, delim, dtype
            )
        else:
            raise RuntimeError("Unsupported input format")

        self.num_partitions = num_partitions

        if self.num_partitions > 1:
            self.partitioner = SparkPartitioner(self.spark, partitioned_evaluation)
        else:
            self.partitioner = None

        self.writer = SparkWriter(self.spark, self.output_dir, partitioned_evaluation)

        self.train_split = None
        self.valid_split = None
        self.test_split = None

        if splits is not None:
            if len(splits) == 2:
                self.train_split = splits[0]
                self.test_split = splits[1]

                assert (self.train_split + self.test_split) == 1
            if len(splits) == 3:
                self.train_split = splits[0]
                self.valid_split = splits[1]
                self.test_split = splits[2]

                assert (self.train_split + self.valid_split + self.test_split) == 1

        self.has_rels = False
        if len(columns) == 3:
            self.has_rels = True

    def convert(self):
        print("Reading edges")
        all_edges_df, train_edges_df, valid_edges_df, test_edges_df = self.reader.read()

        all_edges_df = remap_columns(all_edges_df, self.has_rels)

        if train_edges_df is not None:
            train_edges_df = remap_columns(train_edges_df, self.has_rels)

        if valid_edges_df is not None:
            valid_edges_df = remap_columns(valid_edges_df, self.has_rels)

        if test_edges_df is not None:
            test_edges_df = remap_columns(test_edges_df, self.has_rels)

        print("Assigning unique IDs")

        # get node and relation labels and assign indices
        nodes_df = get_nodes_df(all_edges_df)

        if self.has_rels:
            rels_df = get_relations_df(all_edges_df)
        else:
            rels_df = None

        print("Remapping edges")

        # replace node and relation labels with indices
        if train_edges_df is not None:
            train_edges_df = remap_edges(train_edges_df, nodes_df, rels_df)
        if valid_edges_df is not None:
            valid_edges_df = remap_edges(valid_edges_df, nodes_df, rels_df)
        if test_edges_df is not None:
            test_edges_df = remap_edges(test_edges_df, nodes_df, rels_df)

        if train_edges_df is None:
            all_edges_df = remap_edges(all_edges_df, nodes_df, rels_df)
            if self.test_split is not None:
                # check if a dataset split is needed
                if self.valid_split is not None:
                    print(
                        "Splitting into: {}/{}/{} fractions".format(self.train_split, self.valid_split, self.test_split)
                    )

                    # split into train/valid/test
                    train_edges_df, valid_edges_df, test_edges_df = all_edges_df.randomSplit(
                        [self.train_split, self.valid_split, self.test_split]
                    )
                else:
                    print("Splitting into: {}/{} fractions".format(self.train_split, self.test_split))
                    # split into train/test
                    train_edges_df, test_edges_df = all_edges_df.randomSplit([self.train_split, self.test_split])
            else:
                train_edges_df = all_edges_df
        all_edges_df, train_edges_df, valid_edges_df, test_edges_df = (
            assign_ids(all_edges_df, EDGES_INDEX_COL),
            assign_ids(train_edges_df, EDGES_INDEX_COL),
            assign_ids(valid_edges_df, EDGES_INDEX_COL),
            assign_ids(test_edges_df, EDGES_INDEX_COL),
        )

        if self.partitioner is not None:
            print("Partition nodes into {} partitions".format(self.num_partitions))
            train_edges_df, valid_edges_df, test_edges_df = self.partitioner.partition_edges(
                train_edges_df, valid_edges_df, test_edges_df, nodes_df, self.num_partitions
            )

        return self.writer.write_to_binary(
            train_edges_df, valid_edges_df, test_edges_df, nodes_df, rels_df, self.num_partitions
        )
