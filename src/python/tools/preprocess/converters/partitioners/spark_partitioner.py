from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner
from pyspark.sql.functions import floor
from pyspark.sql.dataframe import DataFrame
import math

from marius.tools.preprocess.converters.spark_constants import *


def assign_partitions(nodes, num_partitions):
    partition_size = math.ceil(nodes.count() / num_partitions)
    nodes_with_partitions = nodes.withColumn(PARTITION_ID, floor(nodes.index / partition_size)).drop(NODE_LABEL)
    return nodes_with_partitions


def get_edge_buckets(edges_df: DataFrame, nodes_with_partitions: DataFrame, num_partitions: DataFrame):

    partitioned_edges = edges_df.join(nodes_with_partitions, edges_df.src == nodes_with_partitions.index) \
        .drop(NODE_LABEL, INDEX_COL) \
        .withColumnRenamed(PARTITION_ID, SRC_EDGE_BUCKET_COL) \
        .join(nodes_with_partitions, edges_df.dst == nodes_with_partitions.index) \
        .drop(NODE_LABEL, INDEX_COL) \
        .withColumnRenamed(PARTITION_ID, DST_EDGE_BUCKET_COL)
    partitioned_edges = partitioned_edges.repartition(SRC_EDGE_BUCKET_COL)
    partitioned_edges = partitioned_edges.orderBy([SRC_EDGE_BUCKET_COL, DST_EDGE_BUCKET_COL])

    return partitioned_edges


class SparkPartitioner(Partitioner):
    def __init__(self, spark, partitioned_evaluation):
        super().__init__()

        self.spark = spark
        self.partitioned_evaluation = partitioned_evaluation

    def partition_edges(self,
                        train_edges_df,
                        valid_edges_df,
                        test_edges_df,
                        nodes_df,
                        num_partitions):
        """

        """

        nodes_df = assign_partitions(nodes_df, num_partitions)

        train_edges_df = get_edge_buckets(train_edges_df, nodes_df, num_partitions)

        if self.partitioned_evaluation:
            if valid_edges_df is not None:
                valid_edges_df = get_edge_buckets(valid_edges_df, nodes_df, num_partitions)

            if valid_edges_df is not None:
                test_edges_df = get_edge_buckets(test_edges_df, nodes_df, num_partitions)

        return train_edges_df, valid_edges_df, test_edges_df
