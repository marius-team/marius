import math

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import floor

from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner
from marius.tools.preprocess.converters.spark_constants import DST_EDGE_BUCKET_COL, INDEX_COL, SRC_EDGE_BUCKET_COL
from marius.tools.preprocess.utils import get_df_count


def get_partition_size(nodes, num_partitions):
    partition_size = math.ceil(get_df_count(nodes, INDEX_COL) / num_partitions)
    return partition_size


def get_edge_buckets(edges_df: DataFrame, partition_size):
    partitioned_edges = edges_df.withColumn(SRC_EDGE_BUCKET_COL, floor(edges_df.src / partition_size)).withColumn(
        DST_EDGE_BUCKET_COL, floor(edges_df.dst / partition_size)
    )
    return partitioned_edges


class SparkPartitioner(Partitioner):
    def __init__(self, spark, partitioned_evaluation):
        super().__init__()

        self.spark = spark
        self.partitioned_evaluation = partitioned_evaluation

    def partition_edges(self, train_edges_df, valid_edges_df, test_edges_df, nodes_df, num_partitions):
        """ """
        partition_size = get_partition_size(nodes_df, num_partitions)
        train_edges_df = get_edge_buckets(train_edges_df, partition_size)

        if self.partitioned_evaluation:
            if valid_edges_df is not None:
                valid_edges_df = get_edge_buckets(valid_edges_df, partition_size)

            if test_edges_df is not None:
                test_edges_df = get_edge_buckets(test_edges_df, partition_size)

        return train_edges_df, valid_edges_df, test_edges_df
