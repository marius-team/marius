from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import Row
import pyspark.sql.functions as f
from pyspark.sql.functions import spark_partition_id, asc, desc, rand, row_number, monotonically_increasing_id, expr, floor, col, count, when
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, LongType
import math
import glob
import os
import path
import pandas as pd
import numpy as np

SRC_COL = "src"
REL_COL = "rel"
DST_COL = "dst"
INDEX_COL = "index"
SRC_EDGE_BUCKET_COL = "src_part"
DST_EDGE_BUCKET_COL = "dst_part"
PARTITION_ID = "partition_id"
NODE_LABEL = "node_label"
RELATION_LABEL = "relation_label"
TMP_DATA_DIRECTORY = "tmp_pyspark"

COLUMN_SCHEMA = [SRC_COL, REL_COL, DST_COL]

SPARK_APP_NAME = "marius_preprocessor"


def remap_columns(df, has_rels):
    if has_rels:
        df = df.rdd.map(lambda x: (x.src, x.rel, x.dst)).toDF(COLUMN_SCHEMA)
    else:
        df = df.rdd.map(lambda x: (x.src, "0", x.dst)).toDF(COLUMN_SCHEMA)

    return df


def convert_to_binary(input_filename, output_filename):

    assert(input_filename != output_filename)

    with open(output_filename, "wb") as output_file:
        for chunk in pd.read_csv(input_filename, header=None, chunksize=10 ** 7, sep="\t", dtype=int):
            chunk_array = chunk.to_numpy(dtype=np.int32)
            output_file.write(bytes(chunk_array))

    os.system("rm {}".format(input_filename))


def write_df_to_csv(df, output_filename):
    df.coalesce(1).write.csv(TMP_DATA_DIRECTORY, mode="overwrite", sep="\t")
    tmp_file = glob.glob("{}/*.csv".format(TMP_DATA_DIRECTORY))[0]
    os.system("mv {} {}".format(tmp_file, output_filename))
    os.system("rm -rf {}".format(TMP_DATA_DIRECTORY))


def write_partitioned_df_to_csv(partition_triples, num_partitions, output_filename):
    partition_triples.write.partitionBy(SRC_EDGE_BUCKET_COL, DST_EDGE_BUCKET_COL).csv(TMP_DATA_DIRECTORY, mode="overwrite", sep="\t")

    partition_offsets = []
    with open(output_filename, "w") as output_file:
        for i in range(num_partitions):
            for j in range(num_partitions):
                tmp_file = glob.glob("{}/{}={}/{}={}/*.csv".format(TMP_DATA_DIRECTORY, SRC_EDGE_BUCKET_COL, str(i), DST_EDGE_BUCKET_COL, str(j)))[0]
                pd.read_csv(tmp_file, sep="\t")
                with open(tmp_file, 'r') as g:
                    lines = g.readlines()
                    partition_offsets.append(len(lines))
                    output_file.writelines(lines)

    os.system("rm -rf {}".format(TMP_DATA_DIRECTORY))

    return partition_offsets


def assign_ids(spark, df, offset=0, col_name="index"):
    new_schema = StructType([StructField(col_name, LongType(), True)] + df.schema.fields)
    zipped_rdd = df.rdd.zipWithIndex()
    new_rdd = zipped_rdd.map(lambda args: ([args[1] + offset] + list(args[0])))

    return spark.createDataFrame(new_rdd, new_schema)


def remap_edges(edges_df, nodes, rels):

    remapped_edges_df = edges_df.join(nodes, edges_df.src == nodes.node_label) \
        .drop(NODE_LABEL, SRC_COL) \
        .withColumnRenamed(INDEX_COL, SRC_COL) \
        .join(rels, edges_df.rel == rels.relation_label) \
        .drop(RELATION_LABEL, REL_COL) \
        .withColumnRenamed(INDEX_COL, REL_COL) \
        .join(nodes, edges_df.dst == nodes.node_label) \
        .drop(NODE_LABEL, DST_COL) \
        .withColumnRenamed(INDEX_COL, DST_COL)

    return remapped_edges_df


def get_edge_buckets(edges_df, nodes_with_partitions, num_partitions):
    partition_triples = edges_df.join(nodes_with_partitions, edges_df.src == nodes_with_partitions.index) \
        .drop(NODE_LABEL, INDEX_COL) \
        .withColumnRenamed(PARTITION_ID, SRC_EDGE_BUCKET_COL) \
        .join(nodes_with_partitions, edges_df.dst == nodes_with_partitions.index) \
        .drop(NODE_LABEL, INDEX_COL) \
        .withColumnRenamed(PARTITION_ID, DST_EDGE_BUCKET_COL) \

    partition_triples = partition_triples.repartition(num_partitions**2, SRC_EDGE_BUCKET_COL, DST_EDGE_BUCKET_COL)

    return partition_triples


def assign_partitions(nodes, num_partitions):
    partition_size = math.ceil(nodes.count() / num_partitions)
    nodes_with_partitions = nodes.withColumn(PARTITION_ID, floor(nodes.index / partition_size)).drop(NODE_LABEL)
    return nodes_with_partitions


def get_nodes_df(spark, edges_df):
    nodes = edges_df.rdd.flatMap(lambda x: (x.src, x.dst))
    nodes = nodes.distinct()
    nodes = nodes.map(Row(NODE_LABEL)).toDF().repartition(1).orderBy(rand())
    nodes = assign_ids(spark, nodes)
    return nodes


def get_relations_df(spark, edges_df):
    rels = edges_df.drop(SRC_COL, DST_COL).distinct().repartition(1).orderBy(rand()).withColumnRenamed(REL_COL, RELATION_LABEL)
    rels = assign_ids(spark, rels)
    return rels


def preprocess_dataset(edges_files, num_partitions, output_dir, splits=(.05, .05), columns=None, header=False):

    map_columns = False
    has_rels = True
    if columns is None:
        columns = COLUMN_SCHEMA
    else:
        map_columns = True
        if REL_COL not in columns:
            has_rels = False

    spark = SparkSession.builder.appName(SPARK_APP_NAME).config('spark.executor.memory', '10g').getOrCreate()

    all_edges_df = None
    train_edges_df = None
    valid_edges_df = None
    test_edges_df = None
    if len(edges_files) == 1:
        train_split = 1.0 - splits[0] - splits[1]
        valid_split = splits[0]
        test_split = splits[1]
        all_edges_df = spark.read.option("header", header).csv(edges_files, sep="\t").toDF(*columns)

        if map_columns:
            all_edges_df = remap_columns(all_edges_df, has_rels)

        train_edges_df, valid_edges_df, test_edges_df = all_edges_df.randomSplit([train_split, valid_split, test_split])

    elif len(edges_files) == 3:
        all_edges_df = spark.read.option("header", header).csv(edges_files, sep="\t").toDF(*columns)
        train_edges_df = spark.read.option("header", header).csv(edges_files[0], sep="\t").toDF(*columns)
        valid_edges_df = spark.read.option("header", header).csv(edges_files[1], sep="\t").toDF(*columns)
        test_edges_df = spark.read.option("header", header).csv(edges_files[2], sep="\t").toDF(*columns)

        if map_columns:
            all_edges_df = remap_columns(all_edges_df, has_rels)
            train_edges_df = remap_columns(train_edges_df, has_rels)
            valid_edges_df = remap_columns(valid_edges_df, has_rels)
            test_edges_df = remap_columns(test_edges_df, has_rels)



    else:
        print("Incorrect number of input files")
        exit(-1)

    nodes = get_nodes_df(spark, all_edges_df)
    rels = get_relations_df(spark, all_edges_df)

    train_edges_df = remap_edges(train_edges_df, nodes, rels)
    valid_edges_df = remap_edges(valid_edges_df, nodes, rels)
    test_edges_df = remap_edges(test_edges_df, nodes, rels)

    write_df_to_csv(nodes, output_dir + "node_mapping.txt")
    write_df_to_csv(rels, output_dir + "relation_mapping.txt")

    tmp_train_file = output_dir + "tmp_train_edges.txt"
    tmp_valid_file = output_dir + "tmp_valid_edges.txt"
    tmp_test_file = output_dir + "tmp_test_edges.txt"

    if num_partitions > 1:
        nodes_with_partitions = assign_partitions(nodes, num_partitions)
        partition_triples = get_edge_buckets(train_edges_df, nodes_with_partitions, num_partitions)
        partition_offsets = write_partitioned_df_to_csv(partition_triples, num_partitions, tmp_train_file)

        with open(output_dir + "train_edges_partitions.txt", "w") as g:
            g.writelines([str(o) + "\n" for o in partition_offsets])
    else:
        write_df_to_csv(train_edges_df, tmp_train_file)

    write_df_to_csv(valid_edges_df, tmp_valid_file)
    write_df_to_csv(test_edges_df, tmp_test_file)

    convert_to_binary(tmp_train_file, output_dir + "train_edges.pt")
    convert_to_binary(tmp_valid_file, output_dir + "valid_edges.pt")
    convert_to_binary(tmp_test_file, output_dir + "test_edges.pt")
