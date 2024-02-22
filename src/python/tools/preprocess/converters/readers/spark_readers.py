from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame

from marius.tools.preprocess.converters.readers.reader import Reader


class SparkDelimitedFileReader(Reader):
    def __init__(
        self,
        spark: SparkSession,
        train_edges: Path,
        valid_edges: Path = None,
        test_edges: Path = None,
        columns: list = [0, 1, 2],
        header_length: int = 0,
        delim: str = "\t",
        dtype: str = "int32",
    ):
        """
        This class converts an input dataset from a delimited file format, into the format required for input to Marius

        :param spark:                       The spark session to use [REQUIRED]
        :param train_edges:                 The path to the raw training edge list [REQUIRED]
        :param valid_edges:                 The path to the raw validation edge list
        :param test_edges:                  The path to the raw test edge list
                                            it is the train/valid/test split. The sum of this list must be 1.
        :param columns:                     Denotes the columns to extract for the edges. The default is [0, 1, 2],
                                            where the first index is the column id of the src nodes, the second the
                                            relations (edge-types), and the third the dst nodes. For graphs without
                                            edge types, only two ids should be provided.
        :param header_length:               The length of the header of the input edge lists
        :param delim:                       The delimiter used between columns of the input edge lists
        :param dtype:                       The datatype of the assign integer ids to each entity
        """

        super().__init__()

        self.spark = spark

        self.train_edges = train_edges
        self.valid_edges = valid_edges
        self.test_edges = test_edges
        self.columns = columns
        self.header_length = header_length

        self.header = False

        if self.header_length > 1:
            raise RuntimeError("Spark reader unable to support files with multiline headers")
        elif self.header_length == 1:
            self.header = True

        self.delim = delim
        self.dtype = dtype

        if len(self.columns) == 2:
            self.has_rels = False
        elif len(self.columns) == 3:
            self.has_rels = True
        else:
            raise RuntimeError(
                "Incorrect number of columns specified, expected length 2 or 3, received {}".format(len(self.columns))
            )

    def read(self):
        all_edges_df: DataFrame = None
        train_edges_df: DataFrame = None
        valid_edges_df: DataFrame = None
        test_edges_df: DataFrame = None

        if self.valid_edges is None and self.test_edges is None:
            # no validation or test edges supplied

            # read in training edge list
            all_edges_df = self.spark.read.option("header", self.header).csv(self.train_edges.__str__(), sep=self.delim)

            column_order = []
            for i in self.columns:
                column_order.append(all_edges_df.columns[i])

            all_edges_df = all_edges_df.select(column_order)
        else:
            # predefined valid and test edges.
            all_edges_df = self.spark.read.option("header", self.header).csv(
                [self.train_edges.__str__(), self.valid_edges.__str__(), self.test_edges.__str__()], sep=self.delim
            )

            train_edges_df = self.spark.read.option("header", self.header).csv(
                self.train_edges.__str__(), sep=self.delim
            )

            valid_edges_df = self.spark.read.option("header", self.header).csv(
                self.valid_edges.__str__(), sep=self.delim
            )

            test_edges_df = self.spark.read.option("header", self.header).csv(self.test_edges.__str__(), sep=self.delim)

            column_order = []
            for i in self.columns:
                column_order.append(all_edges_df.columns[i])

            all_edges_df = all_edges_df.select(column_order)
            train_edges_df = train_edges_df.select(column_order)
            valid_edges_df = valid_edges_df.select(column_order)
            test_edges_df = test_edges_df.select(column_order)

        return all_edges_df, train_edges_df, valid_edges_df, test_edges_df
