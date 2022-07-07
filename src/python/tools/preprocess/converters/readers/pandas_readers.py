from pathlib import Path

import pandas as pd

from marius.tools.preprocess.converters.readers.reader import Reader


class PandasDelimitedFileReader(Reader):
    def __init__(
        self,
        train_edges: Path,
        valid_edges: Path = None,
        test_edges: Path = None,
        columns: list = [0, 1, 2],
        header_length: int = 0,
        delim: str = "\t",
    ):
        """
        This class converts an input dataset from a delimited file format, into the format required for input to Marius

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
        """

        super().__init__()

        self.train_edges = train_edges
        self.valid_edges = valid_edges
        self.test_edges = test_edges
        self.columns = columns
        self.header_length = header_length

        self.delim = delim

        if len(self.columns) == 2:
            self.has_rels = False
        elif len(self.columns) == 3:
            self.has_rels = True
        else:
            raise RuntimeError(
                "Incorrect number of columns specified, expected length 2 or 3, received {}".format(len(self.columns))
            )

    def read(self):
        train_edges_df: pd.DataFrame = None
        valid_edges_df: pd.DataFrame = None
        test_edges_df: pd.DataFrame = None

        assert self.train_edges is not None
        train_edges_df = pd.read_csv(self.train_edges, delimiter=self.delim, skiprows=self.header_length, header=None)
        train_edges_df = train_edges_df[train_edges_df.columns[self.columns]]

        if self.valid_edges is not None:
            valid_edges_df = pd.read_csv(
                self.valid_edges, delimiter=self.delim, skiprows=self.header_length, header=None
            )
            valid_edges_df = valid_edges_df[valid_edges_df.columns[self.columns]]
        if self.test_edges is not None:
            test_edges_df = pd.read_csv(self.test_edges, delimiter=self.delim, skiprows=self.header_length, header=None)
            test_edges_df = test_edges_df[test_edges_df.columns[self.columns]]

        return train_edges_df, valid_edges_df, test_edges_df
