from pathlib import Path

import pandas as pd

from marius.tools.preprocess.converters.readers.reader import Reader
from marius.tools.preprocess.converters.torch_constants import TorchConverterColumnKeys as ColNames


class PandasDelimitedFileReader(Reader):
    def __init__(
        self,
        train_edges: Path,
        valid_edges: Path = None,
        test_edges: Path = None,
        columns: dict = {},
        header_length: int = 0,
        delim: str = "\t",
    ):
        """
        This class converts an input dataset from a delimited file format, into the format required for input to Marius

        :param train_edges:                 The path to the raw training edge list [REQUIRED]
        :param valid_edges:                 The path to the raw validation edge list
        :param test_edges:                  The path to the raw test edge list
                                            it is the train/valid/test split. The sum of this list must be 1.
        :param columns:                     A dict containing the columns we want to extract and the names we want
                                            to assing them. The key should be the name we want to assign the column
                                            and the value is the column id.
                                            Any columns with a None id are ignored.
        :param header_length:               The length of the header of the input edge lists
        :param delim:                       The delimiter used between columns of the input edge lists
        """

        super().__init__()

        assert train_edges is not None
        self.train_edges = train_edges
        self.valid_edges = valid_edges
        self.test_edges = test_edges
        self.header_length = header_length
        self.columns = columns
        self.delim = delim

    def read_single_file(self, file_path):
        if file_path is None:
            return None

        # Determine the columns to read
        cols_to_keeps = []
        id_to_name_mapping = {}
        for col_name, col_id in self.columns.items():
            if col_id is not None:
                cols_to_keeps.append(col_id)
                id_to_name_mapping[col_id] = col_name.value

        # Read the file and extracted the columns we need
        file_data = pd.read_csv(file_path, delimiter=self.delim, skiprows=self.header_length, header=None)
        file_data = file_data[cols_to_keeps]
        file_data = file_data.rename(columns=id_to_name_mapping)

        # Make sure we got the src and dst columns
        columns_read = list(file_data.columns)
        assert "src_column" in columns_read
        assert "dst_column" in columns_read

        # Ensure that data is in the proper order
        cols_order = [ColNames.SRC_COL.value, ColNames.DST_COL.value]
        if "edge_type_column" in columns_read:
            cols_order.insert(len(cols_order) - 1, ColNames.EDGE_TYPE_COL.value)

        if "edge_weight_column" in columns_read:
            cols_order.insert(len(cols_order), ColNames.EDGE_WEIGHT_COL.value)

        file_data = file_data[cols_order]
        return file_data

    def read(self):
        return (
            self.read_single_file(self.train_edges),
            self.read_single_file(self.valid_edges),
            self.read_single_file(self.test_edges),
        )
