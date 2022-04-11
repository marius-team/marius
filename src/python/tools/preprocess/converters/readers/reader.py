from abc import ABC, abstractmethod


class Reader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read(self):
        """
        This function reads a set of input data and converts it to either torch tensors or pyspark dataframes
        """
        pass
