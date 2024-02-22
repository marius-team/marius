from enum import Enum, unique


@unique
class TorchConverterColumnKeys(Enum):
    SRC_COL = "src_column"
    DST_COL = "dst_column"
    EDGE_TYPE_COL = "edge_type_column"
    EDGE_WEIGHT_COL = "edge_weight_column"

    def __hash__(self) -> int:
        return hash(self.name)
