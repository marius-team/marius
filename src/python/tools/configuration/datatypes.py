from enum import Enum
from dataclasses import dataclass
# This file contains enums and detailed option settings for each enum value, where applicable


class LearningTask(Enum):
    NODE_CLASSIFICATION = "nc"
    LINK_PREDICTION = "lp"


class InitializationDistribution(Enum):
    ZEROS = "zeros"
    ONES = "ones"
    CONSTANT = "constant"
    UNIFORM = "uniform"
    NORMAL = "normal"
    GLOROT_UNIFORM = "glorot_uniform"
    GLOROT_NORMAL = "glorot_normal"


class LossFunction(Enum):
    SOFTMAX = "softmax"
    RANKING = "ranking"
    BCE_AFTER_SIGMOID = "bce_sigmoid"
    BCE_WITH_LOGITS = "bce_logits"
    MSE = "mse"
    SOFTPLUS = "softplus"


class LossReduction(Enum):
    MEAN = "mean"
    SUM = "sum"


class ActivationFunction(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    NONE = "none"

# TODO add options for each activation function, if needed


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAGRAD = "adagrad"


class FeaturizerType(Enum):
    NONE = "none"
    CONCAT = "concat"
    SUM = "sum"
    MEAN = "mean"
    LINEAR = "linear"


class GNNLayerType(Enum):
    NONE = "none"
    GRAPH_SAGE = "graphsage"
    GCN = "gcn"
    GAT = "gat"
    RGCN = "rgcn"


class GraphSageAggregator(Enum):
    GCN = "gcn"
    MEAN = "mean"


class DecoderType(Enum):
    NONE = "none"
    DISTMULT = "distmult"
    TRANSE = "transe"
    COMPLEX = "complex"


class StorageBackend(Enum):
    PARTITION_BUFFER = "buffer"
    FLAT_FILE = "file"
    HOST_MEMORY = "host"
    DEVICE_MEMORY = "device"


class EdgeBucketOrdering(Enum):
    OLD_BETA = "old_beta"
    NEW_BETA = "new_beta"
    ALL_BETA = "all_beta"
    TWO_LEVEL_BETA = "two_level_beta"
    CUSTOM = "custom"


class NodePartitionOrdering(Enum):
    DISPERSED = "dispersed"
    SEQUENTIAL = "sequential"
    CUSTOM = "custom"


class NeighborSamplingLayer(Enum):
    ALL = "all"
    UNIFORM = "uniform"
    DROPOUT = "dropout"


# options dataclasses
@dataclass
class InitOptions:
    pass


@dataclass
class UniformInitOptions(InitOptions):
    scale_factor: float = 1

    def __post_init__(self):
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")


@dataclass
class NormalInitOptions(InitOptions):
    mean: float = 0
    std: float = 1

    def __post_init__(self):
        if self.std <= 0:
            raise ValueError("std must be positive")


@dataclass
class ConstantInitOptions(InitOptions):
    constant: float = 0


@dataclass
class LossOptions:
    reduction: str = "SUM"


@dataclass
class RankingLossOptions(LossOptions):
    margin: float = .1


@dataclass
class OptimizerOptions:
    learning_rate: float = .1

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class AdagradOptions(OptimizerOptions):
    learning_rate = .1
    eps: float = 1e-10
    init_value: float = 0
    lr_decay: float = 0
    weight_decay: float = 0


@dataclass
class AdamOptions(OptimizerOptions):
    learning_rate = .1
    amsgrad: bool = False
    beta_1: float = .9
    beta_2: float = .999
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class FeaturizerOptions:
    pass


@dataclass
class GNNLayerOptions:
    input_dim: int = 50
    output_dim: int = 50

    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")


@dataclass
class GraphSageLayerOptions(GNNLayerOptions):
    aggregator: str = "GCN"


@dataclass
class GATLayerOptions(GNNLayerOptions):
    num_heads: int = 10
    average_heads: bool = True
    negative_slope: float = .2
    input_dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")


@dataclass
class DecoderOptions:
    input_dim: int = 50
    inverse_edges: bool = True

    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError("embedding_dim must be positive")


@dataclass
class StorageOptions:
    dtype: str = "float"


@dataclass
class PartitionBufferOptions(StorageOptions):
    num_partitions: int = 16
    buffer_capacity: int = 8
    prefetching: bool = True
    fine_to_coarse_ratio: int = 1
    num_cache_partitions: int = 0
    edge_bucket_ordering: str = "NEW_BETA"
    node_partition_ordering: str = "DISPERSED"
    randomly_assign_edge_buckets: bool = True

    def __post_init__(self):
        if self.num_partitions < 2:
            raise ValueError("There must be at least two partitions to use the partition buffer, got: {}".format(
                self.num_partitions))
        if self.buffer_capacity < 2:
            raise ValueError("The partition buffer must have capacity of at least 2, got: {}".format(
                self.buffer_capacity))

        # no need to have a buffer capacity larger than the number of partitions
        if self.num_partitions < self.buffer_capacity:
            self.buffer_capacity = self.num_partitions


@dataclass
class NeighborSamplingOptions:
    pass


@dataclass
class UniformSamplingOptions(NeighborSamplingOptions):
    max_neighbors: int = 10

    def __post_init__(self):
        if self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")


@dataclass
class DropoutSamplingOptions(NeighborSamplingOptions):
    rate: float = 0.0

    def __post_init__(self):
        if self.rate < 0 or self.rate >= 1:
            raise ValueError("rate must be in [0, 1)")

