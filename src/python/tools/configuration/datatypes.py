from dataclasses import dataclass

# This file contains enums and detailed option settings for each enum value, where applicable


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
    margin: float = 0.1


@dataclass
class OptimizerOptions:
    learning_rate: float = 0.1

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class AdagradOptions(OptimizerOptions):
    learning_rate = 0.1
    eps: float = 1e-10
    init_value: float = 0
    lr_decay: float = 0
    weight_decay: float = 0

    def __post_init__(self):
        if self.init_value < 0:
            raise ValueError("init_value for AdaGradOptimizer must be non-negative")
        # is this the case??
        if self.lr_decay < 0:
            raise ValueError("lr_decay for AdaGradOptimizer must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay for AdaGradOptimizer must be non-negative")


@dataclass
class AdamOptions(OptimizerOptions):
    learning_rate = 0.1
    amsgrad: bool = False
    beta_1: float = 0.9
    beta_2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0

    def __post_init__(self):
        if self.beta_1 < 0:
            raise ValueError("beta_1 for AdamOptimizer must be non-negative")
        # is this the case??
        if self.beta_2 < 0:
            raise ValueError("beta_2 for AdamOptimizer must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay for AdamOptimizer  must be non-negative")


@dataclass
class LayerOptions:
    pass


@dataclass
class EmbeddingLayerOptions(LayerOptions):
    pass


@dataclass
class FeatureLayerOptions(LayerOptions):
    pass


@dataclass
class DenseLayerOptions(LayerOptions):
    type: str = "LINEAR"


@dataclass
class ReductionLayerOptions(LayerOptions):
    type: str = "CONCAT"


@dataclass
class GNNLayerOptions(LayerOptions):
    type: str
    pass


@dataclass
class GraphSageLayerOptions(GNNLayerOptions):
    type: str = "GRAPH_SAGE"
    aggregator: str = "GCN"


@dataclass
class GATLayerOptions(GNNLayerOptions):
    type: str = "GAT"
    num_heads: int = 10
    average_heads: bool = True
    negative_slope: float = 0.2
    input_dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")


@dataclass
class DecoderOptions:
    pass


@dataclass
class EdgeDecoderOptions(DecoderOptions):
    inverse_edges: bool = True
    use_relation_features: bool = False
    edge_decoder_method: str = "CORRUPT_NODE"


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
    edge_bucket_ordering: str = "COMET"
    node_partition_ordering: str = "DISPERSED"
    randomly_assign_edge_buckets: bool = True

    def __post_init__(self):
        if self.num_partitions < 2:
            raise ValueError(
                "There must be at least two partitions to use the partition buffer, got: {}".format(self.num_partitions)
            )
        if self.buffer_capacity < 2:
            raise ValueError(
                "The partition buffer must have capacity of at least 2, got: {}".format(self.buffer_capacity)
            )

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
