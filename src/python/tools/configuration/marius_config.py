import dataclasses
import enum
import random
import sys

from marius.tools.configuration.datatypes import *
from marius.tools.configuration.constants import PathConstants
from dataclasses import field
import os

from pathlib import Path

from omegaconf import MISSING, OmegaConf, DictConfig

import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class NeighborSamplingConfig:
    type: str = NeighborSamplingLayer.ALL.name
    options: NeighborSamplingOptions = NeighborSamplingOptions()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = NeighborSamplingOptions()

        if self.type == NeighborSamplingLayer.UNIFORM.name:
            new_options = UniformSamplingOptions()

        if self.type == NeighborSamplingLayer.DROPOUT.name:
            new_options = DropoutSamplingOptions()

        if "options" in input_config.keys():
            for key in new_options.__dict__.keys():
                if key in input_config.options.keys():
                    val = input_config.options.__getattr__(key)
                    new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class OptimizerConfig:
    type: str = "ADAGRAD"
    options: OptimizerOptions = AdagradOptions()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = OptimizerOptions()

        if self.type == OptimizerType.ADAGRAD.name:
            new_options = AdagradOptions()

        if self.type == OptimizerType.ADAM.name:
            new_options = AdamOptions()

        for key in new_options.__dict__.keys():
            if key in input_config.options.keys():
                val = input_config.options.__getattr__(key)
                new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class InitConfig:
    type: str = "GLOROT_UNIFORM"
    options: InitOptions = InitOptions()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = InitOptions()

        if self.type == InitializationDistribution.CONSTANT.name:
            new_options = ConstantInitOptions()

        if self.type == InitializationDistribution.UNIFORM.name:
            new_options = UniformInitOptions()

        if self.type == InitializationDistribution.NORMAL.name:
            new_options = NormalInitOptions()

        for key in new_options.__dict__.keys():
            if key in input_config.options.keys():
                val = input_config.options.__getattr__(key)
                new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class LossConfig:
    type: str = "SOFTMAX"
    options: LossOptions = LossOptions()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = LossOptions()

        if self.type == LossFunction.RANKING.name:
            new_options = RankingLossOptions()

        for key in new_options.__dict__.keys():
            if key in input_config.options.keys():
                val = input_config.options.__getattr__(key)
                new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class EmbeddingsConfig:
    dimension: int = 50
    init: InitConfig = InitConfig(type="UNIFORM",
                                  options=UniformInitOptions(.001))
    optimizer: OptimizerConfig = OptimizerConfig()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        if "dimension" in input_config.keys():
            self.dimension = input_config.dimension

        if "init" in input_config.keys():
            self.init.merge(input_config.init)

        if "optimizer" in input_config.keys():
            self.optimizer.merge(input_config.optimizer)


@dataclass
class FeaturizerConfig:
    type: str = "NONE"
    options: FeaturizerOptions = FeaturizerOptions()
    optimizer: OptimizerConfig = OptimizerConfig()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        # add featurizer specific options

        if "optimizer" in input_config.keys():
            self.optimizer.merge(input_config.optimizer)


@dataclass
class GNNLayerConfig:
    type: str = "GRAPH_SAGE"
    options: GNNLayerOptions = GraphSageLayerOptions(input_dim=50, output_dim=50)
    train_neighbor_sampling: NeighborSamplingConfig = NeighborSamplingConfig()
    eval_neighbor_sampling: NeighborSamplingConfig = NeighborSamplingConfig()
    init: InitConfig = InitConfig(type="GLOROT_UNIFORM")
    activation: str = "NONE"
    bias: bool = True
    bias_init: InitConfig = InitConfig(type="ZEROS")

    def __init__(self):
        self.type: str = "GRAPH_SAGE"
        self.options: GNNLayerOptions = GraphSageLayerOptions(input_dim=50, output_dim=50)
        self.train_neighbor_sampling: NeighborSamplingConfig = NeighborSamplingConfig()
        self.eval_neighbor_sampling: NeighborSamplingConfig = NeighborSamplingConfig()
        self.init: InitConfig = InitConfig(type="GLOROT_UNIFORM")
        self.activation: str = "NONE"
        self.bias: bool = True
        self.bias_init: InitConfig = InitConfig(type="ZEROS")


    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        if "train_neighbor_sampling" in input_config.keys():
            self.train_neighbor_sampling.merge(input_config.train_neighbor_sampling)

        if "eval_neighbor_sampling" in input_config.keys():
            self.eval_neighbor_sampling.merge(input_config.eval_neighbor_sampling)

        if "init" in input_config.keys():
            self.init.merge(input_config.init)

        new_options = GNNLayerOptions()

        if self.type == GNNLayerType.GRAPH_SAGE.name:
            new_options = GraphSageLayerOptions()

        if self.type == GNNLayerType.GAT.name:
            new_options = GATLayerOptions()

        for key in new_options.__dict__.keys():
            if key in input_config.options.keys():
                val = input_config.options.__getattr__(key)
                new_options.__setattr__(key, val)

        self.options = new_options

        if "activation" in input_config.keys():
            self.activation = input_config.activation.upper()

        if "bias" in input_config.keys():
            self.bias = input_config.bias

        if "bias_init" in input_config.keys():
            self.bias_init.merge(input_config.bias_init)


@dataclass
class EncoderConfig:
    input_dim: int = -1  # these can be inferred from the layers list
    output_dim: int = -1
    layers: list = field(default_factory=list)
    optimizer: OptimizerConfig = OptimizerConfig()
    use_incoming_nbrs: bool = True
    use_outgoing_nbrs: bool = True
    use_hashmap_sets: bool = True

    def __post_init__(self):

        if len(self.layers) > 0:
            self.input_dim = self.layers[0].options.input_dim
            self.output_dim = self.layers[-1].options.output_dim

            prev_layer_output = self.input_dim
            for i, layer in enumerate(self.layers):
                if layer.options.input_dim != prev_layer_output:
                    raise ValueError("Layer {} dimension mismatch. "
                                     "Output dim of previous layer: {}. "
                                     "Input dim of current layer: {}".format(i, prev_layer_output, layer.options.input_dim))
                prev_layer_output = layer.options.output_dim

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        new_layers = []
        if "layers" in input_config.keys():
            for layer_config in input_config.layers:
                base_layer = GNNLayerConfig()
                base_layer.merge(layer_config)
                new_layers.append(base_layer)

        self.layers = new_layers

        if "optimizer" in input_config.keys():
            self.optimizer.merge(input_config.optimizer)

        if "use_incoming_nbrs" in input_config.keys():
            self.use_incoming_nbrs = input_config.use_incoming_nbrs

        if "use_outgoing_nbrs" in input_config.keys():
            self.use_outgoing_nbrs = input_config.use_outgoing_nbrs

        if "use_hashmap_sets" in input_config.keys():
            self.use_hashmap_sets = input_config.use_hashmap_sets

        self.__post_init__()


@dataclass
class DecoderConfig:
    type: str = "DISTMULT"
    options: DecoderOptions = DecoderOptions()
    optimizer: OptimizerConfig = OptimizerConfig()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = DecoderOptions()

        if "options" in input_config.keys():
            for key in new_options.__dict__.keys():
                if key in input_config.options.keys():
                    val = input_config.options.__getattr__(key)
                    new_options.__setattr__(key, val)

        self.options = new_options

        if "optimizer" in input_config.keys():
            self.optimizer.merge(input_config.optimizer)


@dataclass
class ModelConfig:
    random_seed: int = MISSING
    learning_task: str = MISSING
    embeddings: EmbeddingsConfig = MISSING
    featurizer: FeaturizerConfig = MISSING
    encoder: EncoderConfig = MISSING
    decoder: DecoderConfig = MISSING
    loss: LossConfig = MISSING

    def __post_init__(self):
        if (self.learning_task == "NODE_CLASSIFICATION") and (self.decoder is not None):
            raise ValueError("Decoders are currently only supported in link prediction")

        if (self.encoder is not MISSING) and (self.decoder is not MISSING):
            if len(self.encoder.layers) > 0:
                if self.encoder.output_dim != self.decoder.options.input_dim:
                    raise ValueError("Encoder decoder dimension mismatch. "
                                     "Output dim encoder: {}. "
                                     "Input dim of decoder: {}".format(self.encoder.output_dim,
                                                                       self.decoder.options.input_dim))

        if self.random_seed is MISSING:
            self.random_seed = random.randint(0, sys.maxsize)

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return:
        """

        if "random_seed" in input_config.keys():
            self.random_seed = input_config.random_seed

        if "learning_task" in input_config.keys():
            self.learning_task = input_config.learning_task.upper()

        if "embeddings" in input_config.keys():
            if self.embeddings is MISSING:
                self.embeddings = EmbeddingsConfig()

            self.embeddings.merge(input_config.embeddings)

        if "featurizer" in input_config.keys():
            if self.featurizer is MISSING:
                self.featurizer = FeaturizerConfig()
            self.featurizer.merge(input_config.featurizer)

        if "encoder" in input_config.keys():
            if self.encoder is MISSING:
                self.encoder = EncoderConfig()
            self.encoder.merge(input_config.encoder)

        if "decoder" in input_config.keys():
            if self.decoder is MISSING:
                self.decoder = DecoderConfig()
            self.decoder.merge(input_config.decoder)

        if "loss" in input_config.keys():
            if self.loss is MISSING:
                self.loss = LossConfig()
            self.loss.merge(input_config.loss)


@dataclass
class StorageBackendConfig:
    type: str = "DEVICE_MEMORY"
    options: StorageOptions = StorageOptions(dtype="float")

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = self.options

        if self.type == StorageBackend.PARTITION_BUFFER.name:
            new_options = PartitionBufferOptions()

        if "options" in input_config.keys():
            for key in new_options.__dict__.keys():
                if key in input_config.options.keys():
                    val = input_config.options.__getattr__(key)
                    new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class DatasetConfig:
    base_directory: str = MISSING
    num_edges: int = MISSING
    num_nodes: int = MISSING
    num_relations: int = MISSING
    num_train: int = MISSING
    num_valid: int = MISSING
    num_test: int = MISSING
    feature_dim: int = -1
    num_classes: int = -1

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        for key in self.__dict__.keys():
            if key in input_config.keys():
                val = input_config.__getattr__(key)
                self.__setattr__(key, val)


@dataclass
class StorageConfig:
    device_type: str = "cpu"
    device_ids: list = field(default_factory=list)
    dataset: DatasetConfig = DatasetConfig()
    edges: StorageBackendConfig = StorageBackendConfig(options=StorageOptions(dtype="int"))
    nodes: StorageBackendConfig = MISSING
    embeddings: StorageBackendConfig = MISSING
    features: StorageBackendConfig = MISSING
    prefetch: bool = True
    shuffle_input: bool = True
    full_graph_evaluation: bool = True

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        if "device_type" in input_config.keys():
            self.device_type = input_config.device_type

        if "device_ids" in input_config.keys():
            self.device_ids = input_config.device_ids

        if "dataset" in input_config.keys():
            self.dataset.merge(input_config.dataset)

        if "edges" in input_config.keys():
            self.edges.merge(input_config.edges)

        if "nodes" in input_config.keys():
            if self.nodes is MISSING:
                self.nodes = StorageBackendConfig(options=StorageOptions(dtype="int"))
            self.nodes.merge(input_config.nodes)

        if "embeddings" in input_config.keys():
            if self.embeddings is MISSING:
                self.embeddings = StorageBackendConfig(options=StorageOptions(dtype="float"))
            self.embeddings.merge(input_config.embeddings)

        if "features" in input_config.keys():
            if self.features is MISSING:
                self.features = StorageBackendConfig(options=StorageOptions(dtype="float"))
            self.features.merge(input_config.features)

        if "prefetch" in input_config.keys():
            self.prefetch = input_config.prefetch

        if "shuffle_input" in input_config.keys():
            self.shuffle_input = input_config.shuffle_input


@dataclass
class NegativeSamplingConfig:
    num_chunks: int = 1
    negatives_per_positive: int = 1000
    degree_fraction: float = 0
    filtered: bool = False

    def __post_init__(self):
        # for filtered mrr, the sampling class members should be ignored
        if not self.filtered:
            if self.num_chunks <= 0:
                raise ValueError("num_chunks must be positive")
            if self.negatives_per_positive <= 0:
                raise ValueError("negatives_per_positive must be positive")
            if self.degree_fraction < 0:
                raise ValueError("degree_fraction must not be negative")
        else:
            self.num_chunks = MISSING
            self.negatives_per_positive = MISSING
            self.degree_fraction = MISSING

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        if "num_chunks" in input_config.keys():
            self.num_chunks = input_config.num_chunks

        if "negatives_per_positive" in input_config.keys():
            self.negatives_per_positive = input_config.negatives_per_positive

        if "degree_fraction" in input_config.keys():
            self.degree_fraction = input_config.degree_fraction

        if "filtered" in input_config.keys():
            self.filtered = input_config.filtered

        self.__post_init__()


@dataclass
class PipelineConfig:
    sync: bool = True
    staleness_bound: int = 16
    batch_host_queue_size: int = 4
    batch_device_queue_size: int = 4
    gradients_device_queue_size: int = 4
    gradients_host_queue_size: int = 4
    batch_loader_threads: int = 4
    batch_transfer_threads: int = 2
    compute_threads: int = 1
    gradient_transfer_threads: int = 2
    gradient_update_threads: int = 4

    def __post_init__(self):
        # for the sync setting, pipeline values can be ignored
        if not self.sync:
            if self.staleness_bound <= 0:
                raise ValueError("staleness_bound must be positive")
            if self.batch_host_queue_size <= 0:
                raise ValueError("batch_host_queue_size must be positive")
            if self.batch_device_queue_size <= 0:
                raise ValueError("batch_device_queue_size must be positive")
            if self.gradients_device_queue_size <= 0:
                raise ValueError("gradients_device_queue_size must be positive")
            if self.batch_loader_threads <= 0:
                raise ValueError("batch_loader_threads must be positive")
            if self.batch_transfer_threads <= 0:
                raise ValueError("batch_transfer_threads must be positive")
            if self.compute_threads <= 0:
                raise ValueError("compute_threads must be positive")
            if self.gradient_transfer_threads <= 0:
                raise ValueError("gradient_transfer_threads must be positive")
            if self.gradient_update_threads <= 0:
                raise ValueError("gradient_update_threads must be positive")

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        for key in self.__dict__.keys():
            if key in input_config.keys():
                val = input_config.__getattr__(key)
                self.__setattr__(key, val)

        self.__post_init__()


@dataclass
class TrainingConfig:
    batch_size: int = 1000
    negative_sampling: NegativeSamplingConfig = MISSING
    num_epochs: int = 10
    pipeline: PipelineConfig = PipelineConfig()
    epochs_per_shuffle: int = 1
    logs_per_epoch: int = 10

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.epochs_per_shuffle <= 0:
            raise ValueError("epochs_per_shuffle must be positive")
        if self.logs_per_epoch < 0:
            raise ValueError("logs_per_epoch must not be negative")

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        for key in self.__dict__.keys():
            if key in input_config.keys():
                if key == "negative_sampling":
                    if self.negative_sampling is MISSING:
                        self.negative_sampling = NegativeSamplingConfig()
                    self.negative_sampling.merge(input_config.negative_sampling)
                elif key == "pipeline":
                    if self.pipeline is MISSING:
                        self.pipeline = PipelineConfig()
                    self.pipeline.merge(input_config.pipeline)
                else:
                    val = input_config.__getattr__(key)
                    self.__setattr__(key, val)

        self.__post_init__()


@dataclass
class EvaluationConfig:
    batch_size: int = 1000
    negative_sampling: NegativeSamplingConfig = MISSING
    pipeline: PipelineConfig = PipelineConfig()
    eval_checkpoint: int = -1
    epochs_per_eval: int = 1

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        for key in self.__dict__.keys():
            if key in input_config.keys():
                if key == "negative_sampling":
                    if self.negative_sampling is MISSING:
                        self.negative_sampling = NegativeSamplingConfig()
                    self.negative_sampling.merge(input_config.negative_sampling)
                elif key == "pipeline":
                    self.pipeline.merge(input_config.pipeline)
                else:
                    val = input_config.__getattr__(key)
                    self.__setattr__(key, val)

        self.__post_init__()

@dataclass
class MariusConfig:
    model: ModelConfig = ModelConfig()
    storage: StorageConfig = StorageConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    # perform high level validation here
    # TODO, should we perform file validation here or somewhere else?
    #  We should check somewhere the self.storage.dataset.base_directory for each learning task to make sure the
    #  necessary files are present and have the correct sizes
    def __post_init__(self):
        if self.model.learning_task == LearningTask.NODE_CLASSIFICATION:
            # do node classification specific validation
            if self.storage.dataset.num_classes != self.model.encoder.output_dim:
                raise ValueError("The output dimension of the encoder must be equal to the number of class labels")

        elif self.model.learning_task == LearningTask.LINK_PREDICTION:
            # do link prediction specific validation
            pass


def type_safe_merge(base_config: MariusConfig, input_config: DictConfig):
    """
    Merges under specified dictionary config into the current configuration object
    :param input_config: The input configuration dictionary
    :return: Structured output config
    """

    if "model" in input_config.keys():
        base_config.model.merge(input_config.model)

    if "storage" in input_config.keys():
        base_config.storage.merge(input_config.storage)

    if "training" in input_config.keys():
        base_config.training.merge(input_config.training)

    if "evaluation" in input_config.keys():
        base_config.evaluation.merge(input_config.evaluation)

    base_config.__post_init__()

    return base_config


cs = ConfigStore.instance()
cs.store(name="base_config", node=MariusConfig)


def load_config(input_config_path):
    """
    This function loads an input user specified configuration file and creates a full configuration file with all
    defaults set based on the input
    :param input_config_path: path to the input configuration file
    :return: config dict object
    """
    input_cfg = None

    input_config_path = Path(input_config_path).absolute()

    config_dir = input_config_path.parent
    config_name = input_config_path.name

    # get user defined config
    with hydra.initialize_config_dir(config_dir=config_dir.__str__(), version_base="1.1"):
        input_cfg = hydra.compose(config_name=config_name)

    # merge the underspecified input configuration with the fully specified default configuration
    base_config = MariusConfig()
    output_config = type_safe_merge(base_config, input_cfg)

    # we can then perform validation, and optimization over the fully specified configuration file here before returning

    return output_config


@hydra.main(config_path="config_templates", config_name="marius_config", version_base="1.1")
def my_app(cfg: MariusConfig) -> None:

    yaml_file = OmegaConf.to_yaml(cfg)

    # filter out the options with missing values
    new_yaml_file = []
    for line in yaml_file.splitlines():
        if MISSING not in line:
            new_yaml_file.append(line)

    print("\n".join(new_yaml_file))


if __name__ == "__main__":
    print(OmegaConf.to_yaml(load_config("configs/cora.yaml")))
