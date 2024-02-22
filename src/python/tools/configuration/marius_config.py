import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from distutils import dir_util
from pathlib import Path
from typing import List

from omegaconf import MISSING, DictConfig, OmegaConf

from marius.tools.configuration.constants import PathConstants
from marius.tools.configuration.datatypes import (
    AdagradOptions,
    AdamOptions,
    ConstantInitOptions,
    DecoderOptions,
    DenseLayerOptions,
    DropoutSamplingOptions,
    EdgeDecoderOptions,
    GATLayerOptions,
    GNNLayerOptions,
    GraphSageLayerOptions,
    InitOptions,
    LayerOptions,
    LossOptions,
    NeighborSamplingOptions,
    NormalInitOptions,
    OptimizerOptions,
    PartitionBufferOptions,
    RankingLossOptions,
    ReductionLayerOptions,
    StorageOptions,
    UniformInitOptions,
    UniformSamplingOptions,
)
from marius.tools.configuration.validation import (
    check_encoder_layer_dimensions,
    check_full_graph_evaluation,
    check_gnn_layers_alignment,
    validate_dataset_config,
    validate_storage_config,
)


def get_model_dir_path(dataset_dir):
    # will support storing upto 11 different model params when model_dir is not specified.
    # post that, it will overwrite in <dataset_dir>/model_10 directory.
    for i in range(11):
        model_dir = "{}/model_{}".format(dataset_dir, i)
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            return str(model_dir_path)

    return str(model_dir_path)


@dataclass
class NeighborSamplingConfig:
    type: str = "ALL"
    options: NeighborSamplingOptions = NeighborSamplingOptions()
    use_hashmap_sets: bool = False

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = NeighborSamplingOptions()

        if self.type == "UNIFORM":
            new_options = UniformSamplingOptions()

        if self.type == "DROPOUT":
            new_options = DropoutSamplingOptions()

        if "options" in input_config.keys():
            for key in new_options.__dict__.keys():
                if key in input_config.options.keys():
                    val = input_config.options.__getattr__(key)
                    new_options.__setattr__(key, val)

        self.options = new_options

        if "use_hashmap_sets" in input_config.keys():
            self.use_hashmap_sets = input_config.use_hashmap_sets


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

        if self.type == "DEFAULT":
            self.options = new_options
            return

        if self.type == "ADAGRAD":
            new_options = AdagradOptions()

        if self.type == "ADAM":
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

        if self.type == "CONSTANT":
            new_options = ConstantInitOptions()

        if self.type == "UNIFORM":
            new_options = UniformInitOptions()

        if self.type == "NORMAL":
            new_options = NormalInitOptions()

        for key in new_options.__dict__.keys():
            if key in input_config.options.keys():
                val = input_config.options.__getattr__(key)
                new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class LossConfig:
    type: str = "SOFTMAX_CE"
    options: LossOptions = LossOptions()

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        new_options = LossOptions()

        if self.type == "RANKING":
            new_options = RankingLossOptions()

        if "options" in input_config.keys():
            for key in new_options.__dict__.keys():
                if key in input_config.options.keys():
                    val = input_config.options.__getattr__(key)
                    new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class LayerConfig:
    type: str = None
    options: LayerOptions = LayerOptions()
    input_dim: int = -1
    output_dim: int = -1
    init: InitConfig = InitConfig(type="GLOROT_UNIFORM")
    optimizer: OptimizerConfig = OptimizerConfig(type="DEFAULT")
    bias: bool = False
    bias_init: InitConfig = InitConfig(type="ZEROS")
    activation: str = "NONE"

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.type = input_config.type.upper()

        if "init" in input_config.keys():
            self.init.merge(input_config.init)

        if "options" in input_config.keys():
            new_options = LayerOptions()

            if self.type == "GNN":
                new_options = GNNLayerOptions(type="NONE")
                if input_config.options.type.upper() == "GRAPH_SAGE":
                    new_options = GraphSageLayerOptions()
                elif input_config.options.type.upper() == "GAT":
                    new_options = GATLayerOptions()

            if self.type == "DENSE":
                new_options = DenseLayerOptions()

            if self.type == "REDUCTION":
                new_options = ReductionLayerOptions()

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

        if "optimizer" in input_config.keys():
            if self.optimizer is MISSING:
                self.optimizer = OptimizerConfig()
            self.optimizer.merge(input_config.optimizer)

        if "input_dim" in input_config.keys():
            self.input_dim = input_config.input_dim

        if "output_dim" in input_config.keys():
            self.output_dim = input_config.output_dim


@dataclass
class EncoderConfig:
    use_incoming_nbrs: bool = True
    use_outgoing_nbrs: bool = True
    layers: List[List[LayerConfig]] = field(default_factory=list)
    train_neighbor_sampling: List[NeighborSamplingConfig] = field(default_factory=list)
    eval_neighbor_sampling: List[NeighborSamplingConfig] = field(default_factory=list)
    embedding_dim: int = -1

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """
        if "use_incoming_nbrs" in input_config.keys():
            self.use_incoming_nbrs = input_config.use_incoming_nbrs

        if "use_outgoing_nbrs" in input_config.keys():
            self.use_outgoing_nbrs = input_config.use_outgoing_nbrs

        new_layers = []
        if "layers" in input_config.keys():
            for stage in input_config.layers:
                new_stages = []
                for layer_config in stage:
                    base_layer = LayerConfig()
                    base_layer.merge(layer_config)
                    new_stages.append(base_layer)
                    if base_layer.type == "EMBEDDING":
                        self.embedding_dim = base_layer.output_dim

                new_layers.append(new_stages)

        self.layers = new_layers

        new_train = []
        if "train_neighbor_sampling" in input_config.keys():
            for layer_config in input_config.train_neighbor_sampling:
                base_layer = NeighborSamplingConfig()
                base_layer.merge(layer_config)
                new_train.append(base_layer)

        self.train_neighbor_sampling = new_train

        new_eval = []
        if "eval_neighbor_sampling" in input_config.keys():
            for layer_config in input_config.eval_neighbor_sampling:
                base_layer = NeighborSamplingConfig()
                base_layer.merge(layer_config)
                new_eval.append(base_layer)

        self.eval_neighbor_sampling = new_eval


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

        if self.type != "NODE":
            new_options = EdgeDecoderOptions()

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
    encoder: EncoderConfig = MISSING
    decoder: DecoderConfig = MISSING
    loss: LossConfig = MISSING
    dense_optimizer: OptimizerConfig = OptimizerConfig()
    sparse_optimizer: OptimizerConfig = OptimizerConfig()

    def __post_init__(self):
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

        if "dense_optimizer" in input_config.keys():
            self.dense_optimizer.merge(input_config.dense_optimizer)

        if "sparse_optimizer" in input_config.keys():
            self.sparse_optimizer.merge(input_config.sparse_optimizer)

        self.__post_init__()


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

        if self.type == "PARTITION_BUFFER":
            new_options = PartitionBufferOptions()

        if "options" in input_config.keys():
            for key in new_options.__dict__.keys():
                if key in input_config.options.keys():
                    val = input_config.options.__getattr__(key)
                    new_options.__setattr__(key, val)

        self.options = new_options


@dataclass
class DatasetConfig:
    dataset_dir: str = MISSING
    num_edges: int = MISSING
    num_nodes: int = MISSING
    num_relations: int = 1
    num_train: int = MISSING
    num_valid: int = -1
    num_test: int = -1
    node_feature_dim: int = -1
    rel_feature_dim: int = -1
    num_classes: int = -1
    initialized: bool = False

    def __post_init__(self):
        if not self.initialized:
            return

        edges_path = Path(self.dataset_dir) / Path("edges")
        if not edges_path.exists():
            raise ValueError("{} does not exist".format(str(edges_path)))

        train_edges_filepath = edges_path / Path("train_edges.bin")
        if not train_edges_filepath.exists():
            raise ValueError("{} does not exist".format(str(train_edges_filepath)))

        nodes_path = Path(self.dataset_dir) / Path("nodes")
        node_mapping_filepath = nodes_path / Path("node_mapping.txt")
        if node_mapping_filepath.exists():
            num_lines = int(os.popen("wc -l {}".format(node_mapping_filepath)).read().lstrip().split(" ")[0])
            if num_lines != self.num_nodes:
                raise ValueError(
                    "Expected to see {} lines in file {}, but found {}".format(
                        self.num_nodes, str(node_mapping_filepath), num_lines
                    )
                )

        relation_mapping_filepath = edges_path / Path("relation_mapping.txt")
        if relation_mapping_filepath.exists():
            num_lines = int(os.popen("wc -l {}".format(relation_mapping_filepath)).read().lstrip().split(" ")[0])
            if num_lines != self.num_relations:
                raise ValueError(
                    "Expected to see {} lines in file {}, but found {}".format(
                        self.num_relations, str(relation_mapping_filepath), num_lines
                    )
                )

    def populate_dataset_stats(self):
        if self.dataset_dir is MISSING:
            raise ValueError("Path to pre-processed dataset directory <dataset_dir> not found")

        dataset_dir_path = Path(self.dataset_dir)
        if not dataset_dir_path.exists():
            raise ValueError("Path specified as dataset_dir ({}) does not exist".format(str(dataset_dir_path)))

        dataset_stats_path = Path(self.dataset_dir) / Path("dataset.yaml")
        dataset_stats_path = dataset_stats_path.absolute()
        if not dataset_stats_path.exists():
            raise ValueError(
                "{} does not exist, expected to see dataset.yaml file in {} generated by marius_preprocess".format(
                    str(dataset_stats_path), self.dataset_dir
                )
            )

        dataset_cfg = OmegaConf.load(dataset_stats_path)

        keys = self.__dict__.keys()
        for key in dataset_cfg.keys():
            if key in keys:
                val = dataset_cfg.__getattr__(key)
                self.__setattr__(key, val)

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        self.initialized = True
        for key in self.__dict__.keys():
            if key in input_config.keys():
                val = input_config.__getattr__(key)
                self.__setattr__(key, val)

        self.populate_dataset_stats()

        self.__post_init__()


@dataclass
class StorageConfig:
    device_type: str = "cpu"
    device_ids: List[int] = field(default_factory=list)
    dataset: DatasetConfig = DatasetConfig()
    edges: StorageBackendConfig = StorageBackendConfig(options=StorageOptions(dtype="int"))
    nodes: StorageBackendConfig = StorageBackendConfig(options=StorageOptions(dtype="int"))
    embeddings: StorageBackendConfig = StorageBackendConfig(options=StorageOptions(dtype="float"))
    features: StorageBackendConfig = StorageBackendConfig(options=StorageOptions(dtype="float"))
    prefetch: bool = True
    shuffle_input: bool = True
    full_graph_evaluation: bool = True
    export_encoded_nodes: bool = False
    model_dir: str = MISSING
    log_level: str = "info"
    train_edges_pre_sorted: bool = False

    SUPPORTED_EMBEDDING_BACKENDS = ["PARTITION_BUFFER", "DEVICE_MEMORY", "HOST_MEMORY"]
    SUPPORTED_EDGE_BACKENDS = ["FLAT_FILE", "DEVICE_MEMORY", "HOST_MEMORY"]
    SUPPORTED_NODE_BACKENDS = ["DEVICE_MEMORY", "HOST_MEMORY"]

    def __post_init__(self):
        if self.embeddings.type not in self.SUPPORTED_EMBEDDING_BACKENDS:
            raise ValueError(
                "Storage type for embeddings should be one of PARTITION_BUFFER, DEVICE_MEMORY or HOST_MEMORY"
            )

        if self.edges.type not in self.SUPPORTED_EDGE_BACKENDS:
            raise ValueError("Storage type for edges should be one of FLAT_FILE, DEVICE_MEMORY or HOST_MEMORY")

        if self.nodes.type not in self.SUPPORTED_NODE_BACKENDS:
            raise ValueError("Storage type for nodes should be one of DEVICE_MEMORY or HOST_MEMORY")

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

        if "model_dir" in input_config.keys():
            self.model_dir = input_config.model_dir
        else:
            self.model_dir = get_model_dir_path(self.dataset.dataset_dir)

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

        if "full_graph_evaluation" in input_config.keys():
            self.full_graph_evaluation = input_config.full_graph_evaluation

        if "export_encoded_nodes" in input_config.keys():
            self.export_encoded_nodes = input_config.export_encoded_nodes

        self.__post_init__()

        if "log_level" in input_config.keys():
            self.log_level = input_config.log_level

        if "train_edges_pre_sorted" in input_config.keys():
            self.train_edges_pre_sorted = input_config.train_edges_pre_sorted


@dataclass
class NegativeSamplingConfig:
    num_chunks: int = 1
    negatives_per_positive: int = 1000
    degree_fraction: float = 0
    filtered: bool = False
    local_filter_mode: str = "DEG"

    def __post_init__(self):
        # for filtered mrr, the sampling class members should be ignored
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        if self.negatives_per_positive <= 0 and self.negatives_per_positive != -1:
            raise ValueError("negatives_per_positive must be positive or -1 if using all nodes")
        if self.degree_fraction < 0:
            raise ValueError("degree_fraction must not be negative")

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

        if "local_filter_mode" in input_config.keys():
            self.local_filter_mode = input_config.local_filter_mode

        self.__post_init__()


@dataclass
class CheckpointConfig:
    save_best: bool = False
    interval: int = -1
    save_state: bool = False

    def merge(self, input_config: DictConfig):
        """
        Merges under specified dictionary config into the current configuration object
        :param input_config: The input configuration dictionary
        :return: Structured output config
        """

        if "save_best" in input_config.keys():
            self.save_best = input_config.save_best

        if "interval" in input_config.keys():
            self.interval = input_config.interval

        if "save_state" in input_config.keys():
            self.save_state = input_config.save_state


@dataclass
class PipelineConfig:
    sync: bool = True
    gpu_sync_interval: int = 16
    gpu_model_average: bool = True
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
    save_model: bool = True
    checkpoint: CheckpointConfig = CheckpointConfig()
    resume_training: bool = False
    resume_from_checkpoint: str = ""

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
                if input_config.get(key, None) is not None:
                    if key == "negative_sampling":
                        val = input_config.get("negative_sampling", MISSING)
                        if val is not MISSING:
                            if self.negative_sampling is MISSING:
                                self.negative_sampling = NegativeSamplingConfig()
                            self.negative_sampling.merge(val)
                    elif key == "pipeline":
                        if self.pipeline is MISSING:
                            self.pipeline = PipelineConfig()
                        self.pipeline.merge(input_config.pipeline)
                    elif key == "checkpoint":
                        self.checkpoint.merge(input_config.checkpoint)
                    else:
                        val = input_config.__getattr__(key)
                        self.__setattr__(key, val)

        self.__post_init__()


@dataclass
class EvaluationConfig:
    batch_size: int = 1000
    negative_sampling: NegativeSamplingConfig = MISSING
    pipeline: PipelineConfig = PipelineConfig()
    epochs_per_eval: int = 1
    checkpoint_dir: str = ""

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
                    val = input_config.get("negative_sampling", MISSING)
                    if val is not MISSING:
                        if self.negative_sampling is MISSING:
                            self.negative_sampling = NegativeSamplingConfig()
                        self.negative_sampling.merge(val)
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

    # defining this constructor prevents from re-use of old attribute values during testing.
    def __init__(self):
        self.model = ModelConfig()
        self.storage = StorageConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()

    def __post_init__(self):
        if self.model.learning_task == "NODE_CLASSIFICATION":
            # do node classification specific validation
            pass

        elif self.model.learning_task == "LINK_PREDICTION":
            # do link prediction specific validation
            pass


def type_safe_merge(base_config: MariusConfig, input_config: DictConfig):
    """
    Merges under specified dictionary config into the current configuration object
    :param base_config: The default configuration
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


def initialize_model_dir(output_config):
    relation_mapping_filepath = (
        Path(output_config.storage.dataset.dataset_dir) / Path("edges") / Path("relation_mapping.txt")
    )
    if relation_mapping_filepath.exists():
        shutil.copy(
            str(relation_mapping_filepath), "{}/{}".format(output_config.storage.model_dir, "relation_mapping.txt")
        )

    node_mapping_filepath = Path(output_config.storage.dataset.dataset_dir) / Path("nodes") / Path("node_mapping.txt")
    if node_mapping_filepath.exists():
        shutil.copy(str(node_mapping_filepath), "{}/{}".format(output_config.storage.model_dir, "node_mapping.txt"))


def infer_model_dir(output_config):
    # if `output_config.storage.model_dir` points to a path which contains saved model params file, then just use that.
    model_dir_path = Path(output_config.storage.model_dir)
    model_file_path = model_dir_path / Path("model.pt")
    if model_dir_path.exists() and model_file_path.exists():
        return

    # if model_dir is of the form `model_x/`, where x belong to [0, 10], then set model_dir to the largest
    # existing directory. If model_dir is user specified, the control would never reach here.
    # the below regex check is an additional validation step.
    if re.fullmatch(
        "{}model_[0-9]+/".format(output_config.storage.dataset.dataset_dir), output_config.storage.model_dir
    ):
        match_result = re.search(r".*/model_([0-9]+)/$", output_config.storage.model_dir)
        last_model_id = -1
        if len(match_result.groups()) == 1:
            last_model_id = int(match_result.groups()[0]) - 1

        if last_model_id >= 0:
            output_config.storage.model_dir = "{}model_{}/".format(
                output_config.storage.dataset.dataset_dir, last_model_id
            )


def load_config(input_config_path, save=False):
    """
    This function loads an input user specified configuration file and creates a full configuration file with all
    defaults set based on the input
    :param input_config_path: path to the input configuration file
    :param save: If true, the full configuration file will be saved to <dir_of_input_config>/full_config.yaml
    :return: config dict object
    """
    input_config_path = Path(input_config_path).absolute()
    input_cfg = OmegaConf.load(input_config_path)

    # merge the underspecified input configuration with the fully specified default configuration
    base_config = MariusConfig()
    output_config = type_safe_merge(base_config, input_cfg)

    if output_config.storage.dataset.dataset_dir[-1] != "/":
        output_config.storage.dataset.dataset_dir = output_config.storage.dataset.dataset_dir + "/"

    if output_config.storage.model_dir[-1] != "/":
        output_config.storage.model_dir += "/"

    if output_config.training.resume_from_checkpoint != "" and output_config.training.resume_from_checkpoint[-1] != "/":
        output_config.training.resume_from_checkpoint += "/"

    if save and (output_config.training.resume_from_checkpoint != "" or not output_config.training.resume_training):
        # create model_dir when
        # 1. training from scratch [NOT resuming training]
        # 2. resume_training mode, with resume_from_checkpoint specified.
        Path(output_config.storage.model_dir).mkdir(parents=True, exist_ok=True)
        initialize_model_dir(output_config)

        OmegaConf.save(output_config, output_config.storage.model_dir + PathConstants.saved_full_config_file_name)

        # incase of resuming training, copy files from resume_from_checkpoint to the new folder.
        if output_config.training.resume_from_checkpoint != "":
            dir_util.copy_tree(output_config.training.resume_from_checkpoint, output_config.storage.model_dir)

    else:
        # this path is taken in test cases where random configs are passed to this function for parsing.
        # could also be taken when marius_predict is run or marius_train is run with resume_training set to true,
        # but resume_from_checkpoint isn't specified (it will then overwrite the model_dir with new model)
        infer_model_dir(output_config)

    # we can then perform validation, and optimization over the fully specified configuration file here before returning
    validate_dataset_config(output_config)
    validate_storage_config(output_config)
    check_encoder_layer_dimensions(output_config)
    check_gnn_layers_alignment(output_config)
    check_full_graph_evaluation(output_config)

    return output_config
