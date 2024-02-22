.. _config_schema

Configuration Schema
=========================

.. list-table:: MariusConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - model
     - ModelConfig
     - Defines model architecture, learning task, optimizers and loss function.
     - Yes
   * - storage
     - StorageConfig
     - Defines the input graph and how to store the graph (edges, features) and learned model (embeddings).
     - Yes
   * - training
     - TrainingConfig
     - Hyperparameters for training.
     - Training
   * - evaluation
     - EvaluationConfig
     - Hyperparameters for evaluation.
     - Evaluation

Below is a sample end-to-end configuration file for link prediction on `fb15_237` dataset. The model consists of an embedding layer
in the encoder phase which is directly fed to the `DISTMULT` decoder. Both embeddings and edges are stored in `cpu` memory. 

.. code-block:: yaml 

   model:
     learning_task: LINK_PREDICTION
     encoder:
       layers:
         - - type: EMBEDDING
             output_dim: 50
             bias: true
             init:
               type: GLOROT_NORMAL
     decoder:
       type: DISTMULT
     loss:
       type: SOFTMAX_CE
       options:
         reduction: SUM
     dense_optimizer:
       type: ADAM
       options:
         learning_rate: 0.01
     sparse_optimizer:
       type: ADAGRAD
       options:
         learning_rate: 0.1
   storage:
     full_graph_evaluation: true
     device_type: cpu
     dataset:
       dataset_dir: /home/data/datasets/fb15k_237/
     edges:
       type: DEVICE_MEMORY
       options:
         dtype: int
     embeddings:
       type: DEVICE_MEMORY
       options:
         dtype: float
   training:
     batch_size: 1000
     negative_sampling:
       num_chunks: 10
       negatives_per_positive: 10
       degree_fraction: 0
       filtered: false
     num_epochs: 10
     pipeline:
       sync: true
     epochs_per_shuffle: 1
     logs_per_epoch: 10
     resume_training: false
   evaluation:
     batch_size: 1000
     negative_sampling:
       filtered: true
     epochs_per_eval: 1
     pipeline:
       sync: true


Model Configuration
--------------------


.. list-table:: ModelConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - random_seed
     - Int
     - Random seed used to initialize, train, and evaluate the model. If not given, a seed will be generated.
     - No
   * - learning_task
     - String
     - Learning task for which the model is used. Valid values are ["LINK_PREDICTION", "NODE_CLASSIFICATION"] (case insensitive). "LP" and "NC" can be used for shorthand.
     - Yes
   * - :ref: encoder
     - :ref:`EncoderConfig<encoder-conf-section>`
     - Defines the architecture of the encoder and configuration of neighbor samplers.
     - Yes
   * - :ref: decoder
     - :ref:`DecoderConfig<decoder-conf-section>`
     - Denotes the decoder to apply to the output of the encoder. The decoder is learning task specific.
     - Yes
   * - :ref: loss
     - :ref:`LossConfig<loss-conf-section>`
     - Loss function to apply over the output of the decoder.
     - Required for training
   * - dense_optimizer
     - :ref:`OptimizerConfig<optimizer-conf-section>`
     - Optimizer to use for dense model parameters. Where dense model parameters refer to all parameters besides the node embeddings. Where node embeddings are handled by the sparse_optimizer.
     - Required for training
   * - sparse_optimizer
     - :ref:`OptimizerConfig<optimizer-conf-section>`
     - Optimizer to use for the node embedding parameters. Currently only ADAGRAD is supported.
     - No

Below is a full view of the `model` attribute and the corresponding parameters that can be set in the model configuration. It consists
of an embedding layer in the encoder phase and a `DISTMULT` decoder.

.. code-block:: yaml

   model:
     random_seed: 456356765463
     learning_task: LINK_PREDICTION
     encoder:
       layers:
         - - type: EMBEDDING
             output_dim: 50
             bias: true
             init:
               type: GLOROT_NORMAL
             optimizer:
               type: DEFAULT
               options:
                 learning_rate: 0.1
     decoder:
       type: DISTMULT
       options:
         inverse_edges: true
         use_relation_features: false
         edge_decoder_method: CORRUPT_NODE
       optimizer:
         type: ADAGRAD
         options:
           learning_rate: 0.1
     loss:
       type: SOFTMAX_CE
       options:
         reduction: SUM
     dense_optimizer:
       type: ADAM
       options:
         learning_rate: 0.01
     sparse_optimizer:
       type: ADAGRAD
       options:
         learning_rate: 0.1

.. _encoder-conf-section:

Encoder Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: EncoderConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - use_incoming_nbrs
     - Boolean
     - Whether to use incoming neighbors for the encoder. One of use_incoming_nbrs or use_outgoing_nbrs must be set to true.
     - No
    * - use_outgoing_nbrs
     - Boolean
     - Whether to use outgoing neighbors for the encoder. One of use_incoming_nbrs or use_outgoing_nbrs must be set to true.
     - No
   * - layers
     - List[List[:ref:`LayerConfig<layer-conf-section>`]]
     - Defines architecture of the encoder. Layers of the encoder are grouped into stages, where the layers within a stage are executed in parallel and the output of stage is the input to the successive stage.
     - Yes
   * - train_neighbor_sampling
     - List[:ref:`NeighborSamplingConfig<neighbor-sampling-conf-section>`]
     - Sets the neighbor sampling configuration for each GNN layer for training (and evaluation if eval_neighbor_sampling is not set). Defined as a list of neighbor sampling configurations, where the size of the list must match the number of GNN layers in the encoder.
     - Only for GNNs
   * - eval_neighbor_sampling
     - List[:ref:`NeighborSamplingConfig<neighbor-sampling-conf-section>`]
     - Sets the neighbor sampling configuration for each GNN layer for evaluation. Defined as a list of neighbor sampling configurations, where the size of the list must match the number of GNN layers in the encoder. If this field is not set then the sampling configuration used for training will be used for evaluation.
     - No

The below example depicts a configuration where there is one embedding layer, followed by three GNN layers.  

.. code-block:: yaml

   encoder:
     train_neighbor_sampling:
       - type: ALL
       - type: ALL
       - type: ALL
     eval_neighbor_sampling:
       - type: ALL
       - type: ALL
       - type: ALL
     layers:
       - - type: EMBEDDING
           output_dim: 10
           bias: true
           init:
             type: GLOROT_NORMAL

       - - type: GNN
           options:
             type: GAT
           input_dim: 10
           output_dim: 10
           bias: true
           init:
             type: GLOROT_NORMAL

       - - type: GNN
           options:
             type: GAT
           input_dim: 10
           output_dim: 10
           bias: true
           init:
             type: GLOROT_NORMAL

       - - type: GNN
           options:
             type: GAT
           input_dim: 10
           output_dim: 10
           bias: true
           init:
             type: GLOROT_NORMAL


.. _neighbor-sampling-conf-section:

.. list-table:: NeighborSamplingConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - Denotes the type of the neighbor sampling layer. Options: ["ALL", "UNIFORM", "DROPOUT"].
     - Yes
   * - options
     - NeighborSamplingOptions
     - Specific options depending on the type of sampling layer.
     - No


.. list-table:: UniformSamplingOptions[NeighborSamplingOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - max_neighbors
     - Int
     - Number of neighbors to sample in a given uniform sampling layer.
     - Yes

The below configuration might work for a graph configuration where there are 2 GNN layers. The configuration specifies that at most 
10 neighboring nodes will be samples for any given node embedding during training.

.. code-block:: yaml 

   train_neighbor_sampling:
     - type: UNIFORM
       options:
         max_neighbors: 10
     - type: UNIFORM
       options:
         max_neighbors: 10


.. list-table:: DropoutSamplingOptions[NeighborSamplingOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - rate
     - Float
     - The dropout rate for a dropout layer.
     - Yes

`DROPOUT` mode neighbor sampling randomly drops `rate * 100` percent neighbors during sampling. 

.. code-block:: yaml 

   train_neighbor_sampling:
     - type: DROPOUT
       options:
         rate: 0.05


.. _layer-conf-section:

Layer Configuration
"""""""""""""""""""

.. list-table:: LayerConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - Denotes the type of layer. Options: ["EMBEDDING", "FEATURE", "GNN" "REDUCTION"]
     - Yes
   * - options
     - LayerOptions
     - Layer specific options depending on the type.
     - No
   * - input_dim
     - Int
     - The dimension of the input to the layer.
     - GNN and Reduction layers
   * - output_dim
     - Int
     - The output of dimension of the layer.
     - Yes
   * - init
     - :ref:`InitConfig<init-conf-section>`
     - Initialization method for the layer parameters. (Default GLOROT_UNIFORM).
     - No
   * - optimizer
     - OptimizerConfig
     - Optimizer to use for the parameters of this layer. If not given, the dense_optimizer is used.
     - No
   * - bias
     - Bool
     - Enable a bias to be applied to the output of the layer. (Default False)
     - No
   * - bias_init
     - :ref:`InitConfig<init-conf-section>`
     - Initialization method for the bias. The default initialization is zeroes.
     - No
   * - activation
     - String
     - Activation function to apply to the output of the layer. Options ["RELU", "SIGMOID", "NONE"]. (Default "NONE")
     - No

Below is a configuration for creating and embedding layer with output dimension 50. It is initialized with zeros and has no activation 
set.

.. code-block:: yaml

   layers:
   - - type: EMBEDDING
       input_dim: -1
       output_dim: 50
       init:
         type: GLOROT_NORMAL
       optimizer:
         type: DEFAULT
         options:
           learning_rate: 0.1
       bias: true
       bias_init:
         type: ZEROS
       activation: NONE


A GNN layer of type GAT (Graph Attention) with input and output dimension of 50 is as follows.

.. code-block:: yaml 

   layers:
   - - type: GNN
       options:
         type: GAT
       input_dim: 50
       output_dim: 50
       bias: true
       init:
         type: GLOROT_NORMAL


A Reduction layer of type Linear, with input dimension of 100 and output dimension of 50 is as follows. 

.. code-block:: yaml

   layers:
   - - type: REDUCTION
       input_dim: 100
       ouptut_dim: 50
       bias: true
       options:
         type: LINEAR


Below is a simple Feature layer with output dimension of 50. The input dimension is set to -1 by default since both Feature and 
Embedding layers do not have any input. 

.. code-block:: yaml

   layers:
   - - type: FEATURE
       output_dim: 50
       bias: true


Layer Options
"""""""""""""

**GNN Layer Options**

.. list-table:: GraphSageLayerOptions[LayerOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - The type of the GNN layer, for GraphSage, this must be equal to "GRAPH_SAGE".
     - Yes
   * - aggregator
     - String
     - Aggregation to use for graph sage, options are ["GCN", "MEAN"]. (Default "MEAN")
     - No

A GNN layer of type `GRAPH_SAGE` with aggregator set to `MEAN`. Another possbile option is `GCN` (Graph Convolution).

.. code-block:: yaml

   - - type: GNN
       options:
         type: GRAPH_SAGE
         aggregator: MEAN


.. list-table:: GATLayerOptions[LayerOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - The type of the GNN layer, for GAT, this must be equal to "GAT".
     - Yes
   * - num_heads
     - Int
     - Number of attention heads to use. (Default 10)
     - No
   * - average_heads
     - Bool
     - If true, the attention heads will be averaged, otherwise they will be concatenated. (Default True)
     - No
   * - negative_slope
     - Float
     - Negative slope to use for LeakyReLU. (Default .2)
     - No
   * - input_dropout
     - Float
     - Dropout rate to apply to the input to the layer. (Default 0.0)
     - No
   * - attention_dropout
     - Float
     - Dropout rate to apply to the attention weights. (Default 0.0)
     - No

A GNN layer of type `GAT` (Graph Attention) with 50 attention heads. `input_dropout` is set to 0.1 implying that 10 percent of the 
input tensor values will be randomly dropped.

.. code-block:: yaml

   - - type: GNN
       options:
         type: GAT
         num_heads: 50
         average_heads: True
         input_dropout: 0.1


**Reduction Layer Options**

.. list-table:: ReductionLayerOptions[LayerOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - The type of the reduction layer. Options are: ["CONCAT", "LINEAR"]. (Default "CONCAT")
     - Yes

A reduction layer of type `LINEAR`. Another possible type for the reduction layer is `CONCAT`.

.. code-block:: yaml

   - - type: REDUCTION
       options:
         type: LINEAR


.. _init-conf-section:

Initialization Configuration
""""""""""""""""""""""""""""

.. list-table:: InitConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - The type of the initialization. Options are: ["GLOROT_UNIFORM", "GLOROT_NORMAL", "UNIFORM", "NORMAL", "ZEROES", "ONES", "CONSTANT"]. Default "GLOROT_UNIFORM"
     - Yes
   * - options
     - InitOptions
     - Initialization specific options depending on the type.
     - No

.. code-block:: yaml

   init:
     type: GLOROT_NORMAL
     options: {}


**Uniform Init Options**

.. list-table:: UniformInitOptions[InitOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - scale_factor
     - Float
     - The scale factor of the uniform distribution. (Default 1)
     - No

The below configuration is used to initialize a layer with a uniform distribution of values ranging between [-scale_factor, +scale_factor]

.. code-block:: yaml

   init:
     type: UNIFORM
     options:
       scale_factor: 1


**Normal Init Options**

.. list-table:: NormalInitOptions[InitOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - mean
     - Float
     - The mean of the distribution. (Default 0.0)
     - No
   * - std
     - Float
     - The standard deviation of the distribution. (Default 1.0)
     - No

The below configuration is used to initialize a layer with values belonging to a noraml distribution, with mean 0.5 and standard 
deviation 0.1.

.. code-block:: yaml

   init:
     type: NORMAL
     options:
       mean: 0.5
       std: 0.1


**Constant Init Options**

.. list-table:: ConstantInitOptions[InitOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - constant
     - Float
     - The value to set all parameters. (Default 0.0)
     - No

`CONSTANT` initialization mode initializes all parameters of the layer to the specified constant value. 

.. code-block:: yaml

   init:
     type: CONSTANT
     options:
       constant: 0.4

.. _decoder-conf-section:

Decoder Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: DecoderConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - Denotes the type of decoder. Options: ["DISTMULT", "TRANSE", "COMPLEX", "NODE"]. The first three are decoders for link prediction and the "NODE" decoder is used for node classification.
     - Yes
   * - options
     - DecoderOptions
     - Decoder specific options depending on the type.
     - No
   * - optimizer
     - OptimizerConfig
     - Optimizer to use for the parameters of the decoder (if any). If not given, the dense_optimizer is used.
     - No

Below is a `DISTMULT` decoder with Adagrad Optimizer, that optimizes the loss function over edges as well as their inverses (dest->rel->src).

.. code-block:: yaml

   decoder:
     type: DISTMULT
     options:
       inverse_edges: true
     optimizer:
       type: ADAGRAD
       options:
         learning_rate: 0.1


Decoder Options
""""""""""""""""

**Edge Decoder Options**

.. list-table:: EdgeDecoderOptions[DecoderOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - inverse_edges
     - Bool
     - If true, the decoder will use two embeddings per edge-type (relation). Where one embedding is applied to the source node of an edge, and the other is applied to the destination node of an edge. Furthermore, the scores of the inverse of the edges will be computed (dst->rel->src) and used in the loss. (Default True)
     - No
   * - edge_decoder_method
     - String
     - Specifies how to apply the decoder to a given set of edges, and negatives. Options are ["infer", "train"]. (Default "train")
     - No

.. code-block:: yaml

   decoder:
     type: DISTMULT
     options:
       inverse_edges: true
       edge_decoder_method: CORRUPT_NODE


.. _loss-conf-section:

Loss Configuration
^^^^^^^^^^^^^^^^^^

.. list-table:: LossConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - Denotes the type of the loss function. Options: ["SOFTMAX_CE", "RANKING", "CROSS_ENTROPY", "BCE_AFTER_SIGMOID", "BCE_WITH_LOGITS", "MSE", "SOFTPLUS"].
     - Yes
   * - options
     - LossOptions
     - Loss function specific options depending on the type.
     - No

Below is the configuration for a `SOFTMAX_CE` loss function with `SUM` as the reduction method.

.. code-block:: yaml

   loss:
     type: SOFTMAX_CE
     options:
       reduction: SUM


**Loss Options**

.. list-table:: LossOptions
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - reduction
     - String
     - The reduction to use for the loss. Options are ["SUM", "MEAN"]. (Default "SUM")
     - No

Below is the configuration for a `SOFTMAX_CE` loss function with `MEAN` as the reduction method.

.. code-block:: yaml

   loss:
     type: SOFTMAX_CE
     options:
       reduction: MEAN


.. list-table:: RankingLossOptions[LossOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - reduction
     - String
     - The reduction to use for the loss. Options are ["SUM", "MEAN"]. (Default "SUM")
     - No
   * - margin
     - Float
     - The margin for the ranking loss function. (Default .1)
     - No

Below is the configuration for a `RANKING` loss function with `margin` set to 1. 

.. code-block:: yaml

   loss:
     type: RANKING
     options:
       reduction: SUM
       margin: 1


.. _optimizer-conf-section:

Optimizer Configuration
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: OptimizerConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - Denotes the type of the optimizer. Options: ["SGD", "ADAM", "ADAGRAD"].
     - Yes
   * - options
     - OptimizerOptions
     - Optimizer specific options depending on the type.
     - No

The configuration for an `ADAGRAD` optimizer with learning rate of 0.1 is as follows

.. code-block:: yaml

   optimizer:
     type: ADAGRAD
     options:
       learning_rate: 0.1


**SGD Options**

.. list-table:: SGDOptions[OptimizerOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - learning_rate
     - Float
     - SGD learning rate. (Default .1)
     - No

.. code-block:: yaml

   optimizer:
     type: SGD
     options:
       learning_rate: 0.1


**Adagrad Options**

.. list-table:: AdagradOptions[OptimizerOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - learning_rate
     - Float
     - Adagrad learning rate. (Default .1)
     - No
   * - eps
     - Float
     - Term added to the denominator to improve numerical stability. (Default 1e-10)
     - No
   * - init_value
     - Float
     - Initial accumulator value. (Default 0.0)
     - No
   * - lr_decay
     - Float
     - Learning rate decay. (Default 0.0)
     - No
   * - weight_decay
     - Float
     - Weight decay (L2 penalty). (Default 0.0)
     - No

The below configuration shows the options that can be set for `ADAGRAD` optimizer.

.. code-block:: yaml

   optimizer:
     type: ADAGRAD
     options:
       learning_rate: 0.1
       eps: 1.0e-10
       init_value: 0.0
       lr_decay: 0.0
       weight_decay: 0.0


**Adam Options**

.. list-table:: AdamOptions[OptimizerOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - learning_rate
     - Float
     - Adam learning rate. (Default .1)
     - No
   * - amsgrad
     - Bool
     - Whether to use the AMSGrad variant of ADAM.
     - No
   * - beta_1
     - Float
     - Coefficient used for computing running averages of gradient and its square. (Default .9)
     - No
   * - beta_2
     - Float
     - Coefficient used for computing running averages of gradient and its square. (Default .999)
     - No
   * - eps
     - Float
     - Term added to the denominator to improve numerical stability. (Default 1e-8)
     - No
   * - weight_decay
     - Float
     - Weight decay (L2 penalty). (Default 0.0)
     - No

The below configuration shows the options that can be set for `ADAM` optimizer.

.. code-block:: yaml

   optimizer:
     type: ADAM
     options:
       learning_rate: 0.01
       amsgrad: false
       beta_1: 0.9
       beta_2: 0.999
       eps: 1.0e-08
       weight_decay: 0.0


Storage Configuration
----------------------

.. list-table:: StorageConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - device_type
     - String
     - Whether to use cpu or gpu training. Options are ["CPU", "CUDA"]. (Default "CPU")
     - No
   * - dataset
     - DatasetConfig
     - Contains information about the input dataset.
     - Yes
   * - edges
     - StorageBackendConfig
     - Storage backend of the edges. (Default edges.type = DEVICE_MEMORY, edges.options.dtype = int32)
     - No
   * - embeddings
     - StorageBackendConfig
     - Storage backend of the node embedding. (Default embeddings.type = DEVICE_MEMORY, embeddings.options.dtype = float32)
     - No
   * - features
     - StorageBackendConfig
     - Storage backend of the node features. (Default features.type DEVICE_MEMORY, features.options.dtype = float32)
     - No
   * - prefetch
     - Bool
     - If true and the nodes/features storage configuration uses a partition buffer, then node partitions and edge buckets will be prefetched. Note that this introduces additional memory overheads. (Default True)
     - No
   * - full_graph_evaluation
     - Bool
     - If true and the nodes/features storage configuration uses a partition buffer, evaluation will be performed with the full graph in memory (if there is enough memory). This is useful for fair comparisons across different storage configurations. (Default False)
     - No
   * - model_dir
     - String
     - Saves the model parameters in the given directory. If not specified, stores in `model_x` directory within the `dataset_dir` where x changes incrementally from 0 - 10. A maximum of 11 models are stored when `model_dir` is not specified, post which the contents in `model_10/` directory are overwritten with the latest parameters.
     - No

Below is a storage configuration that contains the path to the pre-processed data and specifies storage backends to be used for edges, features 
and embeddings.

.. code-block:: yaml 

   storage:
     device_type: cpu
     dataset:
       dataset_dir: /home/data/datasets/fb15k_237/
     edges:
       type: DEVICE_MEMORY
       options:
         dtype: int
     nodes:
       type: DEVICE_MEMORY
       options:
         dtype: int
     embeddings:
       type: DEVICE_MEMORY
       options:
         dtype: float
     features:
       type: DEVICE_MEMORY
       options:
         dtype: float
     prefetch: true
     shuffle_input: true
     full_graph_evaluation: true
     export_encoded_nodes: true
     log_level: info


Dataset Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: DatasetConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - dataset_dir
     - String
     - Directory containing the prepreprocessed dataset. Also used to store model parameters and embedding table.
     - Yes
   * - num_edges
     - Int
     - Number of edges in the input graph. If link prediction, this should be set to the number of training edges.
     - No
   * - num_nodes
     - Int
     - Number of nodes in the input graph.
     - No
   * - num_relations
     - Int
     - Number of relations (edge-types) in the input graph. (Default 1)
     - No
   * - num_train
     - Int
     - Number of training examples. In link prediction the examples are edges, in node classification they are nodes.
     - No
   * - num_valid
     - Int
     - Number of validation examples. If not given, no validation will be performed
     - No
   * - num_test
     - Int
     - Number of test examples. If not given, only training will occur.
     - No (Evaluation)
   * - node_feature_dim
     - Int
     - Dimension of the node features, if any.
     - No
   * - num_classes
     - Int
     - Number of class labels.
     - No (Node classification)

For Marius in-built datasets, the below numbers are retrieved from output of `marius_preprocess`. For custom user datasets, a 
file with the dataset statistics mentioned above should be present in the `dataset_dir`. Below is the cofiguration for the `fb15k_237` dataset. 

.. code-block:: yaml 

   storage:
     dataset:
       dataset_dir: /home/data/datasets/fb15k_237/


Storage Backend Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: StorageBackendConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - type
     - String
     - The type of storage backend to use. The valid options depend on the data being stored. For edges, the valid backends are ["FLAT_FILE", "HOST_MEMORY" and "DEVICE_MEMORY"]. For embeddings and features, the valid chocies are ["PARTITION_BUFFER", "HOST_MEMORY", "DEVICE_MEMORY"]
     - Yes
   * - options
     - StorageOptions
     - Storage backend options depending on the type of storage.
     - No

Below configuration specifies that the edges be stored in `DEVICE_MEMORY`, i.e CPU/GPU memory based on `device_type`.

.. code-block:: yaml

   edges:
     type: DEVICE_MEMORY
     options:
       dtype: int


Storage Backend Options
"""""""""""""""""""""""

.. list-table:: StorageOptions
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - dtype
     - String
     - The datatype of the storage. Valid options ["FLOAT", "FLOAT32", "DOUBLE", "FLOAT64", "INT", "INT32", "LONG, "INT64"]. The default value depends on the data being stored. For edges, the default is "INT32", otherwise the default is "FLOAT32"
     - No

A configuration defining the datatype of the input edges as `int`.

.. code-block:: yaml

   edges:
     options:
       dtype: int


.. list-table:: PartitionBufferOptions[StorageOptions]
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - dtype
     - String
     - The datatype of the storage. Valid options ["FLOAT", "FLOAT32", "DOUBLE", "FLOAT64"]. (Default "FLOAT32")
     - No
   * - num_partitions
     - Int
     - Number of node partitions.
     - Yes
   * - buffer_capacity
     - Int
     - Number of partitions which can fit in the buffer.
     - Yes
   * - prefetching
     - Bool
     - If true, partitions will be prefetched and written to storage asynchronously. This prevents IO wait times at the cost of additional memory overheads. (Default True)
     - No

Below is a disk-based storage configuration, where at max of `buffer_capacity` embeddings buckets are stored in memory at any given time. 
The dataset must be partitioned using `marius_preprocess` with `--num_partitions` set accordingly. 

.. code-block:: yaml

   embeddings:
     type: PARTITION_BUFFER
     options:
       dtype: float
       num_partitions: 10
       buffer_capacity: 5
       prefetching: true


Training Configuration
-----------------------

.. list-table:: TrainingConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - batch_size
     - Int
     - Amount of training examples per batch. (Default 1000)
     - No
   * - negative_sampling
     - NegativeSamplingConfig
     - Negative sampling configuration for link prediction.
     - Link Prediction
   * - num_epochs
     - Int
     - Number of epochs to train.
     - Yes
   * - pipeline
     - PipelineConfig
     - Advanced configuration of the training pipeline. Defaults to synchronous training.
     - No
   * - epochs_per_shuffle
     - Int
     - Sets how often to shuffle the training data. (Default 1)
     - No
   * - logs_per_epoch
     - Int
     - Sets how often to report progress during an epoch. (Default 10)
     - No
   * - save_model
     - Bool
     - If true, the model will be saved at the end of training. (Default True)
     - No
   * - resume_training
     - Bool
     - If true, the training procedure will resume from the previous state and will train `num_epochs` further epochs.  (Default False)
     - No
   * - resume_from_checkpoint
     - String
     - If set, loads the model from the given directory and resumes training procedure. Will train `num_epochs` further epochs and store the new model parameters in `model_dir`.
     - No

A training configuration with batchsize of 1000 and a total of 10 epochs is as follows. `pipeline` is set to true, which ensures that 
the training is synchronous and doesn't allow staleness. Marius groups edges into chunks and reuses negative samples within the chunk. 
`num_chunks`*`negatives_per_positive` negative edges are sampled for each positive edge.

.. code-block:: yaml

   training:
     batch_size: 1000
     negative_sampling:
       num_chunks: 10
       negatives_per_positive: 10
       degree_fraction: 0.0
       filtered: false
     num_epochs: 10
     pipeline:
       sync: true
     epochs_per_shuffle: 1
     logs_per_epoch: 10
     save_model: true
     resume_training: false


Evaluation Configuration
-------------------------

.. list-table:: EvaluationConfig
   :widths: 15 10 50 15
   :header-rows: 1

   * - Key
     - Type
     - Description
     - Required
   * - batch_size
     - Int
     - Amount of evaluation examples per batch. (Default 1000)
     - No
   * - negative_sampling
     - NegativeSamplingConfig
     - Negative sampling configuration for link prediction.
     - Link Prediction
   * - pipeline
     - PipelineConfig
     - Advanced configuration of the evaluation pipeline. Defaults to synchronous evaluation.
     - No
   * - epochs_per_eval
     - Int
     - Sets how often to evaluate the model. (Default 1)
     - No

An evaluation configuration with batchsize of 1000 is as follows. `num_chunks`*`negatives_per_positive` negative edges are sampled 
for each positive edge.

.. code-block:: yaml

   evaluation:
     batch_size: 1000
     negative_sampling:
       num_chunks: 1
       negatives_per_positive: 1000
       degree_fraction: 0.0
       filtered: true
     pipeline:
       sync: true
     epochs_per_eval: 1