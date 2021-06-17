.. _configuration

*************
Configuration
*************

Executing Marius requires the specification of a .ini format configuration file.
The configuration file is divided up into nine total sections, with multiple configuration options per section.

Example Configuration File
--------------------------

The following is a simple configuration file example for the Freebase15k dataset.

In the ``[general]`` section, the user defines the device type they wish to train on as well as statistics about the input dataset.

The ``[model]`` section defines the properties of the model. In this case we wish to train 100-dimensional embeddings using the ComplEx scoring function.

The ``[training]`` section defines training hyperparameters. Here we set the batch size to be 1000 edges, and use 512 negative samples per edge to compute the loss.

The ``[evaluation]`` section designates how to evaluate the trained embeddings. Here we use LinkPrediction as the evaluation task and sample 1000 negatives per edge to produce the output rank of each edge in the evaluation set.

The ``[path]`` section is used to point to the paths of the preprocessed dataset and where the program data is stored. By setting the ``base_directory`` to ``fb15k_experiment\`` the embeddings and program data will be stored within that directory.

::

    [general]
    device=GPU
    num_train=483142
    num_nodes=14951
    num_relations=1345
    num_valid=50000
    num_test=59071

    [model]
    embedding_size=100
    decoder=ComplEx

    [training]
    batch_size=1000
    negatives=512

    [evaluation]
    negatives=1000
    evaluation_method=LinkPrediction

    [path]
    base_directory=fb15k_experiment/
    train_edges=output_dir/train_edges.pt
    validation_edges=output_dir/valid_edges.pt
    test_edges=output_dir/test_edges.pt

Modifying Options From Command Line
-----------------------------------

Configuration options can also be passed in from the command line. Any parameter passed in the command line will override the value specified in the configuration file.

We can modify the configuration by passing ``--<section>.<key>=<value>`` to the end of the ``marius_train`` command.

For example, let ``fb15k_config.ini`` refer to the config file shown above. The following command will train embeddings with the DistMult scoring function instead of ComplEx.

::

    marius_train fb15k_config.ini --model.decoder=DistMult

Sections
------------

[general]
^^^^^^^^^

The general section contains options for specifying the dataset statistics, execution device, and other miscellaneous options.

===============  ======  ========  =======  =============  =============
   Name          Type    Required  Default  Valid Values   Description
---------------  ------  --------  -------  -------------  -------------
device           string  No        CPU      [CPU, GPU]     The device type to use
gpu_ids          list    No        0                       The ids of the gpus to use. Specified a space separated list. E.g. ``gpu_ids=0 1 2 3``
random_seed      int     No        time(0)                 Random seed used to generate initial embeddings and samples from the graph.
num_train        long    Yes                               Number of edges in the training set
num_valid        long    Yes                               Number of edges in the validation set
num_test         long    Yes                               Number of edges in the test set
num_nodes        int     Yes                               Number of nodes in the graph
num_relations    int     Yes                               Number of relations (edge-types) in the graph
experiment_name  string  No        marius                  Name of the current experiment. Program data and embeddings are stored in ``<path.base_directory>/<general.experiment_name>``
===============  ======  ========  =======  =============  =============

[model]
^^^^^^^^

The model section defines the properties of the model for training and evaluation.

===========================  ======  ========  ========  ===================================  =============
   Name                      Type    Required  Default   Valid Values                         Description
---------------------------  ------  --------  --------  -----------------------------------  -------------
scale_factor                 float   No        .001                                           Factor used to scale the initialization distribution
initialization_distribution  string  No        Normal    [Uniform, Normal]                    Distribution to initialize node embedding vectors from, scaled by the scale_factor
embedding_size               int     No        128                                            Embedding vector dimension for node and relation embeddings
encoder                      string  No        None      [None]                               GNN Encoder to use. Currently unsupported, but coming soon!
decoder                      string  No        DistMult  [TransE, DistMult, ComplEx, Custom]  Decoder scoring function to use.
===========================  ======  ========  ========  ===================================  =============

[storage]
^^^^^^^^^

The storage section defines how embeddings and edges are stored.

===========================  ======  ========  ===========  ==========================================================================================================  ===================
   Name                      Type    Required  Default      Valid Values                                                                                                Description
---------------------------  ------  --------  -----------  ----------------------------------------------------------------------------------------------------------  -------------------
edges_backend                string  No        HostMemory   [DeviceMemory, HostMemory, FlatFile]                                                                        Specifies the location in which the edges will be stored.
embeddings_backend           string  No        HostMemory   [DeviceMemory, HostMemory, PartitionBuffer]                                                                 Specifies the location in which the node embeddings will be stored.
relations_backend            string  No        HostMemory   [DeviceMemory, HostMemory]                                                                                  Specifies the location in which the relation embeddings will be stored.
edges_dtype                  string  No        int32        [int32, int64]                                                                                              Datatype of the edge list. If there are less than 2 billion nodes (which is almost every dataset), int32 should be used.
embeddings_dtype             string  No        float32      [float32, float64]                                                                                          Datatype of the embedding vectors.
reinit_edges                 bool    No        true                                                                                                                     If true, the edges will be reinitialized from the files specified in the path config section. If false, the system will use the edges located in the base_directory, assuming that they have been previously initialized.
remove_preprocessed          bool    No        false                                                                                                                    If true, the input edge files specified in the path config section will be deleted.
shuffle_input_edges          bool    No        true                                                                                                                     If true, the input edge files will be shuffled before being input to the system.
reinit_embeddings            bool    No        true                                                                                                                     If true, the embedding table will be initialized, overwriting any previous embedding data in the base_directory. This should be set to false if the user wishes to train more epochs on previously trained embeddings, or if the user wishes to evaluate the previously trained embeddings.
edge_bucket_ordering         string  No        Elimination  [Elimination, Hilbert, HilbertSymmetric, Random, RandomSymmetric, Sequential, SequentialSymmetric, Custom]  Sets the order in which each edge bucket is processed, see edge bucket orderings for more details. (Only used for the PartitionBuffer embedding backend)
num_partitions               int     No        1                                                                                                                        Sets the number of node partitions. (Only used for the PartitionBuffer embedding backend)
buffer_capacity              int     No        2                                                                                                                        Sets how many node partitions can fit in the buffer. (Only used for the PartitionBuffer embedding backend)
prefetching                  bool    No        true                                                                                                                     If set to true, the partition buffer will use async IO and prefetching of node partitions. (Only used for the PartitionBuffer embedding backend)
conserve_memory              bool    No        false                                                                                                                    Reduces memory consumption of shuffling operations between epochs at the cost of extra IO.
===========================  ======  ========  ===========  ==========================================================================================================  ===================


[training]
^^^^^^^^^^

The training section allows for setting training hyperparameters.

===========================  ======  ========  ==========  ===========================================  ===================
   Name                      Type    Required  Default     Valid Values                                 Description
---------------------------  ------  --------  ----------  -------------------------------------------  -------------------
batch_size                   int     No        10000                                                    The number of edges in each batch.
number_of_chunks             int     No        16                                                       Tunes the amount of reuse of the sampled negatives. See negative sampling for more details.
negatives                    int     No        512                                                      The number of negative edges that should be used per positive edge when computing the loss.
degree_fraction              float   No        .5                                                       The fraction of the negative samples that are sampled proportional to degree, where the rest are sampled uniformly from the graph.
negative_sampling_access     string  No        Uniform      [Uniform, UniformCrossPartition]            This parameter is only used for the PartitionBuffer backend. If set to Uniform, all uniform negative samples will be produced from within the same node partitions as the source and destination nodes of a batch. Setting to UniformCrossPartition will sample from all partitions currently in the buffer, which may result in higher quality embeddings.
learning_rate                float   No        .1                                                       Sets the learning rate of the optimizer.
regularization_coef          float   No        2e-6                                                     Coefficient to scale the regularization loss.
regularization_norm          int     No        2                                                        Norm of the regularization.
optimizer                    string  No        Adagrad      [Adagrad]                                   Currently Adagrad is the only supported optimizer.
average_gradients            bool    No        false                                                    If true, the gradients will be averaged when accumulating gradients for a batch. If false, the gradients will be summed.
synchronous                  bool    No        false                                                    If true, the training will be performed synchronously without use of the training pipeline. If false, the training pipeline will be used. If embedding data is stored in HostMemory or the PartitionBuffer, synchronous training will be slow due to data movement wait times.
num_epochs                   int     No        10                                                       The number of epochs to train to.
checkpoint_interval          int     No        9999                                                     Determines how many epochs should complete before checkpointing the embedding parameters. By default this is set to 9999, a large number which is used to effectively disable checkpointing. Checkpoints are stored in ``<base_directory>/<experiment_name>/embeddings/embeddings_<epoch_id>.bin`` and ``<base_directory>/<experiment_name>/relations/embeddings_<epoch_id>.bin``
shuffle_interval             int     No        1                                                        Determines how many epochs should complete before the edges are shuffled. If set to 1, the edges will be shuffled after every epoch.
===========================  ======  ========  ==========  ===========================================  ===================


.. _loss_option:

[loss]
^^^^^^

The loss section allows for setting loss function options. 

===========================  ======  ========  ==========  ============================================================================================================================================================================================================  ===================
   Name                      Type    Required  Default     Valid Values                                                                                                                                                                                                  Description
---------------------------  ------  --------  ----------  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  -------------------
loss                         string  No        SoftMax      [:ref:`SoftMax<loss_functions>`, :ref:`Ranking<loss_functions>`, :ref:`BCEAfterSigmoid<loss_functions>`, :ref:`BCEWithLogits<loss_functions>`, :ref:`MSE<loss_functions>`, :ref:`SoftPlus<loss_functions>`]  Sets the loss function. The Ranking loss can be tuned with the margin parameter.
margin                       float   No        0                                                                                                                                                                                                                         Sets the margin for the Ranking loss function
reduction                    string  No        Mean         [Mean, Sum]                                                                                                                                                                                                  Sets the reduction to apply to the loss
===========================  ======  ========  ==========  ============================================================================================================================================================================================================  ===================                                                      


[training_pipeline]
^^^^^^^^^^^^^^^^^^^

The training pipeline section is for advanced users who wish to maximize the throughput of the asynchronous training pipeline and/or limit the amount of asynchronicity to improve model accuracy.

==============================  ======  ========  ==========  ===========================================  ===================
   Name                         Type    Required  Default     Valid Values                                 Description
------------------------------  ------  --------  ----------  -------------------------------------------  -------------------
max_batches_in_flight           int     No        16                                                       Sets the maximum number of batches allowed to be in the training pipeline at the same time. Large values may improve training throughput at the cost of model accuracy due to staleness of embedding parameters.
embeddings_host_queue_size      int     No        4                                                        Sets the capacity of the queue on the host machine which stages batches for transfer to the device (if using a GPU for computation) or for preparation for computation (if using CPUs for computation).
embeddings_device_queue_size    int     No        4                                                        Sets the capacity of the queue on the device which contains the batches that have been transferred from the host. (Only used for GPU computation)
gradients_host_queue_size       int     No        4                                                        Sets the capacity of the queue on the device which stages gradients updates for batches to be transferred back to the host. (Only used for GPU computation)
gradients_device_queue_size     int     No        4                                                        Sets the capacity of the queue on the host which contains the gradient updates for batches, which will then be applied to storage.
num_embedding_loader_threads    int     No        2                                                        Number of threads used to load embeddings from storage.
num_embedding_transfer_threads  int     No        2                                                        Number of threads to used transfer batches from host to device.
num_compute_threads             int     No        1                                                        Number of threads used to perform forward and backward pass. Should be set to number of GPUs if using GPU computation.
num_gradient_transfer_threads   int     No        2                                                        Number of threads to used transfer gradient updates from device to host.
num_embedding_update_threads    int     No        2                                                        Number of threads used to apply gradient updates to storage.
==============================  ======  ========  ==========  ===========================================  ===================


[evaluation]
^^^^^^^^^^^^

This section sets the configuration for the evaluation of embeddings.

===========================  ======  ========  ==============  ===========================================  ===================
   Name                      Type    Required  Default         Valid Values                                 Description
---------------------------  ------  --------  --------------  -------------------------------------------  -------------------
batch_size                   int     No        1000                                                         The number of edges in each batch.
number_of_chunks             int     No        1                                                            Tunes the amount of reuse of the sampled negatives. See negative sampling for more details.
negatives                    int     No        1000                                                         The number of negative edges that should be used per positive edge during evaluation.
degree_fraction              float   No        .5                                                           The fraction of the negative samples that are sampled proportional to degree, where the rest are sampled uniformly from the graph.
negative_sampling_access     string  No        Uniform         [Uniform, All]                               If set to All, the negatives parameter will be ignored and all nodes in the graph will be used to produce negatives.
synchronous                  bool    No        false                                                        If true, the evaluation will be performed synchronously without use of the evaluation pipeline. If false, the evaluation pipeline will be used. If embedding data is stored in HostMemory or the PartitionBuffer, synchronous evaluation will be slow due to data movement wait times.
epochs_per_eval              int     No        1                                                            Determines how many epochs will complete before evaluation on the validation set is performed. Setting to 1 will evaluate the embeddings after every epoch.
evaluation_method            string  No        LinkPrediction  [LinkPrediction, NodeClassification]         Sets which evaluation method should be used.
filtered_negatives           bool    No        false                                                        Setting to true requires setting negative_sampling_access=All and will filter out any false negatives that are produced by the negative sampling.
===========================  ======  ========  ==============  ===========================================  ===================

[evaluation_pipeline]
^^^^^^^^^^^^^^^^^^^^^

The evaluation pipeline section is for advanced users who wish to maximize the throughput of the asynchronous evaluation pipeline. The defaults should work well for 99.9% of use cases.

==============================  ======  ========  ==========  ===========================================  ===================
   Name                         Type    Required  Default     Valid Values                                 Description
------------------------------  ------  --------  ----------  -------------------------------------------  -------------------
max_batches_in_flight           int     No        32                                                       Sets the maximum number of batches allowed to be in the evaluation pipeline at the same time. Unlike the training pipeline we can allow as many batches as we need into the pipeline since we are not updating embeddings during the evaluation, and hence there are no stale embeddings.
embeddings_host_queue_size      int     No        8                                                        Sets the capacity of the queue on the host machine which stages batches for transfer to the device (if using a GPU for computation) or for preparation for computation (if using CPUs for computation).
embeddings_device_queue_size    int     No        8                                                        Sets the capacity of the queue on the device which contains the batches that have been transferred from the host. (Only used for GPU computation)
num_embedding_loader_threads    int     No        4                                                        Number of threads used to load embeddings from storage.
num_embedding_transfer_threads  int     No        4                                                        Number of threads to used transfer batches from host to device.
num_evaluate_threads            int     No        1                                                        Number of threads used to perform forward and backward pass. Should be set to number of GPUs used for computation if using GPU computation
==============================  ======  ========  ==========  ===========================================  ===================


[path]
^^^^^^^^

This section is used to denote the location of the preprocessed files for the input dataset.

==============================  ======  ========  ==========  ===========================================  ===================
   Name                         Type    Required  Default     Valid Values                                 Description
------------------------------  ------  --------  ----------  -------------------------------------------  -------------------
base_directory                  string  No        data/                                                    Path to where Marius should store program data and embeddings for experiments: ``<path.base_directory>/<general.experiment_name>``
train_edges                     string  No                                                                 Path to preprocessed training edges file
train_edges_partitions          string  No                                                                 Path to file which denotes the sizes of the edge buckets in the training edges file
validation_edges                string  No                                                                 Path to preprocessed validation edges file
validation_partitions           string  No                                                                 Path to file which denotes the sizes of the edge buckets in the validation edges file
test_edges                      string  No                                                                 Path to preprocessed test edges file
test_edges_partitions           string  No                                                                 Path to file which denotes the sizes of the edge buckets in the test edges file
node_labels                     string  No                                                                 Path to the file which contains the labels of the nodes.
relation_labels                 string  No                                                                 Path to fhe file which contains the labels of the relations (edge-types).
node_ids                        string  No                                                                 Path to the file which contains the ids of nodes.
relations_ids                   string  No                                                                 Path to the file which contains the ids of relations.
custom_ordering                 string  No                                                                 Path to a file which explicitly defines the ordering in which edge buckets should be processed. Used with ``storage.edge_bucket_ordering=Custom``
==============================  ======  ========  ==========  ===========================================  ===================

[reporting]
^^^^^^^^^^^

This section is used to set reporting configuration options.

==============================  ======  ========  ==========  ===========================================  ===================
   Name                         Type    Required  Default     Valid Values                                 Description
------------------------------  ------  --------  ----------  -------------------------------------------  -------------------
logs_per_epoch                  int     No        10                                                       Sets how often Marius should report progress during training. Setting to 10 means that 10 progress updates will be given during an epoch.
log_level                       string  No        info        [info, debug, trace]                         Sets the log level of the console logger.
==============================  ======  ========  ==========  ===========================================  ===================


Full Default Configuration File
-------------------------------

Here we show the defaults for each configuration options in .ini format.

``#`` is used to denote configuration options that do not have defaults or their default value is set programmatically.

::

    [general]
    device=CPU
    gpu_ids=0

    # defaults to using "time(0)" as random seed if not specified
    #random_seed

    #num_train
    #num_nodes
    #num_relations
    #num_valid
    #num_test

    experiment_name=marius

    [model]
    scale_factor=.001
    initialization_distribution=Normal
    embedding_size=128
    encoder=None
    decoder=DistMult

    [storage]
    edges_backend=HostMemory
    reinit_edges=true
    remove_preprocessed=false
    shuffle_input_edges=true
    edges_dtype=int32
    embeddings_backend=HostMemory
    reinit_embeddings=true
    relations_backend=HostMemory
    embeddings_dtype=float32
    edge_bucket_ordering=Elimination
    num_partitions=1
    buffer_capacity=2
    prefetching=true
    conserve_memory=false

    [training]
    batch_size=10000
    number_of_chunks=16
    negatives=512
    degree_fraction=.5
    negative_sampling_access=Uniform
    learning_rate=.1
    regularization_coef=2e-6
    regularization_norm=2
    optimizer=Adagrad
    loss=SoftMax
    margin=0
    average_gradients=false
    synchronous=false
    num_epochs=10

    # large number used to effectively disable checkpointing
    checkpoint_interval=9999
    shuffle_interval=1

    [training_pipeline]
    max_batches_in_flight=16
    embeddings_host_queue_size=4
    embeddings_device_queue_size=4
    gradients_host_queue_size=4
    gradients_device_queue_size=4
    num_embedding_loader_threads=2
    num_embedding_transfer_threads=2
    num_compute_threads=1
    num_gradient_transfer_threads=2
    num_embedding_update_threads=2

    [evaluation]
    batch_size=1000
    number_of_chunks=1
    negatives=1000
    degree_fraction=.5
    negative_sampling_access=Uniform
    epochs_per_eval=1
    evaluation_method=LinkPrediction
    filtered_evaluation=false

    [evaluation_pipeline]
    max_batches_in_flight=32
    embeddings_host_queue_size=8
    embeddings_device_queue_size=8
    num_embedding_loader_threads=4
    num_embedding_transfer_threads=4
    num_evaluate_threads=1

    [path]
    # The following do not have defaults
    # train_edges
    # train_edges_partitions
    # validation_edges
    # validation_partitions
    # test_edges
    # test_edges_partitions
    # node_labels
    # relation_labels
    # node_ids
    # relations_ids
    # custom_ordering

    base_directory=data/

    [reporting]
    logs_per_epoch=10
    log_level=info