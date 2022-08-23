# MariusGNN EuroSys 2023 Artifact Source Code Overview #

In this file, we describe the high level overview of the MariusGNN source code (`src/`) used for the EuroSys 2023 
artifact.

## Table of Contents ##
*   [Recommended Reading Order](#recommended-reading-order)
*   [Layout](#layout)
*   [Important Files](#important-files)
    * [marius](#marius)
    * [configuration](#configuration)
    * [graph_storage](#graph_storage)
    * [dataloader](#dataloader)
    * [model](#model)
    * [trainer](#trainer)
    * [graph](#graph)
    * [graph_samplers](#graph_samplers)
    * [buffer](#buffer)
    * [ordering](#ordering)



## Recommended Reading Order

The C++ sources (`src/cpp/`) contain the bulk of the MariusGNN logic), but are quite extensive, and it’s best to go 
through the code in a top-down approach, first understanding the code at a high level and then going through the 
detailed implementation as needed. Here we describe a recommended reading order for the C++. Note that dataset
preprocessing and configuration file parsing is handled in Python (`src/python/tools/`).

Top-down C++ reading order:

1. `marius.cpp`
    
    1.  This is the main entrypoint to the system and defines the program execution at the highest level.
        
2. `SynchronousTrainer::train()` in `trainer.cpp`
    
    1. The synchronous trainer defines the execution of one training epoch.

    2. The synchronous trainer is much simpler to read than the pipelined trainer, but the pipelined trainer performs the exact same operations (get batch, host to device transfer, train batch, device to host transfer, and update batch). In the pipeline, each operation is handled by a set of workers and bounded queues separate each worker (described below).
        
4. `Dataloader::getBatch()` in `dataloader.cpp`
    
    1.  This function defines how to fill a batch with edges, nodes, and embeddings.
        
    2.  If the task is link prediction, `linkPredictionSample` will be called which will sample edges, negatives, and neighbors for the given batch.
        
    3.  If the task is node classification, `nodeClassificationSample` will be called which will sample nodes and neighbors for the given batch.
        
    4.  After the batch has been filled with edges/nodes, the embeddings/features for each node will be loaded with `loadCPUParameters` if the data is stored in CPU memory, or `loadGPUParameters` if the data is stored in GPU memory.
        
5. `LinkPredictionModel::train_batch()` in `model.cpp`
    
    1.  This performs the forward and backward pass of the model.
        
    2.  For link prediction, the model is made up of an encoder (GNN) which takes the embeddings/features of the neighbors for a batch of nodes and then outputs new embeddings for those nodes. These embeddings are then passed to the decoder which will compute the scores for each edge in the batch. The encoder is optional and the model may be decoder only (e.g. DistMult).
        
    3.  For node classification, the model contains only an encoder, which takes the features of the neighbors for the nodes and outputs a label vector for each node in the batch. This label vector is then compared in the loss function with the true label.
        

After following this high level reading approach, you should be familiar with the basic structure of the code, 
an epoch of training, and how batches are formed.



## Layout

Here is the directory layout of the C++ headers/sources:

```plain
/src/cpp/
    include/                        # Contains project header files
        configuration/              # C++ configuration file management
            config.h                # Defines config objects and Python to C++ config conversion
            constants.h             # Contains constants for data storage paths
            options.h               # Configuration enums and option classes
        decoders/               
            comparators.h           # Distance functions to compute scores between embeddings
            relation_operators.h    # Operator functions for combining node embeddings with relation embeddings
            decoder.h               # Decoder class and forward pass 
            complex.h               # Defines ComplEx decoder and initialization
            distmult.h              # Defines DistMult decoder and initialization
            transe.h                # Defines TransE decoder and initialization
        encoders/
            gnn.h                   # General GNN class, is composed of arbitrary GNN layers 
        featurizers/
            featurizer.h            # Featurizer class, defines how to combine node features and node embeddings
        layers/
            gat_layer.h             # Defines initialization and forward pass for GAT
            gcn_layer.h             # Defines initialization and forward pass for GCN
            gnn_layer.h             # Abstract GNN layer class 
            graph_sage_layer.h      # Defines initialization and forward pass for GraphSage
            layer_helpers.h         # Helper methods for segmented operations
            rgcn_layer.h            # Defines initialization and forward pass for RGCN
        activation.h                # Activation functions 
        batch.h                     # Defines the Batch class, which contains edges, embeddings and other metadata for a training or evaluation batch
        buffer.h                    # Defines the PartitionBuffer--used for storing graph partitions in memory during out-of-core training
        dataloader.h                # Contains Dataloader class which provides an interface to get batches
        datatypes.h                 # Contains typedefs and other useful headers used project-wide 
        evaluator.h                 # Abstract Evaluator class and both a synchronous and pipelined evalautor
        graph.h                     # Graph class in CSR format, and a graph class for fast GNN encoding
        graph_samplers.h            # Edge, negative, and neighborhood samplers
        graph_storage.h             # GraphModelStorage object which contains storage for edges, the embedding table, and features
        initialization.h            # Initialization functions for initializing both large-scale and small-scale tensors
        io.h                        # Initializes storage objects for the learning task
        logger.h                    # Logger class with multiple log level streams
        loss.h                      # Loss function definitions
        marius.h                    # Main entrypoint for the code, sets up storage, model and trainer/evaluators
        model.h                     # Model class which contains a featurizer, encoder, decoder and loss function. Defines forward pass for node classification and link prediction models
        model_helpers.h             # Helper methods for cloning models to multiple devices (GPUs)
        ordering.h                  # Contains BETA/COMET ordering and node classification orderings
        pipeline.h                  # Defines GPU and CPU pipelines for training and evaluation 
        regularizer.h               # Regularization functions 
        reporting.h                 # Extensible reporting module for recording evaluation statistics and tracking epoch progress
        storage.h                   # Storage class definitions for PartitionBuffer, FlatFile, HostMemory, and DeviceMemory storage
        trainer.h                   # Synchronous and pipelined trainer definitions
        util.h                      # Misc utility functions and classes
    src/                            # Implementation for the headers, structure matches the include directory almost exactly
    python_bindings/                # Python bindings for C++ classes and functions
    third_party/                    # Third party libaries used within Marius
```



## Important Files
The .cpp/.h files with the following names contain core pieces of MariusGNN;

### marius
This is the entrypoint to the system for the `marius_train` and `marius_eval` executables. The function `void marius(int argc, char *argv[])` operates at a high level to define the training/evaluation process. The flow of this function is as follows:

1.  Parse the input configuration file: `initConfig`
    
2.  Initialize the global model (GNN and edge type parameters): `initializeModel`
    
3.  Initialize edges, nodes, and embedding table storage from preprocessed graph stored on disk: `initializeStorage`
    
4.  Create dataloader over the storage, which will provide batches of embeddings and edges: `DataLoader`
    
5.  Setup and run `Trainer` and `Evaluator` classes, which will perform the training and evaluation for a specified number of epochs
    

This function can be viewed as an example usage of the Marius C++ API as it operates at a high level using the main API 
objects: Model, GraphModelStorage, DataLoader, and Trainer/Evaluator. So any user defined programs using the C++ or 
Python API will largely follow these 5 steps, where each of the 5 steps can be extended or customized by 
extending/override the API objects.

### configuration
Marius is driven by configuration objects which define nearly everything about program execution.

These files define the configuration schema and how to convert python configuration objects into the corresponding C++ 
configuration objects. Python object definitions map to the C++ definitions one-to-one. 
Python config objects are in `src/python/tools/configuration/marius_config.py`.

Take the TrainingConfig struct in this file as an example:

```cpp
struct TrainingConfig {
    int batch_size;
    shared_ptr<NegativeSamplingConfig> negative_sampling;
    int num_epochs;
    shared_ptr<PipelineConfig> pipeline;
    int epochs_per_shuffle;
    int logs_per_epoch;
};
```

This struct has a corresponding python object:

```py
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
```

The C++ configuration performs no validation or setting of default values, this is all performed in Python, 
e.g. the `__post_init__(self)` method.

The translation between Python and C++ configuration objects happens in `configuration.cpp`. 
For the above config class the following function performs the conversion:

```cpp
shared_ptr<TrainingConfig> initTrainingConfig(pyobj python_config) {
    shared_ptr<TrainingConfig> ret_config = std::make_shared<TrainingConfig>();

    ret_config->batch_size = cast_helper<int>(python_config.attr("batch_size"));
    ret_config->negative_sampling = initNegativeSamplingConfig(python_config.attr("negative_sampling"));
    ret_config->pipeline = initPipelineConfig(python_config.attr("pipeline"));
    ret_config->logs_per_epoch = cast_helper<int>(python_config.attr("logs_per_epoch"));
    ret_config->epochs_per_shuffle = cast_helper<int>(python_config.attr("epochs_per_shuffle"));
    ret_config->num_epochs = cast_helper<int>(python_config.attr("num_epochs"));

    return ret_config;
}
```

The conversion process requires casting from a pybind object (pyobj) to the appropriate datatype.

### graph_storage
This class manages storage and access for the edges, nodes, features, embeddings, and embedding optimizer state for 
the current configuration. This class is backed by a set of pointers to abstract tensor storage objects (`storage.cpp`), 
where the storage backend can be independently set for each storage object.

```cpp
struct GraphModelStoragePtrs {
    Storage *edges;                         // Current edges storage used by the dataloader, can be switched between the train/valid/test set
    Storage *edge_features;                 // Features for the edges (currently not supported)
    Storage *train_edges;                   // The set of edges for training, sorted by the source node
    Storage *train_edges_dst_sort;          // The set of edges for training, sorted by the destination node 
    Storage *train_edges_features;          
    Storage *validation_edges;              // Set of edges for validation (link prediction)
    Storage *validation_edges_features; 
    Storage *test_edges;                    // Set of edges for testing (link prediction)
    Storage *test_edges_features;
    Storage *nodes;                         // Current nodes storage used by the dataloader (node classification)
    Storage *train_nodes;                   // Node ids used for training (node classification)
    Storage *valid_nodes;                   // Node ids used for validation (node classification)
    Storage *test_nodes;                    // Node ids user for testing (node classification)
    Storage *node_features;                 // Features for each node
    Storage *node_labels;                   // Labels for each node (node classification)
    Storage *relation_features;             // Features for each edge-type (currently not supported)
    Storage *relation_labels;               // Labels for each edge-type (currently not supported)
    Storage *node_embeddings;               // Embeddings for each node
    Storage *node_optimizer_state;          // Optimizer state for each node embedding
};
```

By using this abstract storage interface we can easily store the node embeddings on the GPU, CPU or on disk with the 
partition buffer, independently of how we store the edges. Further, we can implement the storage interface using a 
variety of backends (NVM, Remote storage, S3, etc.) to support more deployment scenarios.

When using the partition buffer to store the embeddings, this class (graph_storage) also maintains an 
**in-memory subgraph**. This subgraph contains all edges between the nodes which currently have their embeddings 
in the partition buffer. This subgraph is structured in CSR format, with support for fast neighborhood 
lookups and sampling.

### dataloader
The dataloader provides a batching interface over the edges/nodes of the graph 
(depending on if using link prediction or node classification).

The dataloader contains samplers, which define how to iterate over the edges, sample negatives, and sampler neighbors 
for any given batch. When `getBatch` is called, the samplers will fill the batch with edges, 
negatives, and neighbors and then load any node embeddings or features for the nodes that were sampled. 

For disk-based training, once `getBatch` has exhausted all the training examples associated with the partitions in
memory, it will trigger a partition swap to load new graph data from disk into CPU memory. It will then prepare start
creating batches from the new training data associated with the new in-memory subgraph.

### model
The Model class defines initialization and forward pass for each task and is composed of the following modules:

Inputs and outputs are defined on a node level here, but in the code the modules operate at a batch level.

*   Featurizer
    
    *   Input: Node feature, Node embedding
        
    *   Output: Featurized embedding
        
    *   Role: Defines how to combine node features with node embeddings (not used in the artifact).
        
*   Encoder (GNN)
    
    *   Input: Node, Neighborhood, Embeddings/Features/FeaturizedEmbeddings
        
    *   Output: New embedding for the node
        
    *   Role: Encodes a node based on its embedding/feature and the embeddings/features of its neighbors.
        
*   Decoder
    
    *   Input: Edge (s, r, d), node embeddings th\_s, th\_d. Relation embedding th\_r.
        
    *   Output: Score for the edge
        
    *   Role: The decoder computes scores for input edges based on the embeddings of the nodes and the edge-types.
        
*   Loss function
    
    *   Input: scores/labels
        
    *   Output: loss value
        
    *   Role: Computes the loss function for the learning tasks. Ranking based loss functions are used for link prediction, while traditional classification losses are used for node classification.
        

These modules are computed sequentially so the execution flow is:

Batch → Featurizer → Encoder → Decoder → Loss

Note that these modules are optional depending on the desired model, and the Featurizer has not been implemented yet.
If a user only wants to train DistMult, this is a decoder only model so the execution flow is:

Batch → Decoder → Loss

If a user wants to train a node classification model the flow is:

Batch → Encoder → Loss

Each module is a `torch::nn::module` which provides support to save/load and clone models and their parameters.

### trainer
The Trainer class defines one epoch of training and the iteration for each batch.

**SynchronousTrainer**

The synchronous trainer performs the training process synchronously. It’s best to understand the trainer by 
directly reading the code. Here is a slightly simplified and annotated version of the train method:

```cpp
void SynchronousTrainer::train(int num_epochs) {

    dataloader_->setTrainSet(); // Need to tell the dataloader we want to iterate over the training set
    dataloader_->loadStorage(); // Load edges/embeddings/features into memory (or partition buffer)

    for (int epoch = 0; epoch < num_epochs; epoch++) {
    
        // Iterate over all batches
        while (dataloader_->hasNextBatch()) {

            // Get edges/embeddings/features and places them into a batch object
            Batch *batch = dataloader_->getBatch();

            // Send the batch to the GPU the model resides on 
            batch->to(model_->current_device_);

            // Compute forward and backward pass of the model
            model_->train_batch(batch);
            
            // Accumulates node embedding gradients and applies optimizer rule (e.g. Adagrad)
            batch->accumulateGradients(model_->embedding_learning_rate);
            
            // Transfer gradients for batch back to the CPU
            batch->embeddingsToHost();

            // Update node embedding table
            dataloader_->updateEmbeddingsForBatch(batch);
        }
        
        // Notify the dataloader that the epoch has been completed prepare for another epoch
        dataloader_->nextEpoch();
    }
    dataloader_->unloadStorage(true); // Unload the edges/embeddings/features. The parameter true denotes that we should write back the embedding table to storage.
}
```

**PipelinedTrainer**

The pipelined trainer performs the same training process as the synchronous trainer above, but in a pipelined manner. 
I.e. the main operations in an iteration use multiple workers and place output batches onto queues which separate 
the workers. The structure of the GPU pipeline is as follows:

```cpp
LoadEmbeddingsWorker
{
    IF DATALOADER HAS A BATCH AND OUR PIPELINE HAS NOT REACHED CAPACITY (STALENESS BOUND)
    
    Batch *batch = dataloader_->getBatch();
    
    PUSH TO NEXT QUEUE
}

// QUEUE OF BATCHES

H2DTransferWorker
{
    PULL FROM PRECEEDING QUEUE
    
    batch->to(model_->current_device_);
    
    PUSH TO NEXT QUEUE
}

// QUEUE OF BATCHES

GPUComputeWorker 
{
    PULL FROM PRECEEDING QUEUE

    model_->train_batch(batch);
    batch->accumulateGradients(model_->embedding_learning_rate);
    
    PUSH TO NEXT QUEUE
}

// QUEUE OF BATCHES

D2HTransferWorker
{
    PULL FROM PRECEEDING QUEUE
    
    batch->embeddingsToHost();
    
    PUSH TO NEXT QUEUE
}

// QUEUE OF BATCHES

UpdateEmbeddingsWorker
{
    PULL FROM PRECEEDING QUEUE
    
    dataloader_->updateEmbeddingsForBatch(batch);
    
    SIGNAL TO PIPELINE THAT A BATCH HAS LEFT THE PIPELINE
}
```

These are the default pipeline options which set the capacity of the pipeline, the number of workers for each worker 
type, and the size of each queue.

```java
staleness_bound: int = 16               // Denotes how many batches can be in the pipeline at once
batch_host_queue_size: int = 4          // Size of the queue between Load/H2DTransfer    (ON CPU)
batch_device_queue_size: int = 4        // Size of the queue between H2DTransfer/Compute (ON GPU)
gradients_device_queue_size: int = 4    // Size of the queue between Compute/D2HTransfer (ON GPU)
gradients_host_queue_size: int = 4      // Size of the queue between D2HTransfer/Update  (ON CPU)
batch_loader_threads: int = 4           // Number of loading workers
batch_transfer_threads: int = 2         // Number of H2D transfer workers
compute_threads: int = 1                // Number of GPU compute workers, we only use 1 per GPU. For CPU training this can be arbitrarly increased.
gradient_transfer_threads: int = 2      // Number of D2H transfer workers
gradient_update_threads: int = 4        // Number of update workers 
```

### graph
These files contain a number of methods and classes related to the **DENSE** data structure and neighborhood sampling. 

`MariusGraph` stores the two sorted versions of the edge list (required for single-hop sampling) for the in-memory 
subgraph. During disk-based training, this subgraph is updated after each partition swap (according to 
`updateInMemorySubGraph_` in graph_storage.cpp). For in-memory training, the MariusGraph edge lists contain all edges
in the graph.

`MariusGraph::getNeighborsForNodeIds()` performs the single-hop sampling for a set of node IDs.

`GNNGraph` stores the DENSE data structure output from multi-hop neighborhood sampling.

### graph_samplers
These files contain graph samplers used to create mini batches for training.

In particular, `LayeredNeighborSampler::getNeighbors()` samples multi-hop neighbors for a set of node IDs and returns 
the DENSE data structure.

### buffer
The `PartitionBuffer` class is responsible for storing node partitions in CPU memory during disk-based training. It
also handles swapping partitions from disk to CPU memory.

### ordering
The `ordering.cpp` file contains the partition replacement and training example selection policies used by MariusGNN
(e.g., COMET (called Two_Level_Beta in the code)). These policies decide what partitions will be loaded into memory 
and in what order for each epoch and  which training examples will be used to create batches for each in-memory 
subgraph. Orderings are used by the `PartitionBuffer` class
to decided which partitions to move between disk and CPU memory during each swap.