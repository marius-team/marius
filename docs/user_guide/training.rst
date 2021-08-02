.. _training:

****************
Training
****************

Training process
-----------------

Configuring training parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Negative Sampling
^^^^^^^^^^^^^^^^^

Two-sided relation embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-GPU training
------------------

Under development.

Configuring dataset and embedding storage
---------------------------------------------------

Marius uses configurable storage across the full memory hierarchy (disk/cpu/gpu) for efficient training on datasets of all scales on varying hardware deployments. Best practices for datasets of varying scale are given here.

For the following examples, assume Marius is deployed on the AWS p3.2xLarge instance with 16 GB of GPU memory, 64 GB of CPU memory, and a large (TB+) EBS volume.

Edge and embedding storage overheads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The overheads of storing and training each dataset is calculated as follows:
For training d-dimensional graph embeddings with n nodes, r edge-types and e edges:

* Overhead of storing node embedding parameters + optimizer state: N = 2 * n * d * 4 bytes
* Overhead of edge-type embedding parameters + optimizer state: R = 2 * r * d * 4 bytes
* Overhead of storing edges: E = e * 3 * 4 bytes (with int32 node ids)

The sum of these overheads is the total overhead of training: T = N + R + E

Note the extra factor of two to store the embedding parameters, this is because training is done using a stateful optimizer such as Adagrad, as it empirically yields much better quality embeddings.

Small Scale Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By small scale, we mean graphs for which embeddings can be trained entirely in GPU memory. For our given hardware deployment, we have 16 GB of GPU memory. A graph that is small scale will have a total overhead T that is less than 16 GB.

For small scale graphs we store edges and parameters in GPU memory and use synchronous training.

Small graph example:
n = 1 million,  e = 100 million , r = 1000, and d = 100.

* N = 2 * 1 million * 100 * 4 bytes = 800MB
* R = 2 * 1000 * 100 * 4 bytes = .8 MB
* E = 100 million * 3 * 4 bytes = 1.2 GB

We can see that the total overhead is only about 2GB.
This will fit just fine in GPU memory.
Therefore all can be stored in GPU memory with the ``DeviceMemory`` backend.

Recommended configuration options
+++++++++++++++++++++++++++++++++

::

    [storage]
    edges_backend=DeviceMemory
    embeddings_backend=DeviceMemory
    relations_backend=DeviceMemory

    [training]
    synchronous=true


Medium Scale Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By medium scale, we mean graphs for which the size of the embedding table exceeds GPU memory capacity. For our given hardware deployment, we have 64 GB of CPU memory. A graph that is medium scale will have a total overhead T that is less than 64 GB.

For these medium scale graphs we store parameters in CPU memory and use asynchronous pipelining to maximize utilization of the GPU.

Medium scale graph example:
n = 50 million,  e = 1 billion , r = 10000, and d = 100.

* N = 2 * 50 million * 100 * 4 bytes = 40GB
* R = 2 * 10000 * 100 * 4 bytes = 8 MB
* E = 1 billion * 3 * 4 bytes = 12 GB

The total overhead is about 52GB.

For this graph, the edges and node embeddings should be stored in CPU memory using the ``HostMemory`` backend. The relation embedding parameters are small and can be kept in GPU memory.

Recommended configuration options
+++++++++++++++++++++++++++++++++

Because node embedding parameters are stored off-GPU memory, we recommend using asynchronous training for best training times. We have observed little to no loss in accuracy from using asynchronous training, which we discuss in our paper.

::

    [storage]
    edges_backend=HostMemory
    embeddings_backend=HostMemory
    relations_backend=DeviceMemory

    [training]
    synchronous=false


Large Scale Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By large scale, we mean graphs for which the size of the embedding table exceeds CPU memory capacity and must use partitioning and disk storage to train.

For large scale graphs, we store the edges on disk and use a partition buffer for node embedding parameters with asynchronous IO and training to mitigate data movement overheads. The dataset will need to be partitioned at preprocessing time using the ``--num_partitions`` parameter with ``marius_preprocess``.

Large scale graph example:
n = 100 million,  e = 10 billion , r = 10000, and d = 100.

* N = 2 * 1 billion * 100 * 4 bytes = 80GB
* R = 2 * 10000 * 100 * 4 bytes = 8 MB
* E = 10 billion * 3 * 4 bytes = 120 GB

The total overhead is about 200GB.

For this graph we will store and access the edges sequentially from a file on disk, using the ``FlatFile`` backend. For the node embedding parameters we will use the ``PartitionBuffer`` backend which allocates a buffer residing in CPU memory which buffers partitions of node embedding parameters backed by a file on disk.


Recommended configuration options
+++++++++++++++++++++++++++++++++

For the above example graph and deployment we recommend the following configuration.

::

    [storage]
    edges_backend=FlatFile
    embeddings_backend=PartitionBuffer
    relations_backend=DeviceMemory
    num_partitions=16
    buffer_capacity=10
    prefetching=true

    [training]
    synchronous=false

In this case we have the nodes embeddings partitioned into 16 partitions and allow for 8 partitions to reside in buffer at one time. We also enable prefetching which allows for async swapping of partitions from disk to memory at the cost of using 2 * ``partition_size`` extra bytes of memory.

So the breakdown of the overheads is as follows:

partition_size = embedding_table_size / num_partitions = 80 GB / 16 = 5 GB.

- Disk: embedding_table_size + edges_size = 80 GB + 120 GB.
- CPU Memory: (buffer_capacity * partition_size) + (2 * partition_size) (prefetching overhead) = 10 * 5 GB + 2 * 5 GB = 60 GB

In this configuration, the buffer uses 60 GB of CPU memory to train 80GB of embeddings.

Choosing the number of partitions and sizing the buffer
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Choosing the number of partitions and sizing the buffer is determined by the deployment hardware, the input dataset, and training configuration.

Generally, the size of the buffer should be maximized such that as much CPU memory is being used as possible, a larger buffer means less IO and faster training IO bound workloads.

The following formula can be used to determine the best buffer capacity and number of partitions:

Given ``embedding_table_size`` and ``cpu_memory_size`` the ratio ``embedding_table_size`` / ``cpu_memory_size`` should match the ratio ``num_partitions`` / (``buffer_capacity`` + 2)

So for a 500 GB embedding table on a machine with 200 GB memory, about 40% of our embedding table will be able to fit in CPU memory.

So ``embedding_table_size`` / ``cpu_memory_size`` = 2.5 = ``num_partitions`` / (``buffer_capacity`` + 2).

This leaves us with the expression ``num_partitions`` / (``buffer_capacity`` + 2) < 2.5, and we have the freedom to chose any integer value of ``num_partitions`` and ``buffer_capacity`` which satisfies this expression.

Valid options:

- num_partitions = 20, buffer_capacity = 6
- num_partitions = 30, buffer_capacity = 10
- num_partitions = 60, buffer_capacity = 22
- ...

There are other allowable options for these parameters, but to maximize training efficiency it is best to closely match the ``embedding_table_size`` / ``cpu_memory_size`` ratio.


The number of partitions can impact both training time and accuracy. Too few partitions will result in IO bottlenecks, too many will result in poor embedding quality.

Recommendation:

- Size the buffer capacity to the maximum allowable value (determined by CPU memory capacity), buffer capacity should be no smaller than four for best performance.
- Use as few partitions as possible, but no fewer than eight partitions
- Large numbers of partitions 128-256+ can be detrimental to model accuracy, if this many are needed then consider using a smaller embedding dimension.
