.. _User_guide_intro:

****************
User Guide Intro
****************

User Guide
==========

This is the Marius user guide. It helps users figure out how to deploy Marius,
what configuration parameters are available for graph embedding model architectures.
Below are the links to each section of the guide.


**********************
Command Line Interface
**********************

Commands
========

Marius provides several command line interface entry points:

* :ref:`marius_preprocess<user_guide_marius_preprocess>`: preprocess a dataset
* ``marius_config_generator``: generates configuration file
* :ref:`marius_train<user_guide_marius_train>`: trains a graph embedding model
* :ref:`marius_eval<user_guide_marius_eval>`: evaluate the trained graph embeddings and model
* :ref:`marius_postprocess<user_guide_marius_postprocess>`: retrieves trained graph embeddings
* :ref:`marius_predict<user_guide_marius_predict>`: performs link prediction over trained embeddings and model

The details of using these commands are described below.

.. _user_guide_marius_preprocess:

marius_preprocess
^^^^^^^^^^^^^^^^^

This command allows the users to preprocess a dataset to the Marius-trainable format.
This command can be called with:

::

    marius_preprocess <output_directory> [OPTIONS]

The available options:

::

    Preprocess datasets
    
    positional arguments:
        output_directory      Directory to put graph data

    optional arguments:
        -h, --help          show this help message and exit
        --dataset           A Marius supported dataset or custom dataset files
        --num_partitions num_partitions
                            Number of partitions to split the nodes into
        --overwrite           Overwrites the output_directory if this issetOtherwise, files with same the names will be treated as the data for current dataset.
        --generate_config [generate_config], -gc [generate_config]
                            Generates a single-GPU/multi-CPU/multi-GPU training configuration file by default.
                            Valid options (default to GPU): [GPU, CPU, multi-GPU]
        --format            Format of data, eg. srd
        --delim delim, -d delim
                            Specifies the delimiter
        --dtype dtype       Indicates the numpy.dtype
        --not_remap_ids     If set, will not remap ids
        --dataset_split dataset_split dataset_split, -ds dataset_split dataset_split
                            Split dataset into specified fractions
        --start_col start_col, -sc start_col
                            Indicates the column index to start from
        --num_line_skip num_line_skip, -nls num_line_skip
                            Indicates number of lines to skip from the beginning

                            Specify certain config (optional): [--<section>.<key>=<value>]

``marius_preprocess`` is able to prerpocess both the custom dataset provided by users 
and the 21 supported dataset Marius comes included out of the box.

It now supports 
custom datasets in the format of CSV, TSV, TXT. Each edge of the graph must be stored on 
one line with delimiter between the node and relation. The following example presents two 
lines of valid edges using "," as delimiter.

Only ``<output_directory>`` and ``<dataset>`` are required for preprocessing supported datasets.

::

    s1,r1,s2
    s3,r2,s1

output_directory
++++++++++++++++
``<output_directory>`` is a **required** argument for ``marius_preprocess``. 
It is the directory where all the files created by ``marius_preprocess`` wil be stored.
``marius_preprocess`` will create this file if it does not exist.
``maiurs_preprocess`` outputs the following files to ``<output_directory>``.
For the preprocessing of supported datasets, ``<output_directory>`` also includes
the downloaded raw dataset.

==================  ============
File                Description
------------------  ------------
train_edges.pt      Contains edges for training set;

                    Should be set for ``path.train_edges`` in Marius configuration file
valid_edges.pt      Contains edges for validation set; 

                    Should be set for ``path.train_edges`` in Marius configuration file
test_edges.pt       Contains edges for test set; 

                    Should be set for ``path.train_edges`` in Marius configuration file
node_mapping.txt    Contains 2 columns; 
                    The first column is all the original node IDs from raw data, the second column is all the remapped node IDs
rel_mapping.txt     Contains 2 columns; 

                    The first column is all the original relation IDs from raw data, the second column is all the remapped relation IDs
==================  ============

Each edge in ``train_edges.pt``, ``valid_edges.pt``, and ``test_edges.pt`` is stored
in the format of ``source relation destination`` on 1 row.
The 2 Node IDs and relation IDs are stored as 3 4-byte integers (or 8-byte integers
if the storage data type is set to int64). 

The source, relation and destination of edge ``i`` can be retrieved from 
``train_edges.pt``, ``valid_edges.pt``, and ``test_edges.pt``
files by reading 3 4-byte integers (or 8-byte integers if using int64 data type for storage)
at the offset in the file ``i * 3 * 4`` (or ``i * 3 * 8`` when using int64).

\-\-dataset
+++++++++++
``<dataset>`` is a **required** argument for ``marius_preprocess``. 
It can be a list of custom dataset files separated by space. provided by users. It can also be the name
of a Marius supported dataset. To see which datasets are supported by Marius, check out
:ref:`dataset` table.

For example, the following command preprocesses the custom dataset composed of ``custom_train.csv``,
``custom_valid.csv`` and ``custom_test.csv`` and stores them into directory ``output_dir``.

::

    marius_preprocess output_dir --dataset custom_train.csv custom_valid.csv custom_test.csv

\-\-num_partitions
++++++++++++++++++
``--num_partitions <num_partitions>`` is an optional argument for ``marius_preprocess``.
If this option is specified, the nodes of the input graph will be partitioned into ``<num_partitions>``.
The default value for ``<num_partitions>`` is one.

\-\-overwrite
+++++++++++++
``--overwrite`` is an **optional** argument for ``marius_preprocess``. If this option is set, then
the ``<output_directory>`` will be overwritten. Otherwise, ``marius_preprocess`` will treat the files
in ``<output_directory>`` with the same file names as the latest files for current run. When switching
from one dataset to another one, the converted data files of the previous dataset in same ``<output_directory>``
may be treated as the already-preprocessed data files for the current dataset if this option is not set.

\-\-generate_config <device>, \-gc <device>
+++++++++++++++++++++++++++++++++++++++++++
``--generate_config <device>, -gc <device>`` is an **optional** argument for ``marius_preprocess``.
If this option is set, ``marius_preprocess`` will generate a Marius configuration
file in the ``<output_directory>`` with all configuration parameters set to the recommended defaults if not 
explicitly defined.

The generated Marius configuration is for single-GPU setting by default if ``<device>`` is not set.
If other device, such as ``CPU`` or ``multi-GPU``, is required, users can just append the option after
``--generate_config``, e.g. ``--generate_config CPU``.

For example, the following example will set ``general.device=CPU`` in the Marius 
configuration file generated for dataset WordNet18 (``wn18_cpu.ini``).

::

    marius_preprocess wn18 ./output_dir --generate_config CPU

\-\-<section>.<key>=<value>
+++++++++++++++++++++++++++
``--<section>.<key>=<value>`` is an **optional** argument for ``marius_preprocess``.
When ``--generate_config <device>`` is set, ``--<section>.<key>=<value>`` can be used
to change the value of certain option in the Marius configuration file generated.
For example, the following example will set ``model.embedding_sze=256`` and ``training.num_epochs=100``
in the Marius configuration file generated for custom dataset composed of ``custom_dataset.csv`` (``custom_gpu.ini``).

::

    marius_preprocess custom_dataset.csv ./output_dir --generate_config --model.embedding_sze=256 --training.num_epochs=100

\-\-format <format>
+++++++++++++++++++
``--format <format>`` is an **optional** argument for ``marius_preprocess``.
This is the sequence of the source node, relation, and destination node appears on one row of the 
dataset file. ``<format>`` should be specified using a string composed of only ``s`` for source, 
``r`` for relation and ``d`` for destination.

For example, the following command shows the how to preprocess a dataset file 
storing edges in the sequence of source node, relation and destination node.

::

    marius_preprocess custome_dataset.csv ./output_dir --format src

\-\-delim delim, -d delim
+++++++++++++++++++++++++
``--delim=<delim>`` is an **optional** argument for ``marius_preprocess``.
``<delim>`` defines the delimiter between nodes and relations in the dataset files.
If ``<delim>`` is not set, ``marius_preprocess`` will use Python Sniffer to detect a delimiter.
The delimiter is printed to the terminal so users can verify it.

\-\-dtype <dtype>
+++++++++++++++++
``--dtype <dtype>`` is an **optional** argument for ``marius_preprocess``.
It defines the format for storing each node remapped ID and relation remapped ID. The current supported
format is ``int32`` and ``int64``. 
When storing in ``int32``, each remapped ID will be a 4-byte integer.
When storing in ``int64``, each remapped ID will be a 8-byte integer.
If the total number of nodes or relations is smaller than 2.1 billion,
it is recommended to choose ``int32`` to avoid unnecessary waste of space.
On the other hand, if the total number of nodes or relations is bigger than 2.1 billion,
it is recommended to choose``int64`` to ensure all IDs to be remapped.
The default ``<dtype>`` is set to ``int32``.

\-\-not_remap_ids
+++++++++++++++++
``--not_remap_ids`` is an **optional** argument for ``marius_preprocess``.
If this option is set, the remapped IDs of nodes and relations will be the same 
as the read-in order of the nodes and relations from original dataset.

\-\-dataset_split <validation proportion> <test proportion>, \-ds <validation proportion> <test proportion>
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
``--dataset_split <validation proportion> <test proportion>, -ds <validation proportion> <test proportion>``
is an **optional** argument for ``marius_preprocess``.
It offers the user the option to split the dataset. By appending the proportion of validation
and test set out of total dataset, users can split the original dataset into training, validation,
and test sets. An exception would be raised if the sum of these two proportions exceeds or equal to one.
By default, ``marius_preprocess`` merges all dataset files and produce one training set containing all edges.

For example, the following command splits the ``custome_dataset.csv`` into training,
validation, and test sets with a corresponding proportion of 0.99, 0.05, and 0.05.

::

    marius_preprocess custom_dataset.csv ./output_dir --dataset_split 0.05 0.05

\-\-start_col <start_col>
+++++++++++++++++++++++++
``--start_col <start_col>`` is an **optional** argument for ``marius_preprocess``.
This is the column to treat as the nodes/relations column in custom dataset files.
The next two columns will be treated as relations/nodes. Whether a column is treated
as a node column or relation column is defined by the ``<format>`` argument.
The default value for ``<start_col>`` is zero.

\-\-num_line_skip <num_line_skip>, \-nls <num_line_skip>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++
``--num_line_skip <num_line_skip>, \-nls <num_line_skip>`` is an **optional** argument for ``marius_preprocess``.
It is the number of lines of headers to skip when reading the custom dataset files.
If this value is not set. ``marius_preprocess`` uses Python Sniffer to determine the number of header row.

.. _user_guide_marius_train:

marius_train
^^^^^^^^^^^^

This command allows users to train a graph embedding model over the preprocessed data.
A Marius configuration file is required for this command. See :ref:`Configuration<user_guide_configuration>`
for full details of Marius configuration file.

This command can be called with:

::

    marius_train <config_file> [OPTIONS]

The available options:

::

    Train and evaluate graph embeddings
    Usage:
    marius_train config_file [OPTIONS...] [<section>.<option>=<value>...]

    -h, --help  Print help and exit.

The ``config_file`` is the Marius configuration file that includes all configuration
options for model architectures and training pipeline.

The configuration options can also be modified by passing ``--<section>.<key>=<value>``
to the end of the ``marius_train`` command.
Any parameter passed in the command line will override the value specified 
in the configuration file. The details about ``<section>.<key>`` can be 
found in :ref:`Configuration<user_guide_configuration>`. The following is an example
of overriding the decoder model to ``TransE`` and using ``config.ini`` as the configuration
file:

::

    marius_train config.ini --model.decoder=TransE

During the execution of this ``marius_train``, information about training progress
and model performance is printed to terminal. The output of the first epoch would be 
similar to the following. 

::

    [info] [03/18/21 01:33:18.778] Metadata initialized
    [info] [03/18/21 01:33:18.778] Training set initialized
    [info] [03/18/21 01:33:18.779] Evaluation set initialized
    [info] [03/18/21 01:33:18.779] Preprocessing Complete: 2.605s
    [info] [03/18/21 01:33:18.791] ################ Starting training epoch 1 ################
    [info] [03/18/21 01:33:18.836] Total Edges Processed: 40000, Percent Complete: 0.082
    [info] [03/18/21 01:33:18.862] Total Edges Processed: 80000, Percent Complete: 0.163
    [info] [03/18/21 01:33:18.892] Total Edges Processed: 120000, Percent Complete: 0.245
    [info] [03/18/21 01:33:18.918] Total Edges Processed: 160000, Percent Complete: 0.327
    [info] [03/18/21 01:33:18.944] Total Edges Processed: 200000, Percent Complete: 0.408
    [info] [03/18/21 01:33:18.970] Total Edges Processed: 240000, Percent Complete: 0.490
    [info] [03/18/21 01:33:18.996] Total Edges Processed: 280000, Percent Complete: 0.571
    [info] [03/18/21 01:33:19.021] Total Edges Processed: 320000, Percent Complete: 0.653
    [info] [03/18/21 01:33:19.046] Total Edges Processed: 360000, Percent Complete: 0.735
    [info] [03/18/21 01:33:19.071] Total Edges Processed: 400000, Percent Complete: 0.816
    [info] [03/18/21 01:33:19.096] Total Edges Processed: 440000, Percent Complete: 0.898
    [info] [03/18/21 01:33:19.122] Total Edges Processed: 480000, Percent Complete: 0.980
    [info] [03/18/21 01:33:19.130] ################ Finished training epoch 1 ################
    [info] [03/18/21 01:33:19.130] Epoch Runtime (Before shuffle/sync): 339ms
    [info] [03/18/21 01:33:19.130] Edges per Second (Before shuffle/sync): 1425197.8
    [info] [03/18/21 01:33:19.130] Edges Shuffled
    [info] [03/18/21 01:33:19.130] Epoch Runtime (Including shuffle/sync): 339ms
    [info] [03/18/21 01:33:19.130] Edges per Second (Including shuffle/sync): 1425197.8
    [info] [03/18/21 01:33:19.148] Starting evaluating
    [info] [03/18/21 01:33:19.254] Pipeline flush complete
    [info] [03/18/21 01:33:19.271] Num Eval Edges: 50000
    [info] [03/18/21 01:33:19.271] Num Eval Batches: 50
    [info] [03/18/21 01:33:19.271] Auc: 0.973, Avg Ranks: 24.477, MRR: 0.491, Hits@1: 0.357, Hits@5: 0.651, Hits@10: 0.733, Hits@20: 0.806, Hits@50: 0.895, Hits@100: 0.943

After the training is finished, Marius will generate a directory ``data/`` for storing all the trained model
and a directory ``logs/`` for all the log information during the training.

The following is the description of each file in ``data/``.

=========================================  ================
File                                       Description
-----------------------------------------  ----------------
data/marius/edges/train/edges.bin          contains the edges in training set
data/marius/edges/evaluation/edges.bin     contains the edges in validation set
data/marius/edges/test/edges.bin           contains the edges in test set
data/marius/embeddings/embeddings.bin      contains the embedding vectors for each node
data/marius/embeddings/state.bin           contains the embedding optimizer state for each node
data/marius/relations/src_relations.bin    contains the embedding vectors for relations with source nodes
data/marius/relations/src_state.bin        contains the embedding optimizer state for relations with source nodes
data/marius/relations/dst_relations.bin    contains the embedding vectors for relations with destination nodes
data/marius/relations/dst_state.bin        contains the embedding optimizer state for relations with destination nodes
=========================================  ================

The following is the description of each file in ``logs/``.

==============================  ================
File                            Description
------------------------------  ----------------
logs/marius_debug.log           contains detailed logs for debugging purposes
logs/marius_error.log           contains the error messages produced by the system
logs/marius_evaluation.trace    contains the status of the pipeline during evaluation
logs/marius_info.log            contains the information about training progress and model performance; the same information printed to terminal
logs/marius_trace.log           contains system trace logs for tracing program execution
logs/marius_train.trace         contains the status of the pipeline during training
logs/marius_warn.log            contains the warning messages produced by the system
==============================  ================


.. _user_guide_marius_eval:

marius_eval
^^^^^^^^^^^

This command lets users perform evaluation on the trained embeddings and model.
It can be called with:

::

    marius_eval <config_file>

The available arguments:

::

    Train and evaluate graph embeddings
    Usage:
    marius_eval config_file [OPTIONS...] [<section>.<option>=<value>...]

    -h, --help  Print help and exit.

``marius_eval`` performs evaluations to the trained embeddings and model without training them again.
The ``<config_file>`` is the same config_file used for ``marius_train``. The output of ``marius_eval``
will be similar to the following.

::

    [info] [07/28/21 01:58:10.368] Start preprocessing
    [info] [07/28/21 01:58:10.407] Preprocessing Complete: 0.039s
    [info] [07/28/21 01:58:10.473] Starting evaluating
    [info] [07/28/21 01:58:10.546] Pipeline flush complete
    [info] [07/28/21 01:58:10.547] Num Eval Edges: 5000
    [info] [07/28/21 01:58:10.547] Num Eval Batches: 5
    [info] [07/28/21 01:58:10.547] Auc: 0.605, Avg Ranks: 394.716, MRR: 0.052, Hits@1: 0.029, Hits@5: 0.065, Hits@10: 0.086, Hits@20: 0.117, Hits@50: 0.179, Hits@100: 0.250
    [info] [07/28/21 01:58:10.549] Evaluation complete: 76ms


..  _user_guide_marius_postprocess:

marius_postprocess
^^^^^^^^^^^^^^^^^^

This command lets users to retrieve the trained graph embeddings and store in the desired format.
``marius_postprocess`` creates a file containing all the trained embeddings.

This command can be called with:

::

    marius_postprocess <output directory> <format>

The available options:

::

    Will be updated once functionalities of marius_postprocess is added.

The ``<output directory>`` is the directory where the retrieved graph embeddings 
will be stored.The ``<format>`` is the format of the retrieved graph embeddings.
Currently, the supported formats include CSV, TSV, PyTroch Tensor, parquet. 

The index of the embeddings in the output file follows the remmaped IDs of the node or entity.
The mapping information between the original IDs and remapped IDs is in ``node_mapping.txt`` and 
``rel_mapping.txt`` created by ``marius_preprocess``. See :ref:`marius_preprocess<user_guide_marius_preprocess>`
for detailed description.

The following command shows how to use marius_postprocess for retrieving trained graph embeddings.

::

    marius_postprocess output_dir CSV

In this case, the ``output_dir`` is the directory containing the files with embeddings.
These embeddings will be stored in the CSV format.


.. _user_guide_marius_predict:

marius_predict
^^^^^^^^^^^^^^

This command lets users to perform link predictions using trained graph embeddings.
It can be called with:

::

    marius_predict <embedding_directory> <dataset_directory> <source node> <relation> <k>

The ``<embedding_directory>`` is the directory ``data/`` created by ``marius_train``.
The ``<dataset_directory>`` is the directory containing the ``node_mapping.txt`` and ``rel_mapping.txt`` files.
Given the source node and relation, ``marius_predict`` outputs the top-``k`` destination nodes.

The following example shows how ``marius_predict`` is used for link prediction.

::

    example coming soon after maris_predict is created






Data Preprocessing
==================



.. _user_guide_configuration:


Configuration
=============

**Can use the configuration doc we have for now: https://github.com/marius-team/marius/blob/main/docs/configuration.rst**

Hyper-parameter optimization
============================

Marius provides 90 configurable parameters divided into 
nine main section, including model, storage, training, pipelining and evaluation.
These parameters can be tuned to achieve the best system efficiency and training performance
for dataset with certain properties.

Memory Hierarchy Usage
^^^^^^^^^^^^^^^^^^^^^^
Marius achieves great training efficiency via its novel in-memory replacement method.
Marius configuration parameters allows users to use the entire memory hierarchy efficiently by 
tuning the hardware used by Marius, the batch size used during training, the number of 
partitions of the input graph, as well as the asynchrony of Marius training pipeline.

Marius denotes the memory hierarchy according to the follow table.

===============  ===========
Parameter value  Description
---------------  -----------
DeviceMemory     GPU memory
HostMemory       CPU memory
PartitionBuffer  Disk???
FlatFile         Disk
===============  ===========

The overheads of storing and training each dataset is calculated as follows:
For training d-dimensional graph embeddings with n nodes, r edge-types and e edges:

* Overhead of storing node embedding parameters + optimizer state: N = 2 * n * d * 4 bytes
* Overhead of edge-type embedding parameters + optimizer state: R = 2 * r * d * 4 bytes
* Overhead of storing edges: E = e * 3 * 4 bytes (with int32 node ids)

The sum of these overheads is the total overhead of training: N + R + E

Large graph example
+++++++++++++++++++

For a graph with n = 100 million,  e = 1 billion, r = 10000, and d = 100 we have:

* N = 2 * 100 million * 100 * 4 bytes = 80GB
* R = 2 * 10000 * 100 * 4 bytes = 8 MB
* E = 1 billion * 3 * 4 bytes = 12 GB

The AWS p3.2xlarge instance instance has 64 GB of CPU memory and 16 GB of GPU memory.
Assume we have our hardware setting as mentioned above.

Edge-type embeddings
""""""""""""""""""""

As we can see the overhead of the edge type embeddings is only eight MB, quite small.
Most publicly available datasets have few edge-types and we have not observed a dataset 
with more than fifteen thousand edge-types. Therefore, the relations_backend should almost always 
be set to ``DeviceMemory`` (gpu memory), unless the graph has millions of edge types.

Node embeddings
"""""""""""""""

The overhead of the node embedding parameters in our example exceeds the CPU memory 
capacity. Therefore we will have to partition the embedding parameters and use the 
partition buffer (using ``PartitionBuffer`` backend) swap partitions in and out of CPU memory. 
We can define the capacity of the buffer and the number of partitions in the following manner:

* The capacity of the buffer should be set to the maximum value such that it does not 
  exceed CPU memory capacity. 
  So in our example, if we partition the node embedding parameters into 10 partitions, 
  each partition has the size 80 GB / 10 = 10 GB. 
  Since our CPU memory capacity is 64 GB, we can fit 6 total partitions in CPU memory. 
  Therefore we should set the buffer capacity to 6, and the buffer uses 60GB of CPU memory.
* For best accuracy, the fewer partitions used generally results in better quality embeddings.
  Using a few partitions (8-16) is recommend for small datasets. 
  For large datasets, 16+ partitions may be needed.

Note that if we used a smaller embedding size such as d = 50. 
N would become 40GB. It would fit in CPU memory. 
In that case we would use the ``HostMemory`` storage backend for node embeddings. 
If we used an even smaller embedding size such that d = 10, 
then the overhead is only 10 GB. 
Therefore we could store them using the ``DeviceMemory`` backend, since GPU memory capacity is generally larger than 16GB.


Edges
"""""

In this example, storing our edges requires 12 GB. 
While in isolation, this will fit just fine in CPU memory. 
But when combined with the overhead of storing the partition buffer (60 GB), 
it will exceed CPU memory capacity. 
As edges are accessed by the system using sequential IO, 
they can safely be stored on disk assuming at least 100 MBps of disk throughput. 
This can be done by setting the edges_backend to ``FlatFile``.

Small graph example
+++++++++++++++++++

For smaller graphs the overheads much smaller, take one with
n = 1 million,  e = 100 million , r = 1000, and d = 100 we have:

* N = 2 * 1 million * 100 * 4 bytes = 800MB
* R = 2 * 1000 * 100 * 4 bytes = .8 MB
* E = 100 million * 3 * 4 bytes = 1.2 GB

We can see that the total overhead is only about 2GB.
This will fit just fine in GPU memory. 
Therefore all can be stored with the ``DeviceMemory`` backend.

Notes for storing and training of large graphs:

* If any of the edges, node embeddings or edge-types embeddings are stored off GPU, 
  then asynchronous training should be used for fastest training times. 
  The default training pipeline configuration parameters should be sufficient for most graphs and deployments.
* To best hide IO wait times and improve training times when using the partition buffer, 
  prefetching should be enabled. 
  When prefetching is enabled the overhead of the buffer will increase to 
  partition_size * (buffer_capacity + 2), 
  as we use two partition sized regions of memory for async writes and prefetching.



Datasets
========

The following table contains the information of the 21 datasets Marius comes included out of the box.

==================  ==========  ======================  ==========
Dataset Name        Entities    Relations (edge-types)  Edges  
------------------  ----------  ----------------------  ----------
live_journal        4847571     1                       68993773
fb15k               14951       1345                    592213
fb15k_237           114541      237                     310116
wn18                40943       18                      151442
wn18rr              40943       11                      93003
codex_s             2034        42                      36543
codex_m             17050       51                      206205
codex_l             77951       69                      612437
drkg                97238       107                     5874261
hetionet            45160       25                      2250198
freebase86m         86054151    14824                   338586276
kinships            24          12                      112
ogbl_ppa            576289      1                       30326273
ogbl_ddi            4267        1                       1334889
ogbl_collab         235868      1                       1285465
ogbl_biokg          45085       51                      5088434
ogbn_arxiv          169341      1                       1166243
ogbn_proteins       132534      1                       39561254
ogbn_products       2400608     1                       61859140
openbiolink_hq      184635      28                      4563405
openbiolink_lq      486942      32                      27320889
==================  ==========  ======================  ==========