.. _command_line_interface:

**********************
Command Line Interface
**********************

Here we go into detail on the arguments and usage of the command line interface to Marius.

- marius_preprocess: convert datasets into the Marius input format
- marius_train: train embeddings for input dataset
- marius_eval: perform evaluation on the link prediction task
- marius_postprocess: convert trained embeddings to desired output format
- marius_predict: perform interactive link prediction over trained embeddings


.. _user_guide_marius_preprocess:

marius_preprocess
^^^^^^^^^^^^^^^^^

See :ref:`preprocess` for additional context and usage.

This command allows the users to preprocess a dataset to the Marius-trainable format.
This command can be called with:

::

 marius_preprocess [OPTIONS]

The available options:

::

    usage: preprocess [-h] [--output_directory output_directory] [--files files [files ...] | --dataset dataset] [--num_partitions num_partitions] [--overwrite]
                    [--generate_config [generate_config]] [--format format] [--delim delim] [--dtype dtype] [--not_remap_ids] [--dataset_split dataset_split dataset_split]
                    [--start_col start_col] [--num_line_skip num_line_skip]

    Preprocess Datasets

    optional arguments:
    -h, --help            show this help message and exit
    --output_directory output_directory
                            Directory to put graph data
    --files files [files ...]
                            Files containing custom dataset
    --dataset dataset     Supported dataset to preprocess
    --num_partitions num_partitions
                            Number of partitions to split the edges into
    --overwrite           Overwrites the output_directory if this is set. Otherwise, files with same the names will be treated as the data for current dataset.
    --generate_config [generate_config], -gc [generate_config]
                            Generates a single-GPU training configuration file by default.
                            Valid options (default to GPU): [GPU, CPU, multi-GPU]
    --format format       Format of data, eg. srd
    --delim delim, -d delim
                            Specifies the delimiter
    --dtype dtype         Indicates the numpy.dtype
    --not_remap_ids       If set, will not remap ids
    --dataset_split dataset_split dataset_split, -ds dataset_split dataset_split
                            Split dataset into specified fractions
    --start_col start_col, -sc start_col
                            Indicates the column index to start from
    --num_line_skip num_line_skip, -nls num_line_skip
                            Indicates number of lines to skip from the beginning

    Specify certain config (optional): [--<section>.<key>=<value>]

output_directory
++++++++++++++++
``<output_directory>`` is a **optional** argument for ``marius_preprocess``. 
It is the base directory where all the files created by ``marius_preprocess`` wil be stored.
``marius_preprocess`` will create this directory if it does not exist. If users specifies a 
built-in dataset, the default base directory name would be ``<built-in dataset>_dataset``.
If users want to preprocess a custom dataset, the default base directory name would 
be ``custom_dataset``. If there is already a file with the same name, ``marius_preprocess`` 
would incrementally append a number after the original file name.
``marius_preprocess`` outputs the following files to ``<output_directory>``.
For the preprocessing of supported datasets, ``<output_directory>`` also includes
the downloaded raw dataset.

==================  ============
File                Description
------------------  ------------
train_edges.pt      Contains edges for training set;

                    Should be set for ``path.train_edges`` in Marius configuration file
valid_edges.pt      Contains edges for validation set; 

                    Should be set for ``path.valid_edges`` in Marius configuration file
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

\-\-files <files ...>
+++++++++++++++++++++
``--files`` is an **optional** argument for ``marius_preprocess``.
It should be a list of files containing custom dataset. It should not be used
at the same time when ``--dataset`` is used.

For example, the following command preprocesses the custom dataset composed of ``custom_train.csv``,
``custom_valid.csv`` and ``custom_test.csv`` and stores them into directory ``output_dir``.

::

    marius_preprocess --output_directory output_dir --files custom_train.csv custom_valid.csv custom_test.csv

\-\-dataset <dataset>
+++++++++++++++++++++
``--dataset`` is an **optional** argument for ``marius_preprocess``.
It can be one of the names of a Marius supported dataset. 
It should not be used at the same time when ``--files`` is used.
To see which datasets are supported by Marius, check out
:ref:`dataset` table.

\-\-num_partitions <num_partitions>
+++++++++++++++++++++++++++++++++++
``--num_partitions`` is an optional argument for ``marius_preprocess``.
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

    marius_preprocess --output_directory ./output_dir --dataset wn18 --generate_config CPU

\-\-<section>.<key>=<value>
+++++++++++++++++++++++++++
``--<section>.<key>=<value>`` is an **optional** argument for ``marius_preprocess``.
When ``--generate_config <device>`` is set, ``--<section>.<key>=<value>`` can be used
to change the value of certain option in the Marius configuration file generated.
For example, the following example will set ``model.embedding_sze=256`` and ``training.num_epochs=100``
in the Marius configuration file generated for custom dataset composed of ``custom_dataset.csv`` (``custom_gpu.ini``).

::

    marius_preprocess --output_directory ./output_dir --files custom_dataset.csv --generate_config --model.embedding_sze=256 --training.num_epochs=100

\-\-format <format>
+++++++++++++++++++
``--format <format>`` is an **optional** argument for ``marius_preprocess``.
This is the sequence of the source node, relation, and destination node appears on one row of the 
dataset file. ``<format>`` should be specified using a string composed of only ``s`` for source, 
``r`` for relation and ``d`` for destination.

For example, the following command shows the how to preprocess a dataset file 
storing edges in the sequence of source node, relation and destination node.

::

    marius_preprocess --output_directory ./output_dir --files custom_dataset.csv --format src

\-\-delim <delim>, \-d <delim>
+++++++++++++++++++++++++++++
``--delim`` is an **optional** argument for ``marius_preprocess``.
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

For example, the following command splits the ``custom_dataset.csv`` into training,
validation, and test sets with a corresponding proportion of 0.99, 0.05, and 0.05.

::

    marius_preprocess --output_directory ./output_dir --files custom_dataset.csv --dataset_split 0.05 0.05

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

.. _user_guide_marius_config_generator:

marius_config_generator
^^^^^^^^^^^^^^^^^^^^^^^

This command lets users to create a Marius configuration file from the command line with
some parameters specified according to their needs.
This command can be called with:

::

    marius_config_generator <output_directory> [OPTIONS]

The available options:

::

    usage: config_generator [-h] [--data_directory data_directory] [--dataset dataset | --stats num_nodes num_edge_types num_train num_valid num_test]
    [--device [generate_config]]
    output_directory

    Generate configs

    positional arguments:
    output_directory      Directory to put configs
    Also assumed to be the default directory of preprocessed data if --data_directory is not specified

    optional arguments:
    -h, --help            show this help message and exit
    --data_directory data_directory
    Directory of the preprocessed data
    --dataset dataset, -d dataset
    Dataset to preprocess
    --stats num_nodes num_edge_types num_train num_valid num_test, -s num_nodes num_edge_types num_train num_valid num_test
    Dataset statistics
    Enter in order of num_nodes, num_edge_types, num_train num_valid, num_test
    --device [generate_config], -dev [generate_config]
    Generates configs for a single-GPU/multi-CPU/multi-GPU training configuration file by default.
    Valid options (default to GPU): [GPU, CPU, multi-GPU]

    Specify certain config (optional): [--<section>.<key>=<value>]

<output_directory>
++++++++++++++++++
``<output_directory>`` is a **required** argument. It specifies the output directory of the created configuration file.

\-\-data_directory <data_directory>
+++++++++++++++++++++++++++++++++++
``--data_directory`` is an **optional** argument. It specifies the directory where ``marius_preprocess`` stores
preprocessed data.

\-\-dataset <dataset>, \-d <dataset>
++++++++++++++++++++++++++++++++++++
``--dataset`` is an **optional** argument. It specifies the name of the supported dataset. It should not be
used when ``--stats`` is in use.

\-\-stats <num_nodes> <num_relations> <num_train> <num_valid> <num_test>, \-s <num_nodes> <num_relations> <num_train> <num_valid> <num_test>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
``--stats <num_nodes> <num_relations> <num_train> <num_valid> <num_test>, -s <num_nodes> <num_relations> <num_train> <num_valid> <num_test>``
is an **optional** argument. It specifies the stats of the dataset to be trained over. It should not be used at the same
time with option ``--dataset``.

\-\-device <device>, \-dev <device>
+++++++++++++++++
``--device`` is an **optional** argument. The default value of it is GPU. It takes only three values: GPU, CPU, multi-GPU.
It specifies the device option.


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

After the training is finished, Marius will generate a directory according to the ``path.data_directory`` option for storing all the trained model
and a directory ``logs/`` for all the log information during the training.

The following is the description of each file in ``path.data_directory``.

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

This command lets users perform link-prediction evaluation on the trained embeddings and model.
It can be called with:

::

    marius_eval <config_file>

The available arguments:

::

    Train and evaluate graph embeddings
    Usage:
    marius_eval config_file [OPTIONS...] [<section>.<option>=<value>...]

    -h, --help  Print help and exit.

The ``<config_file>`` can be the same configuration file used for ``marius_train``, or a separate configuration file defined for different evaluation scenarios. The output of ``marius_eval``
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

    marius_postprocess <trained_embedding_directory> <dataset_directory> [OPTIONS]

The available options:

::

    usage: postprocess [-h] [--output_directory output_directory] [--format format] trained_embeddings_directory dataset_directory

    Retrieve trained embeddings

    positional arguments:
    trained_embeddings_directory
                            Directory containing trained embeddings
    dataset_directory     Directory containing the dataset for training

    optional arguments:
    -h, --help            show this help message and exit
    --output_directory output_directory, -o output_directory
                            Directory to put retrieved embeddings. If is not set, will output retrieved embeddings to dataset directory.
    --format format, -f format
                            Data format to store retrieved embeddings

The ``<trained_embedding_directory>`` is the directory created 
by ``marius_train`` containing all trained embeddings.
The ``<dataset_directory>`` is the directory created by ``marius_preprocess`` to store preprocessed data.

\-\-output_directory <output_directory>, \-o <output_directory>
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``--output directory`` is an **optional** argument. It is
the directory where the retrieved graph embeddings will be stored.

\-\-format <format>, \-f <format>
+++++++++++++++++++++++++++++++++

``--format`` is an **optional** argument. It specifies the storing format of the retrieved graph embeddings.
Currently, the supported formats include CSV, TSV and PyTorch Tensor.

The index of the embeddings in the output file follows the remmaped IDs of the node or entity.
The mapping information between the original IDs and remapped IDs is in ``node_mapping.txt`` and
``rel_mapping.txt`` created by ``marius_preprocess``. See :ref:`marius_preprocess<user_guide_marius_preprocess>`
for detailed description.

The following command shows how to use ``marius_postprocess`` for retrieving trained graph embeddings.

::

    marius_postprocess ./data/ ./dataset_directory --output_directory output_dir -f CSV

In this case, ``./data/`` is the directory created by ``marius_train`` containing all the
trained embeddings. ``./dataset_directory`` is the directory created by ``marius_preprocess``
containing all preprocessed data files.
These embeddings will be stored in the CSV format.


.. _user_guide_marius_predict:

marius_predict
^^^^^^^^^^^^^^

This command lets users to perform link predictions using trained graph embeddings.
Users can either perform link prediction for a single node and edge-type, or pass in many nodes and edge-types from a file and perform batched link prediction.

It can be called with:

::

    marius_predict <trained_embeddings_directory> <dataset_directory> <k> [OPTIONS]


The available options are:

::

    usage: predict [-h] [--src src] [--dst dst] [--rel rel] [--decoder decoder] [--file_input file_input] trained_embeddings_directory dataset_directory k

    Perform link prediction

    positional arguments:
    trained_embeddings_directory
                            Directory containing trained embeddings
    dataset_directory     Directory containing the dataset for training
    k                     Number of predicted nodes to output

    optional arguments:
    -h, --help            show this help message and exit
    --src src, -s src     Source node, the original ID of a certain node
    --dst dst, -d dst     Destination node, the original ID of a certain node
    --rel rel, -r rel     Relation (edge-type), the original ID of a certain relation
    --decoder decoder, -dc decoder
                            Specifies the decoder used for training
    --file_input file_input, -f file_input
                            File containing all required information for batch inference

The ``<trained_embeddings_directory>`` is the directory ``data/`` created by ``marius_train``.
The ``<dataset_directory>`` is the directory containing the ``node_mapping.txt`` and ``rel_mapping.txt`` files.
The ``<k>`` controls the number of predicted node to output.

\-\-src <src>, \-s <src>
++++++++++++++++++++++++
``--src <src>, -s <src>`` is an **optional** argument. It is the original node ID of source node.

\-\-rel <rel>, \-r <rel>
++++++++++++++++++++++++
``--rel <rel>, -r <rel>`` is an **optional** argument. It is the original relation ID of the relation.

\-\-dst <dst>, \-d <dst>
++++++++++++++++++++++++
``--dst <rel>, -d <rel>`` is an **optional** argument. It is the original node ID of destination node.

\-\-decoder <decoder>, \-dc <decoder>
+++++++++++++++++++++++++++++++++++++
``--decoder <decoder>, -dc <decoder>`` is an **optional** argument. It specifies the decoder used
for training. Input values must be chosen from ``DisMult``, ``TransE``, ``ComplEx``.
The default value is ``DisMult``.

\-\-file_input <file_input>, \-f <file_input>
+++++++++++++++++++++++++++++++++++++++++++++
``--file_input <file_input>, -f <file_input>`` is an **optional** argument. User can put all
inferences they want to perform in this file and make all inferences in one run.

Each inference in the file should take one row. On each row, there should be two commas as
the delimiters between nodes and relation. Node IDs and relation IDs in the original
dataset file should be used. Replace the target of the inference use an empty string.
If the dataset has multiple relation types,
each inference needs to contain a node id and a relation type. If the dataset only has one
relation type, each inference only needs a node id.

The following example is valid as contents of the inference file:

::

    00789448,_verb_group,
    ,_hyponym,10682169
    ,_member_of_domain_region,05688486
    02233096,_member_meronym,
    01459242,_part_of,


Given the source node, relation and other necessary arguments,
 ``marius_predict`` outputs the top-five destination nodes
in the following example.

::

    marius_predict ./data/ ./dataset_directory 5 -s source_node_id -r relation_id
