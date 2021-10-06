.. _preprocessing:

*************
Preprocessing
*************

Training or evaluation on a dataset (graph) requires conversion from the original format of the dataset into a binary format used internally by Marius.
Marius provides ``marius_preprocess`` command-line as tool for conversion of user datasets, as well as the download and conversion of 21 built-in datasets.

To get information about arguments use the ``--help`` flag and see :ref:`marius_preprocess<user_guide_marius_preprocess>`

Currently, Marius only supports conversion from delimited file formats such as TSVs or CSVs, and for unsupported formats custom converters can be written.

Below we will cover:

* How to preprocess a dataset with a delimited file format.
* How to write a custom dataset converter.
* Preprocessing built-in datasets.

Preprocessing CSVs, TSVs and other delimited formats
----------------------------------------------------------

The dataset must be stored as an edge list in a delimited file format, where each row of the file(s) corresponds to a single edge in the input dataset. Where each row is a tuple of source, edge-type, and destination ids. If the input dataset has only one type of edge, then there are only two columns in the file.

When the delimiter is a comma, the file is a standard CSV as below.

::

    lionel_messi,plays_for,fc_barcelona
    lionel_messi,born_in,argentina
    buenos_aires,capitol_of,argentina
    ...

To perform the conversion on this dataset of ``|E|`` edges we call ``marius_preprocess``

::

    marius_preprocess example_dir/ --files example.csv

Where ``example_dir/`` contains the location of the dataset after preprocessing. If our file was a TSV instead, the preprocessing command will be the same as the preprocessor automatically detects the delimiter of the file.



Looking at the contents of the directory we will see the following files were created.

::

    example_dir/
        train_edges.pt    // Dump of tensor memory: [|E|, 3] sized int32 tensor ->  |E| * 3 * 4 Byte file
        node_mapping.txt  // Mapping of original node ids to unique int32 ids.
        rel_mapping.txt   // Mapping of original edge_type ids to unique int32 ids.

During preprocessing, Marius has randomly assigned integer ids to each node and edge_type, where the mappings to the original ids are stored in ``node_mapping.txt`` and ``rel_mapping.txt``.
The edge list in CSV format is then converted to an [``|E|``, 3] int32 tensor, shuffled and then the contents of the tensor are written to the ``train_edges.pt`` file.

**The path of train_edges.pt will need to be set in the configuration file** when performing training or evaluation.

Splitting datasets for evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The preprocessor supports datasets with predefined train/valid/test splits by passing multiple files to the ``--files`` argument, assuming all files have the same CSV format.

::

    marius_preprocess example_dir/ --files train.csv valid.csv test.csv --delim ","

After this, the files ``valid_edges.pt`` and ``test_edges.pt`` will be created in ``example_dir`` and their paths will need to be set in the configuration file.

If the train/valid/test splits are not predefined, the preprocessor can perform a split on the input dataset with the ``dataset_split`` argument. E.g.

.85/.05/.1 train/valid/test split.

::

    marius_preprocess example_dir/ --files example.csv --delim "," --dataset_split .05 .1


.9/.1 train/test split.

::

    marius_preprocess example_dir/ --files example.csv --delim "," --dataset_split 0 .1


Handling file headers, column order, and offset columns in the input dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Datasets with headers, or different orderings of the columns are supported.

Take the following CSV as an example, which has a three line header, an index column, and a different ordering of the src_id, dst_id, and edge_type columns.

::

    // HEADER
    // EXAMPLE KNOWLEDGE GRAPH
    idx,src_id,dst_id,edge_type
    0,lionel_messi,plays_for,fc_barcelona
    1,lionel_messi,born_in,argentina
    2,buenos_aires,capitol_of,argentina
    ...

We can preprocess this dataset using the following command

::

    marius_preprocess example_dir/ --files example.csv --delim "," --num_line_skip 3 --start_col 1 --format "sdr"

``--num_line_skip 3`` Tells the preprocessor to skip the first three lines of the file.

``--start_col 1`` Indicates the number of columns to ignore, starting from the first column.

``--format "sdr"`` Denotes the ordering of the edge columns, where s is the src_id column, d is the dst_id column, and r is the edge-type or relation column. Any permutation of "srd" and "sd" are supported column orders.

Partitioning the graph
^^^^^^^^^^^^^^^^^^^^^^

Large scale graphs may have an embedding table which exceeds CPU memory capacity. These graphs will need to be partitioned to train. The preprocessor can perform the partitioning using the ``--num_partitions`` option. E.g.

::

    marius_preprocess example_dir/ --files example.csv --delim "," --num_partitions 16

This will partition the nodes of the graph uniformly into 16 partitions and will group the edges into edge buckets, see partition_scheme for more details. The output directory will look like the following:

::

    example_dir/
        train_edges.pt                // ordered according to the edge buckets
        train_edges_partitions.txt    // text file with num_partitions^2 lines, where each line denotes the size of an edge bucket
        node_mapping.txt
        rel_mapping.txt

The edges in ``train_edges.pt`` are ordered according to the edge buckets, where the edges in edge bucket (0, 0) are first in the file, then (0, 1), then (0, 2), .... (15, 15). The sizes of each edge bucket are in ``train_edges_partitions.txt`` and follow the same ordering.

Writing a dataset custom converter
----------------------------------------------------------

If your dataset is not in a supported file format there are a couple options. Convert dataset from original format into a CSV/TSV and use ``marius_preprocess``, or directly convert the dataset into the format used as input to the Marius system.

The first approach can be done by converting the dataset into an edge list stored as three or two column CSV and preprocessed following the instructions given in the previous section.

The second approach can be done in the following steps:

**Without partitioning**

1. Assign each node a unique random int32 id between [0, n), where n is the number of nodes.
2. Assign each edge-type a unique random int32 id between [0, r), where r is the number of edge-types.
3. Create edge list with the new ids.
4. Write the edge list sequentially to a file in binary format, where the first 4 bytes is the source node id for the first edge, the next 4 is the edge-type id, the next the destination node id, and so on.
5. Set path to the edge list in the configuration file.

**With partitioning**

1. Assign each node a unique random int32 id between [0, n), where n is the number of nodes.
2. Assign each edge-type a unique random int32 id between [0, r), where r is the number of edge-types.
3. Partition the nodes of the graph and group edges into edge buckets, as described in partition_scheme.
4. Write edges to a file in binary format, where the edges in edge bucket (0,0) are written first, then (0, 1) ..., (15, 14), (15, 15).
5. Write the number of edges in each edge bucket to another file in a multi-line text format. Where the first line is the size of bucket (0,0), the second line (0, 1), and so on.
6. Set paths to the edge list and edge bucket sizes in the configuration file.

The names of the output files can be anything, as long as the path options are set in the configuration file.

.. _built-in datasets:

Built-in datasets
----------------------------------------------------------

Datasets can be downloaded and preprocessed by using:

::

    marius_preprocess example_dir/ --dataset <dataset_name>

Marius supports the following datasets out-of-the-box:

==================  ================  ======================  ==========
Dataset Name        Entities (nodes)  Relations (edge-types)  Edges
------------------  ----------------  ----------------------  ----------
live_journal        4847571           1                       68993773
fb15k               14951             1345                    592213
fb15k_237           114541            237                     310116
wn18                40943             18                      151442
wn18rr              40943             11                      93003
codex_s             2034              42                      36543
codex_m             17050             51                      206205
codex_l             77951             69                      612437
drkg                97238             107                     5874261
hetionet            45160             25                      2250198
freebase86m         86054151          14824                   338586276
kinships            24                12                      112
ogbl_ppa            576289            1                       30326273
ogbl_ddi            4267              1                       1334889
ogbl_collab         235868            1                       1285465
ogbl_biokg          45085             51                      5088434
ogbn_arxiv          169341            1                       1166243
ogbn_proteins       132534            1                       39561254
ogbn_products       2400608           1                       61859140
openbiolink_hq      184635            28                      4563405
openbiolink_lq      486942            32                      27320889
==================  ================  ======================  ==========

For example, preprocessing the wn18 dataset produces the following output

::

    user@ubuntu: marius_preprocess output_dir/ --dataset wn18
    wn18
    Downloading fetch.phpmedia=en:wordnet-mlj12.tar.gz to download_dir/fetch.phpmedia=en:wordnet-mlj12.tar.gz
    Extracting
    Extraction completed
    Detected delimiter: ~   ~
    Reading in download_dir/wordnet-mlj12-train.txt   1/3
    Reading in download_dir/wordnet-mlj12-valid.txt   2/3
    Reading in download_dir/wordnet-mlj12-test.txt   3/3
    Number of instance per file:[141442, 5000, 5000]
    Number of nodes: 40943
    Number of edges: 151442
    Number of relations: 18
    Delimiter: ~    ~

Generating configuration files
------------------------------

The ``marius_preprocess`` tool can generate a training configuration file for the input dataset using the argument ``--generate_template_config <device>``, where the <device> is CPU for cpu-based processing, and GPU for gpu-based processing.

Specific configuration options can be set by passing ``--<section>.<key>=<value>`` to the command for each option. E.g.

::

    marius_preprocess output_dir/ --dataset wn18 --generate_template_config CPU --model.embedding_size=256 --training.num_epochs=100

This will preprocess the wn18 dataset and will generate a configuration file with following options set:

::

    [general]
    device=CPU
    num_train=141442
    num_nodes=40943
    num_relations=18
    num_valid=5000
    num_test=5000

    [model]
    embedding_size=256

    [path]
    train_edges=output_dir/train_edges.pt
    validation_edges=output_dir/valid_edges.pt
    test_edges=output_dir/test_edges.pt

