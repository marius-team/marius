
Command Line Preprocessing
================================

The preprocessing procedure takes datasets in their raw format and converts them to the input format required by Marius.

Built-in datasets
-----------------------

Preprocessing the FB15K-237 knowledge graph

.. code-block:: bash

   $ marius_preprocess --dataset fb15k_237 --output_directory datasets/fb15k_237_example/
   Downloading FB15K-237.2.zip to datasets/fb15k_237_example/FB15K-237.2.zip
   Reading edges
   Remapping Edges
   Node mapping written to: datasets/fb15k_237_example/nodes/node_mapping.txt
   Relation mapping written to: datasets/fb15k_237_example/edges/relation_mapping.txt
   Dataset statistics written to: datasets/fb15k_237_example/dataset.yaml

The  ``--dataset`` flag specifies which of the built-in datasets ``marius_preprocess`` will preprocess and download.

The  ``--output_directory`` flag specifies where the preprocessed graph will be output and is set by the user. In this example, assume we have not created the datasets/fb15k_237_example repository. ``marius_preprocess`` will create it for us.

See `Usage`_ for detailed options.

Here are the contents of the output directory after preprocessing

.. code-block:: bash

   $ ls -l datasets/fb15k_237_example/
   dataset.yaml                       # input dataset statistics
   nodes/
     node_mapping.txt                 # mapping of raw node ids to integer uuids
   edges/
     relation_mapping.txt             # mapping of raw edge(relation) ids to integer uuids
     test_edges.bin                   # preprocessed testing edge list
     train_edges.bin                  # preprocessed training edge list
     validation_edges.bin             # preprocessed validation edge list
   train.txt                          # raw training edge list
   test.txt                           # raw testing edge list
   valid.txt                          # raw validation edge list
   text_cvsc.txt                      # relation triples as used in Toutanova and Chen CVSM-2015
   text_emnlp.txt                     # relation triples as used inToutanova et al. EMNLP-2015
   README.txt                         # README of the downloaded FB15K-237 dataset


List of built-in datasets

.. code-block:: text

    # node classification
    ogbn_arxiv
    ogbn_products
    ogbn_papers100m
    ogb_mag240m

    # link prediction
    fb15k
    fb15k_237
    livejournal
    twitter
    freebase86m
    ogbl_wikikg2
    ogbl_citation2
    ogbl_ppa
    ogb_wikikg90mv2


Custom datasets
-----------------------

.. _custom_dataset_example: http://marius-project.org/marius/examples/config/lp_custom.html#preprocess-dataset

Datasets in delimited file formats such as CSVs can be preprocessed with ``marius_preprocess``

See this `example <custom_dataset_example_>`_.


Usage
-----------------------

.. code-block:: text

    usage: marius_preprocess [-h] [--output_directory output_directory] [--edges edges [edges ...]] [--dataset dataset] [--num_partitions num_partitions] [--partitioned_eval] [--delim delim]
                      [--dataset_split dataset_split [dataset_split ...]] [--overwrite] [--spark] [--no_remap_ids]

    Preprocess built-in datasets and custom link prediction datasets

    optional arguments:
      -h, --help            show this help message and exit
      --output_directory output_directory
                            Directory to put graph data
      --edges edges [edges ...]
                            File(s) containing the edge list(s) for a custom dataset
      --dataset dataset     Name of dataset to preprocess
      --num_partitions num_partitions
                            Number of node partitions
      --partitioned_eval    If true, the validation and/or the test set will be partitioned.
      --delim delim, -d delim
                            Delimiter to use for delimited file inputs
      --dataset_split dataset_split [dataset_split ...], -ds dataset_split [dataset_split ...]
                            Split dataset into specified fractions
      --overwrite           If true, the preprocessed dataset will be overwritten if it already exists
      --spark               If true, pyspark will be used to perform the preprocessing
      --no_remap_ids        If true, the node ids of the input dataset will not be remapped to random integer ids.
      --columns [columns [columns ...]]
                            List of column ids of input delimited files which
                            denote the src node, edge-type, and dst node of edges.
