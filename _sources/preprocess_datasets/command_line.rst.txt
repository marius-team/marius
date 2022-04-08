
The marius_preprocess command can be used to download and preprocess built-in datasets for link prediction and node classification. Custom link prediction datasets can also be preprocessed with this command.

::

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



Preprocessing Built-in Datasets



Preprocess the FB15K-237 knowledge graph
``marius_preprocess --dataset fb15k_237 --output_directory example_dir/``

Preprocess the ogbn_arxiv graph for node classification
``marius_preprocess --dataset ogbn_arxiv --output_directory example_dir/``

Preprocessing custom edge lists for link prediction tasks

Assume we have some input edge list in CSV form

::
    src,rel,dst

``marius_preprocess --dataset ogbn_arxiv --output_directory example_dir/``
