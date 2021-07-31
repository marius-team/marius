.. _examples:

********
Examples
********

This section contains several end-to-end tutorials on how to deploy Marius training pipeline 
for various applications and two showcases demonstrating the training over different datasets using different 
configuration combinations of Marius based on different needs.
For each tutorial, there is a real-life dataset and the complete process of
preprocessing, training, evaluation and postprocessing over the dataset. 
The two showcases are presented with showcase datasets of different formats and 
the corresponding configurations for training over the data.


Showcases
=========

This is a showcase demonstrates how to deploy Marius on a dataset with only one edge type.
The ``num_relations`` is set to one since there is only one edge type in this relation.

==========  ======================  ==========
Entities    Relations (edge-types)  Edges  
----------  ----------------------  ----------
4847571     1                       68993773
==========  ======================  ==========


==================  =======================
Source                        Destination
------------------  -----------------------
0                   1
1                   1413125
2                   59129
...                                ...
==================  =======================

Marius configurations:

::

    [general]
    device=GPU
    num_train=62094395
    num_nodes=4847571
    num_relations=1
    num_valid=3449689
    num_test=3449689

    [path]
    base_directory=data/
    train_edges=./output_dir/train_edges.pt
    validation_edges=./output_dir/valid_edges.pt
    test_edges=./output_dir/test_edges.pt
    node_ids=./output_dir/node_mapping.bin
    relations_ids=./output_dir/rel_mapping.bin



The following showcase demonstrates how to deploy Marius on a dataset with multiple edge types.
The ``num_relations`` is set to eighteen which indicates that there are eighteen edge types 
within this showcase dataset.
To improve the accuracy of the embeddings, a larger embedding size ``embedding_size=256`` can be set.
The decoder scoring function ``TransE`` could capture more graph structure information over certain dataset.
For this showcase, ``decoder=TransE`` is set. The size of this showcase dataset is relatively small and can fit
in the GPU memory. Therefore, set ``synchronous=true`` in the training section to perform synchronous training
without the usage of pipeline could be a good choice.

==========  ======================  ==========
Entities    Relations (edge-types)  Edges  
----------  ----------------------  ----------
40943       18                      151442
==========  ======================  ==========

==================  ==================  =======================
Source              Relation            Destination
------------------  ------------------  -----------------------
__wisconsin_NN_2    _instance_hypernym  __madison_NN_2
__scandinavia_NN_2  _member_meronym     __sweden_NN_1
__kobenhavn_NN_1    _instance_hypernym  __national_capital_NN_1
...                 ...                 ...
==================  ==================  =======================


Marius configurations:

::

    [general]
    device=GPU
    num_train=141442
    num_nodes=40943
    num_relations=18
    num_valid=5000
    num_test=5000

    [model]
    embedding_size=256
    decoder=TransE

    [training]
    synchronous=true

    [path]
    base_directory=data/
    train_edges=./output_dir/train_edges.pt
    validation_edges=./output_dir/valid_edges.pt
    test_edges=./output_dir/test_edges.pt
    node_ids=./output_dir/node_mapping.bin
    relations_ids=./output_dir/rel_mapping.bin