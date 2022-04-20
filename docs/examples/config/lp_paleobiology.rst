.. _lp_paleo:

Paleobiology Dataset Link Prediction
---------------------------------------------
In this tutorial, we will use Marius to perform link prediction on a paleobiology knowledge graph dataset, i.e. predicting the existence of an edge between two nodes in a graph. This will cover the end-to-end process of downloading a dataset, running Marius to learn the embeddings of all the nodes and edges in the graph, and then using these embeddings to infer new links and make discoveries.

Dataset Information
^^^^^^^^^^^^^^^^^^^^^
In our example we will train on a graph-structured paleobiology dataset which contains information about fossils and their relations to Earth. In this knowledge graph dataset, nodes represent different types of entities and directed edges are relationships between them. This dataset contains 14,752 nodes and 107,247 edges, with 5 possible relation types.

The **nodes/entities** in this graph fall into 10 different types. They are:

1. Country
2. State
3. County
4. Lithology
5. Formation
6. Geological Group
7. Member
8. Taxon
9. Environment
10. Geological Interval

The directed edges/relations in this graph fall into 5 categories:

1. Consist of
2. Collected from
3. Located in
4. Assigned to
5. Interpretted as

Every triplet in this knowledge graph dataset follows the structure of

``<[Source Node], [Relation], [Target Node]>``

For example, the triplet

``<47880_taxon, collected_from, Wisconsin_state>``

signifies a directed edge of type ``collected from`` pointing from the taxon (i.e. biological group) node with an ID number of 47880 to the state node ``wisconsin_state``. With our dataset comes a taxon ID lookup table, from which we can see that the ID of 47880 represents Mammuthus, or mammoths. Semantically, this triplet means that mammoth fossils have been collected from Wisconsin.

The goal of generating embeddings for the edges in this graph will be to use them to predict gaps in our knowledge base. After training our embeddings, we can fix a target node and relation type and predict potential source nodes. For example, we can fix our taxon type and relation type and make a query for potential source nodes, i.e. possible locations mammoths could be collected from which were not present in our knowledge base.

``<47880_taxon, collected_from, ?>``

Using our embeddings, we can find the source node(s) with the highest probability of existing. If the probability is higher than some threshold, we can say that these predicted links should be considered true.

1. Download the Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To download the dataset and helper files, clone the following repository into the directory of your choosing::

    git clone https://github.com/marius-team/marius-examples.git

Enter the link prediction example directory::

    cd marius-examples/link-predict-example

This contains our paleobiology dataset, located in ``dataset/``, along with a sample configuration file ``paleo_config.yaml`` and link prediction Python script predict.py.

2. Train Model with Marius
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We assume Marius has been downloaded and installed with PIP::

    git clone https://github.com/marius-team/marius.git
    cd marius; python3 -m pip install .

**Step 1: Define a Configuration File**

Before we can run Marius we need to specify our model hyperparameters and path to our preprocessed graph dataset. The following configuration file, ``paleo_config.yaml``, is included in ``marius-examples/link-predict-example``. It contains standard options for link prediction. Note a few important parameters that might need to be changed based on your system:

``storage.device_type`` : We assume training on GPU. If using CPU only, switch from ``cuda`` to ``CPU``.

``training.num_epochs``: The number of epochs to train for. Adjust higher or lower based on the desired accuracy of the embeddings. Our default is ``100`` epochs.

``storage.dataset.dataset_dir``: We assume we are in the link prediction example directory ``~/marius-examples/link-predict-example/`` when running the training process, so our default relative path is ``dataset/``. Change if running from another directory.

``paleo_config.yaml``

.. code-block:: yaml

        model:
            learning_task: LINK_PREDICTION
            encoder:
                layers:
                - - type: EMBEDDING
                    output_dim: 50
            decoder:
                type: COMPLEX
            loss:
                type: SOFTMAX_CE
            sparse_optimizer:
                type: ADAGRAD
                options:
                learning_rate: 0.1
        storage:
            device_type: cuda
            dataset:
                dataset_dir: dataset/
                num_edges: 96522
                num_train: 96522
                num_nodes: 14752
                num_relations: 5
                num_valid: 5362
                num_test: 5363
            edges:
                type: DEVICE_MEMORY
            embeddings:
                type: DEVICE_MEMORY
            save_model: true
        training:
            batch_size: 1000
            negative_sampling:
                num_chunks: 100
                negatives_per_positive: 512
                degree_fraction: 0.0
                filtered: false
            num_epochs: 100
            pipeline:
                sync: true
            epochs_per_shuffle: 1
        evaluation:
            batch_size: 1000
            negative_sampling:
                filtered: true
            pipeline:
                sync: true

**Step 2: Run Marius**

Now that we have a configuration file and dataset ready, we simply need to run the training executable with our config file as the argument.::

    marius_train paleo_config.yaml

The output should appear similar to::

    [info] [marius.cpp:45] Start initialization
    Initialization Complete: 4.424s
    ################ Starting training epoch 1 ################
    Edges processed: [10000/96522], 10.36%
    Edges processed: [20000/96522], 20.72%
    Edges processed: [30000/96522], 31.08%
    Edges processed: [40000/96522], 41.44%
    Edges processed: [50000/96522], 51.80%
    Edges processed: [60000/96522], 62.16%
    Edges processed: [70000/96522], 72.52%
    Edges processed: [80000/96522], 82.88%
    Edges processed: [90000/96522], 93.24%
    Edges processed: [96522/96522], 100.00%
    ################ Finished training epoch 1 ################
    Epoch Runtime: 527ms
    Edges per Second: 183153.7
    Evaluating validation set
    =================================
    Link Prediction: 10724 edges evaluated
    Mean Rank: 1426.696568
    MRR: 0.115575
    Hits@1: 0.058653
    Hits@3: 0.128683
    Hits@5: 0.169153
    Hits@10: 0.229952
    Hits@50: 0.392111
    Hits@100: 0.459437
    =================================

After this has finished, our output will be in our ``[dataset_dir]`` (using the provided config, this will be ``dataset/``.

Here are the files that were created in training:
Let's check again what was added in the ``datasets/custom_lp_example/`` directory. For clarity, we only list the files that were created in training. Notice that several files have been created, including the trained model, the embedding table, a full configuration file, and output logs:

.. code-block:: bash

   $ ls datasets/custom_lp_example/ 
   model.pt                           # contains the dense model parameters, embeddings of the edge-types
   model_state.pt                     # optimizer state of the trained model parameters
   full_config.yaml                   # detailed config generated based on user-defined config
   metadata.csv                       # information about metadata
   logs/                              # logs containing output, error, debug information, and etc.
   nodes/  
     embeddings.bin                   # trained node embeddings of the graph
     embeddings_state.bin             # node embedding optimizer state
     ...
   edges/   
     ...
   ...

3. Inference with Python: Using Embeddings for Link Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Now that we have generated our embeddings for the dataset, we can perform link prediction to infer new edges in the graph. As previously mentioned, we can fix a target node and relation type and predict potential source nodes.

.. Let's use our prediction tool that uses the embeddings to predict the top K most likely destination nodes for a given source and relation. For this example, we will set the source node to ``29741_taxon`` and the relation to ``collected_from``.

.. ``<29741_taxon, collected_from, ?>``

.. Semantically, we are predicting where Taxon #29741, which represents the fossil Whitfieldella (a type of extint mollusk) could be collected from. If our embeddings are accurate, we should not only find existing edges in our output but also new ones, leading to the discovery of possible new places to search for these fossils.

.. change from here once marius_infer updated
We will use the Marius inference tool ``marius_infer`` to perform link prediction with our trained model. In ``~marius-examples/link-predict-example/``, run::

    marius_predict --config paleo_config.yaml --output_dir results/ --metrics mrr mean_rank hits1 hits10 hits50 --save_scores --save_ranks

This tool takes our config, an output directory, and our desired metrics as input, and perform link prediction evaluation over the test set of edges provided in the config file. Metrics are saved to ``results/metrics.txt`` and scores and ranks for each test edge are saved to ``results/scores.csv``.

.. [TODO: CHANGE FROM PREDICT.PY TO  ``marius_predict``]

.. Our output should appear similar to the following

..     ['Iowa_state',
..     'limestone_lithology',
..     'United States_country',
..     'Hopkinton Dolomite_formation',
..     'dolomite_lithology',
..     'Yushan_county',
..     'Jiangxi_state',
..     'Dubuque_county',
..     'China_country',
..     'Shiyang_formation',
..     'JiangShan_county',
..     'Cordell_formation',
..     'shale_lithology',
..     'mudstone_lithology',
..     'South Carolina_state']

.. We've successfully predicted destination nodes! Most of these edges already exist in the graph, proving the accuracy of our embeddings. However, new edges also appear, such as the edge to ``Cordell_formation``. Outside experts have confirmed that this prediction is accurate, i.e. Whitfieldella can be found at the Cordell Formation, a geologic formation in Michigan. This example shows how using the embeddings generated by Marius, we can perform link prediction on a graph and make new discoveries. 

