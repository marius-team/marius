Small Scale Node Classification (OGBN-Arxiv)
---------------------------------------------

In this tutorial, we use the **OGBN-Arxiv dataset** as an example to demonstrate a step-by-step walkthrough from preprocessing the dataset to defining the configuration file and to training **a node classification with 3-layer GraphSage model**.

1. Preprocess Dataset
^^^^^^^^^^^^^^^^^^^^^

Preprocessing a dataset is straightforward with the ``marius_preprocess`` command. This command comes with ``marius`` when ``marius`` is installed. See (TODO link) for installation information.

Assuming ``marius_preprocess`` has been built, we preprocess the OGBN-Arxiv dataset by running the following command (assuming we are in the ``marius`` root directory):

.. code-block:: bash

   $ marius_preprocess --dataset ogbn_arxiv --output_directory datasets/ogbn_arxiv_example/
   Downloading arxiv.zip to datasets/ogbn_arxiv_example/arxiv.zip
   Reading edges
   Remapping Edges
   Node mapping written to: datasets/ogbn_arxiv_example/nodes/node_mapping.txt
   Dataset statistics written to: datasets/ogbn_arxiv_example/dataset.yaml

The  ``--dataset`` flag specifies which of the pre-set datasets ``marius_preprocess`` will preprocess and download.

The  ``--output_directory`` flag specifies where the preprocessed graph will be output and is set by the user. In this example, assume we have not created the ``datasets/ogbn_arxiv_example/`` repository, ``marius_preprocess`` will create it for us. 

For detailed usages of  ``marius_preprocess``, please execute the following command:

.. code-block:: bash

   $ marius_preprocess -h

Let's check what is inside the created directory:

.. code-block:: bash

   $ ls -1 datasets/ogbn_arxiv_example/ 
   dataset.yaml                       # input dataset statistics                                
   nodes/  
     node_mapping.txt                 # mapping of raw node ids to integer uuids
     features.bin                     # preprocessed features list
     labels.bin                       # preprocessed labels list
     test_nodes.bin                   # preprocessed testing nodes list
     train_nodes.bin                  # preprocessed training nodes list
     validation_nodes.bin             # preprocessed validation nodes list
   edges/   
     train_edges.bin                  # preprocessed training edge list
   arxiv/                             # dir with provided source files of the downloaded OGBN-Arxiv dataset
     RELEASE_v1.txt  
     mapping/  
     processed/  
     raw/  
     split/
   edge.csv                           # raw edge list
   train.csv                          # raw training edge list                                              
   test.csv                           # raw testing edge list    
   valid.csv                          # raw validation edge list    
   node-feat.csv                      # node features
   node-label.csv                     # node labels
   README.txt                         # README of the downloaded OGBN-Arxiv dataset
   arxvi.zip                          # downloaded OGBN-Arxiv dataset


Let's check what is inside the generated ``dataset.yaml`` file:

.. code-block:: bash

   $ cat datasets/ogbn_arxiv_example/dataset.yaml
   dataset_dir: /marius-internal/datasets/ogbn_arxiv_example/
   num_edges: 1166243
   num_nodes: 169343
   num_relations: 1
   num_train: 90941
   num_valid: 29799
   num_test: 48603
   node_feature_dim: 128
   rel_feature_dim: -1
   num_classes: 40
   initialized: false


.. note:: 
   If the above ``marius_preprocess`` command fails due to any missing directory errors, please create the ``<output_directory>/edges`` and ``<output_directory>/nodes`` directories as a workaround.

2. Define Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model, we need to define a YAML configuration file based on information created from marius_preprocess. 

The configuration file contains information including but not limited to the inputs to the model, training procedure, and hyperparameters to optimize. Given a configuration file, marius assembles a model depending on the given parameters. The configuration file is grouped up into four sections:

* Model: Defines the architecture of the model, neighbor sampling configuration, loss, and optimizer(s)
* Storage: Specifies the input dataset and how to store the graph, features, and embeddings.
* Training: Sets options for the training procedure and hyperparameters. E.g. batch size, negative sampling.
* Evaluation: Sets options for the evaluation procedure (if any). The options here are similar to those in the training section.

For the full configuration schema, please refer to ``docs/config_interface``.

An example YAML configuration file for the OGBN_Arxiv dataset is given in ``examples/configuration/ogbn_arxiv.yaml``. Note that the ``dataset_dir`` is set to the preprocessing output directory, in our example, ``datasets/ogbn_arxiv_example/``.

Let's create the same YAML configuration file for the OGBN_Arxiv dataset from scratch. We follow the structure of the configuration file and create each of the four sections one by one. In a YAML file, indentation is used to denote nesting and all parameters are in the format of key-value pairs. 

#. | First, we define the **model**. We begin by setting all required parameters. This includes ``learning_task``, ``encoder``, ``decoder``, and ``loss``.
   | Note that the output of the encoder is the output label vector for a given node. (E.g. For node classification with 5 classes, the output label vector from the encoder might look like this: [.05, .2, .8, .01, .03]. In this case, an argmax will return a class label of 2 for the node.) The rest of the configurations can be fine-tuned by the user.

    .. code-block:: yaml
    
        model:
          learning_task: NODE_CLASSIFICATION # set the learning task to node classification
          encoder:
            train_neighbor_sampling:
              - type: ALL
              - type: ALL
              - type: ALL
            layers: # define three layers of GNN of type GRAPH_SAGE
              - - type: FEATURE
                  output_dim: 128 # set to 128 (to match "node_feature_dim=128" in "dataset.yaml") for each layer except for the last
                  bias: true
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 128 # set to 128 (to match "node_feature_dim=128" in "dataset.yaml") for each layer except for the last
                  output_dim: 128
                  bias: true
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 128
                  output_dim: 128
                  bias: true
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 128
                  output_dim: 40 # set "output_dim" to 40 (to match "num_classes=40") in "dataset.yaml" for the last layer
                  bias: true
          decoder:
            type: NODE
          loss:
            type: CROSS_ENTROPY
            options:
              reduction: SUM
          dense_optimizer:
            type: ADAM
            options:
              learning_rate: 0.01
        storage:
          # omit
        training:
          # omit
        evaluation:
          # omit
      
#. | Next, we set the **storage** and **dataset**. We begin by setting all required parameters. This includes ``dataset``. Here, the ``dataset_dir`` is set to ``datasets/ogbn_arxiv_example/``, which is the preprocessing output directory.

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          device_type: cuda
          dataset:
            dataset_dir: datasets/ogbn_arxiv_example/
          edges:
            type: DEVICE_MEMORY
            options:
              dtype: int
          features:
            type: DEVICE_MEMORY
            options:
              dtype: float
        training:
          # omit
        evaluation:
          # omit

#. Lastly, we configure **training** and **evaluation**. We begin by setting all required parameters. This includes ``num_epochs``. We set ``num_epochs=10`` (10 epochs to train) to demonstrate this example. 

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          # omit
        training:
          batch_size: 1000
          num_epochs: 10
          pipeline:
            sync: true
        evaluation:
          batch_size: 1000
          pipeline:
            sync: true
     
3. Train Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After defining our configuration file, training is run with ``marius_train <your_config.yaml>``.

We can now train our example using the configuration file we just created by running the following command (assuming we are in the ``marius`` root directory):

.. code-block:: bash

   $ marius_train datasets/ogbn_arxiv_example/ogbn_arxiv.yaml
    [2022-04-05 18:50:11.677] [info] [marius.cpp:45] Start initialization
    [04/05/22 18:50:15.807] Initialization Complete: 4.13s
    [04/05/22 18:50:15.877] ################ Starting training epoch 1 ################
    [04/05/22 18:50:16.310] Nodes processed: [10000/90941], 11.00%
    [04/05/22 18:50:16.753] Nodes processed: [20000/90941], 21.99%
    [04/05/22 18:50:17.192] Nodes processed: [30000/90941], 32.99%
    [04/05/22 18:50:17.641] Nodes processed: [40000/90941], 43.98%
    [04/05/22 18:50:18.089] Nodes processed: [50000/90941], 54.98%
    [04/05/22 18:50:18.538] Nodes processed: [60000/90941], 65.98%
    [04/05/22 18:50:18.983] Nodes processed: [70000/90941], 76.97%
    [04/05/22 18:50:19.424] Nodes processed: [80000/90941], 87.97%
    [04/05/22 18:50:19.861] Nodes processed: [90000/90941], 98.97%
    [04/05/22 18:50:19.904] Nodes processed: [90941/90941], 100.00%
    [04/05/22 18:50:19.904] ################ Finished training epoch 1 ################
    [04/05/22 18:50:19.904] Epoch Runtime: 4027ms
    [04/05/22 18:50:19.904] Nodes per Second: 22582.816
    [04/05/22 18:50:19.904] Evaluating validation set
    [04/05/22 18:50:20.795]
    =================================
    Node Classification: 29799 nodes evaluated
    Accuracy: 65.753884%
    =================================
    [04/05/22 18:50:20.795] Evaluating test set
    [04/05/22 18:50:22.194]
    =================================
    Node Classification: 48603 nodes evaluated
    Accuracy: 63.909635%
    =================================


After running this configuration for 10 epochs, we should see a result similar to below with arruracy roughly equal to 67%:

.. code-block:: bash

    =================================
    [04/05/22 18:51:12.589] ################ Starting training epoch 10 ################
    [04/05/22 18:51:13.024] Nodes processed: [10000/90941], 11.00%
    [04/05/22 18:51:13.456] Nodes processed: [20000/90941], 21.99%
    [04/05/22 18:51:13.889] Nodes processed: [30000/90941], 32.99%
    [04/05/22 18:51:14.336] Nodes processed: [40000/90941], 43.98%
    [04/05/22 18:51:14.789] Nodes processed: [50000/90941], 54.98%
    [04/05/22 18:51:15.240] Nodes processed: [60000/90941], 65.98%
    [04/05/22 18:51:15.678] Nodes processed: [70000/90941], 76.97%
    [04/05/22 18:51:16.119] Nodes processed: [80000/90941], 87.97%
    [04/05/22 18:51:16.556] Nodes processed: [90000/90941], 98.97%
    [04/05/22 18:51:16.599] Nodes processed: [90941/90941], 100.00%
    [04/05/22 18:51:16.599] ################ Finished training epoch 10 ################
    [04/05/22 18:51:16.599] Epoch Runtime: 4010ms
    [04/05/22 18:51:16.599] Nodes per Second: 22678.553
    [04/05/22 18:51:16.599] Evaluating validation set
    [04/05/22 18:51:17.485]
    =================================
    Node Classification: 29799 nodes evaluated
    Accuracy: 69.445283%
    =================================
    [04/05/22 18:51:17.485] Evaluating test set
    [04/05/22 18:51:18.882]
    =================================
    Node Classification: 48603 nodes evaluated
    Accuracy: 68.078102%
    =================================


Let's check again what was added in the ``datasets/ogbn_arxiv_example/`` directory. For clarity, we only list the files that were created in training. Notice that several files have been created, including the trained model, the embedding table, a full configuration file, and output logs:

.. code-block:: bash

   $ ls -1 datasets/ogbn_arxiv_example/ 
   model.pt                           # contains the dense model parameters, including the GNN parameters
   model_state.pt                     # optimizer state of the trained model parameters
   full_config.yaml                   # detailed config generated based on user-defined config
   metadata.csv                       # information about metadata
   logs/                              # logs containing output, error, debug information, and etc.
   nodes/  
     ...
   edges/   
     ...
   ...

.. note::
  ``model.pt`` contains the dense model parameters. For GNN encoders, this file will include the GNN parameters.

4. Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1 Command Line
""""""""""""""""

4.2 Load Into Python
""""""""""""""""""""
