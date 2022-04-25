Custom Dataset Link Prediction
---------------------------------------------

In this tutorial, we use the **OGBN_Arxiv dataset** as an example to demonstrate a step-by-step walkthrough from preprocessing a **custom dataset** to defining the configuration file and to training **a link prediction model with the DistMult algorithm**.

1. Preprocess Dataset
^^^^^^^^^^^^^^^^^^^^^

Preprocessing a custom dataset is straightforward with the ``marius_preprocess`` command. This command comes with ``marius`` when ``marius`` is installed. See (TODO link) for installation information.

Let's start by downloading and extracting the OGBN_Arxiv dataset we will use in this example if it has not been downloaded (assuming we are in the ``marius`` root directory):
 
.. code-block:: bash

   $ wget http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip      # download original dataset
   $ unzip arxiv.zip -d datasets/custom_lp_example/                     # extract downloaded dataset
   $ gzip -dr datasets/custom_lp_example/arxiv/raw/                     # extract raw dataset files
   $ gzip -dr datasets/custom_lp_example/arxiv/split/time/              # extract raw split files

After the previous step, we should have the directory ``datasets/custom_lp_example/arxiv/raw/`` created containing the following raw files downloaded and extracted from the OGBN_Arxiv dataset:

.. code-block:: bash

   $ ls -1 datasets/custom_lp_example/arxiv/raw/ 
    edge.csv                        # raw edge list
    node-feat.csv                   # raw node features
    node-label.csv                  # raw node lables
    node_year.csv  
    num-edge-list.csv  
    num-node-list.csv
   $ head -5 arxiv/raw/edge.csv
    104447,13091
    15858,47283
    107156,69161
    107156,136440
    107156,107366

Assuming ``marius_preprocess`` has been built, we preprocess the OGBN_Arxiv dataset by running the following command (assuming we are in the ``marius`` root directory):

.. code-block:: bash

   $ marius_preprocess --output_dir datasets/custom_lp_example/ 
                        --edges datasets/custom_lp_example/arxiv/raw/edge.csv 
                        --dataset_split 0.8 0.1 0.1 --delim="," --columns 0 1
    Preprocess custom dataset
    Reading edges
    Remapping Edges
    Node mapping written to: datasets/custom_lp/nodes/node_mapping.txt
    Dataset statistics written to: datasets/custom_lp/dataset.yaml

In the above command, we set ``dataset_split`` to a list of ``0.8 0.1 0.1``. Under the hood, this splits ``edge.csv`` into ``edges/train_edges.bin``, ``edges/validation_edges.bin`` and ``edges/test_edges.bin`` based on the given list of fractions.

Note that ``edge.csv`` contains two columns delimited by comma, so we set ``--columns 0,1`` and ``--delim=","``.

The  ``--edges`` flag specifies the raw edge list file that ``marius_preprocess`` will preprocess (and train later).

The  ``--output_directory`` flag specifies where the preprocessed graph will be output and is set by the user. In this example, assume we have not created the datasets/fb15k_237_example repository. ``marius_preprocess`` will create it for us. 

For detailed usages of  ``marius_preprocess``, please execute the following command:

.. code-block:: bash

   $ marius_preprocess -h

Let's check again what was created inside the ``datasets/custom_lp_example/`` directory:

.. code-block:: bash

   $ ls -1 datasets/fb15k_237_example/ 
   dataset.yaml                       # input dataset statistics                                
   nodes/  
     node_mapping.txt                 # mapping of raw node ids to integer uuids
   edges/   
     test_edges.bin                   # preprocessed testing edge list 
     train_edges.bin                  # preprocessed training edge list 
     validation_edges.bin             # preprocessed validation edge list 
   arxiv/                             # existing arxiv dir
     ...  

Let's check what is inside the generated ``dataset.yaml`` file:

.. code-block:: bash

   $ cat datasets/custom_lp_example/dataset.yaml
    dataset_dir: /marius-internal/datasets/custom_lp_example/
    num_edges: 932994
    num_nodes: 169343
    num_relations: 1
    num_train: 932994
    num_valid: 116624
    num_test: 116625
    node_feature_dim: -1
    rel_feature_dim: -1
    num_classes: -1
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

An example YAML configuration file for the OGBN_Arxiv dataset (link prediction model with DistMult) is given in ``examples/configuration/custom_lp.yaml``. Note that the ``dataset_dir`` is set to the preprocessing output directory, in our example, ``datasets/custom_lp_example/``.

Let's create the same YAML configuration file for the OGBN_Arxiv dataset from scratch. We follow the structure of the configuration file and create each of the four sections one by one. In a YAML file, indentation is used to denote nesting and all parameters are in the format of key-value pairs. 

.. note:: 
   String values in the configuration file are case insensitive but we use capital letters for convention.

#. First, we define the **model**. We begin by setting all required parameters. This includes ``learning_task``, ``encoder``, ``decoder``, and ``loss``. The rest of the configurations can be fine-tuned by the user.

    .. code-block:: yaml
    
        model:
          learning_task: LINK_PREDICTION # set the learning task to link prediction
          encoder:
            layers:
              - - type: EMBEDDING # set the encoder to be an embedding table with 50-dimensional embeddings
                  output_dim: 50
          decoder:
            type: DISTMULT # set the decoder to DistMult
            options:
              input_dim: 50
          loss:
            type: SOFTMAX_CE
            options:
              reduction: SUM
          dense_optimizer: # optimizer to use for dense model parameters. In this case these are the DistMult relation (edge-type) embeddings
              type: ADAM
              options:
                learning_rate: 0.1
          sparse_optimizer: # optimizer to use for node embedding table
              type: ADAGRAD
              options:
                learning_rate: 0.1
        storage:
          # omit
        training:
          # omit
        evaluation:
          # omit
      
#. Next, we set the **storage** and **dataset**. We begin by setting all required parameters. This includes ``dataset``. Here, the ``dataset_dir`` is set to ``datasets/custom_lp_example/``, which is the preprocessing output directory.

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          device_type: cuda
          dataset:
            dataset_dir: /marius-internal/datasets/custom_lp_example/
          edges:
            type: DEVICE_MEMORY
          embeddings:
            type: DEVICE_MEMORY
          save_model: true
        training:
          # omit
        evaluation:
          # omit

#. Lastly, we configure **training** and **evaluation**. We begin by setting all required parameters. We begin by setting all required parameters. This includes ``num_epochs`` and ``negative_sampling``. We set ``num_epochs=10`` (10 epochs to train) to demonstrate this example. Note that ``negative_sampling`` is required for link prediction.

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          # omit
        training:
          batch_size: 1000
          negative_sampling:
            num_chunks: 10
            negatives_per_positive: 500
            degree_fraction: 0.0
            filtered: false
          num_epochs: 10
          pipeline:
            sync: true
          epochs_per_shuffle: 1        
        evaluation:
          batch_size: 1000
          negative_sampling:
            filtered: true
          pipeline:
            sync: true   
     
3. Train Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After defining our configuration file, training is run with ``marius_train <your_config.yaml>``.

We can now train our example using the configuration file we just created by running the following command (assuming we are in the ``marius`` root directory):

.. code-block:: bash

   $ marius_train datasets/custom_lp_example/custom_lp.yaml
    [2022-04-04 17:11:53.029] [info] [marius.cpp:45] Start initialization
    [04/04/22 17:11:57.581] Initialization Complete: 4.552s
    [04/04/22 17:11:57.650] ################ Starting training epoch 1 ################
    [04/04/22 17:11:57.824] Edges processed: [94000/932994], 10.08%
    [04/04/22 17:11:57.988] Edges processed: [188000/932994], 20.15%
    [04/04/22 17:11:58.153] Edges processed: [282000/932994], 30.23%
    [04/04/22 17:11:58.317] Edges processed: [376000/932994], 40.30%
    [04/04/22 17:11:58.484] Edges processed: [470000/932994], 50.38%
    [04/04/22 17:11:58.650] Edges processed: [564000/932994], 60.45%
    [04/04/22 17:11:58.817] Edges processed: [658000/932994], 70.53%
    [04/04/22 17:11:59.008] Edges processed: [752000/932994], 80.60%
    [04/04/22 17:11:59.200] Edges processed: [846000/932994], 90.68%
    [04/04/22 17:11:59.408] Edges processed: [932994/932994], 100.00%
    [04/04/22 17:11:59.408] ################ Finished training epoch 1 ################
    [04/04/22 17:11:59.408] Epoch Runtime: 1758ms
    [04/04/22 17:11:59.408] Edges per Second: 530713.3
    [04/04/22 17:11:59.408] Evaluating validation set
    [04/04/22 17:12:00.444]
    =================================
    Link Prediction: 116624 edges evaluated
    Mean Rank: 10927.984317
    MRR: 0.088246
    Hits@1: 0.043936
    Hits@3: 0.091285
    Hits@5: 0.123697
    Hits@10: 0.176499
    Hits@50: 0.337538
    Hits@100: 0.414872
    =================================
    [04/04/22 17:12:00.444] Evaluating test set
    [04/04/22 17:12:01.470]
    =================================
    Link Prediction: 116625 edges evaluated
    Mean Rank: 10928.291687
    MRR: 0.088237
    Hits@1: 0.043798
    Hits@3: 0.091670
    Hits@5: 0.123190
    Hits@10: 0.176377
    Hits@50: 0.337749
    Hits@100: 0.414697
    =================================

After running this configuration for 10 epochs, we should see a result similar to below:

.. code-block:: bash

    =================================
    [04/04/22 17:12:32.312] ################ Starting training epoch 10 ################
    [04/04/22 17:12:32.475] Edges processed: [94000/932994], 10.08%
    [04/04/22 17:12:32.638] Edges processed: [188000/932994], 20.15%
    [04/04/22 17:12:32.800] Edges processed: [282000/932994], 30.23%
    [04/04/22 17:12:32.963] Edges processed: [376000/932994], 40.30%
    [04/04/22 17:12:33.126] Edges processed: [470000/932994], 50.38%
    [04/04/22 17:12:33.313] Edges processed: [564000/932994], 60.45%
    [04/04/22 17:12:33.500] Edges processed: [658000/932994], 70.53%
    [04/04/22 17:12:33.666] Edges processed: [752000/932994], 80.60%
    [04/04/22 17:12:33.835] Edges processed: [846000/932994], 90.68%
    [04/04/22 17:12:33.988] Edges processed: [932994/932994], 100.00%
    [04/04/22 17:12:33.988] ################ Finished training epoch 10 ################
    [04/04/22 17:12:33.988] Epoch Runtime: 1676ms
    [04/04/22 17:12:33.988] Edges per Second: 556679
    [04/04/22 17:12:33.988] Evaluating validation set
    [04/04/22 17:12:35.010]
    =================================
    Link Prediction: 116624 edges evaluated
    Mean Rank: 5765.685716
    MRR: 0.132049
    Hits@1: 0.048926
    Hits@3: 0.149883
    Hits@5: 0.210797
    Hits@10: 0.304637
    Hits@50: 0.536768
    Hits@100: 0.626072
    =================================
    [04/04/22 17:12:35.011] Evaluating test set
    [04/04/22 17:12:36.034]
    =================================
    Link Prediction: 116625 edges evaluated
    Mean Rank: 5797.073741
    MRR: 0.132749
    Hits@1: 0.049406
    Hits@3: 0.151588
    Hits@5: 0.211944
    Hits@10: 0.304437
    Hits@50: 0.536549
    Hits@100: 0.626006
    =================================


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

.. note:: 
    ``model.pt`` contains the dense model parameters. For DistMult, this is the embeddings of the edge-types. For GNN encoders, this file will include the GNN parameters.

4. Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.1 Command Line
""""""""""""""""

4.2 Load Into Python
""""""""""""""""""""
