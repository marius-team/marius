Custom Dataset Node Classification
---------------------------------------------
In this tutorial, we use the **Cora dataset** as an example to demonstrate a step-by-step walkthrough from preprocessing the dataset to defining the configuration file and to training **a node classification with 3-layer GraphSage model**.

1. Preprocess Dataset
^^^^^^^^^^^^^^^^^^^^^

Preprocessing a custom dataset is straightforward with the help of Marius python API. Preprocessing using the Marius Python API requires creating a custom Dataset class of type ``NodeClassificationDataset`` or ``LinkPredictionDataset``. An example python script which preprocesses, trains, and evaluates the Cora dataset is provided in ``examples/python/custom_nc_graphsage.py``. For detailed steps, please refer to (link).

Let's borrow the provided ``examples/python/custom_nc_graphsage.py`` and modify it to suit our purpose. We first ``download()`` the dataset to ``datasets/custom_nc_example/cora/``, then ``preprocess()``,. Note that the ``MYDATASET`` class is a child class of ``NodeClassificationDataset``: 

.. code-block:: python

    import marius as m
    import torch
    from omegaconf import OmegaConf

    import numpy as np
    import pandas as pd

    from pathlib import Path

    from marius.tools.preprocess.dataset import NodeClassificationDataset
    from marius.tools.preprocess.utils import download_url, extract_file
    from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
    from marius.tools.configuration.constants import PathConstants
    from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes

    def switch_to_num(row):
        names = ['Neural_Networks', 'Rule_Learning', 'Reinforcement_Learning', 'Probabilistic_Methods',\
                'Theory', 'Genetic_Algorithms', 'Case_Based']
        idx = 0
        for i in range(len(names)):
            if (row == names[i]):
                idx = i
                break
        
        return idx

    class MYDATASET(NodeClassificationDataset):
        
        def __init__(self, output_directory: Path, spark=False):

            super().__init__(output_directory, spark)

            self.dataset_name = "cora"
            self.dataset_url = "http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz"
        
        def download(self, overwrite=False):

            # These are the files that I want to make my the end of the the download
            self.input_edge_list_file = self.output_directory / Path("edge.csv")
            self.input_node_feature_file = self.output_directory / Path("node-feat.csv")
            self.input_node_label_file = self.output_directory / Path("node-label.csv")
            self.input_train_nodes_file = self.output_directory / Path("train.csv")
            self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
            self.input_test_nodes_file = self.output_directory / Path("test.csv")

            download = False
            if not self.input_edge_list_file.exists():
                download = True
            if not self.input_node_feature_file.exists():
                download = True
            if not self.input_node_label_file.exists():
                download = True
            if not self.input_train_nodes_file.exists():
                download = True
            if not self.input_valid_nodes_file.exists():
                download = True
            if not self.input_test_nodes_file.exists():
                download = True
            
            if download:
                archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
                extract_file(archive_path, remove_input=False)

                # Reading and processing the csv
                df = pd.read_csv(dataset_dir / Path("cora/cora.content"), sep="\t", header=None)
                cols = df.columns[1:len(df.columns)-1]

                # Getting all the indices
                indices = np.array(range(len(df)))
                np.random.shuffle(indices)
                train_indices = indices[0:int(0.8*len(df))]
                valid_indices = indices[int(0.8*len(df)):int(0.8*len(df))+int(0.1*len(df))]
                test_indices = indices[int(0.8*len(df))+int(0.1*len(df)):]

                np.savetxt(dataset_dir / Path("train.csv"), train_indices, delimiter=",", fmt="%d")
                np.savetxt(dataset_dir / Path("valid.csv"), valid_indices, delimiter=",", fmt="%d")
                np.savetxt(dataset_dir / Path("test.csv"), test_indices, delimiter=",", fmt="%d")


                # Features
                features = df[cols]
                features.to_csv(index=False, sep=",", path_or_buf = dataset_dir / Path("node-feat.csv"), header=False)

                # Labels
                labels = df[df.columns[len(df.columns)-1]]
                labels = labels.apply(switch_to_num)
                labels.to_csv(index=False, sep=",", path_or_buf = dataset_dir / Path("node-label.csv"), header=False)

                # Edges
                node_ids = df[df.columns[0]]
                dict_reverse = node_ids.to_dict()
                nodes_dict = {v: k for k, v in dict_reverse.items()}
                df_edges = pd.read_csv(dataset_dir / Path("cora/cora.cites"), sep="\t", header=None)
                df_edges.replace({0: nodes_dict, 1: nodes_dict},inplace=True)
                df_edges.to_csv(index=False, sep=",", path_or_buf = dataset_dir / Path("edge.csv"), header=False)

            
        def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False):
            train_nodes = np.genfromtxt(self.input_train_nodes_file, delimiter=",").astype(np.int32)
            valid_nodes = np.genfromtxt(self.input_valid_nodes_file, delimiter=",").astype(np.int32)
            test_nodes = np.genfromtxt(self.input_test_nodes_file, delimiter=",").astype(np.int32)

            converter = TorchEdgeListConverter(
                output_dir=self.output_directory,
                train_edges=self.input_edge_list_file,
                num_partitions=num_partitions,
                src_column = 0,
                dst_column = 1,
                remap_ids=remap_ids,
                sequential_train_nodes=sequential_train_nodes,
                delim=",",
                known_node_ids=[train_nodes, valid_nodes, test_nodes],
                partitioned_evaluation=partitioned_eval
            )
            dataset_stats = converter.convert()

            features = np.genfromtxt(self.input_node_feature_file, delimiter=",").astype(np.float32)
            labels = np.genfromtxt(self.input_node_label_file, delimiter=",").astype(np.int32)

            if remap_ids:
                node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
                train_nodes, valid_nodes, test_nodes, features, labels = remap_nodes(node_mapping, train_nodes, valid_nodes, test_nodes, features, labels)

            with open(self.train_nodes_file, "wb") as f:
                f.write(bytes(train_nodes))
            with open(self.valid_nodes_file, "wb") as f:
                f.write(bytes(valid_nodes))
            with open(self.test_nodes_file, "wb") as f:
                f.write(bytes(test_nodes))
            with open(self.node_features_file, "wb") as f:
                f.write(bytes(features))
            with open(self.node_labels_file, "wb") as f:
                f.write(bytes(labels))

            # update dataset yaml
            dataset_stats.num_train = train_nodes.shape[0]
            dataset_stats.num_valid = valid_nodes.shape[0]
            dataset_stats.num_test = test_nodes.shape[0]
            dataset_stats.node_feature_dim = features.shape[1]
            dataset_stats.num_classes = 40

            dataset_stats.num_nodes = dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_test

            with open(self.output_directory / Path("dataset.yaml"), "w") as f:
                yaml_file = OmegaConf.to_yaml(dataset_stats)
                f.writelines(yaml_file)

            return

    if __name__ == '__main__':
        # initialize and preprocess dataset
        dataset_dir = Path("datasets/custom_nc_example/cora/") # note that we write to this directory
        dataset = MYDATASET(dataset_dir)
        if not (dataset_dir / Path("edges/train_edges.bin")).exists():
            dataset.download()
            dataset.preprocess()

We preprocess the Cora dataset by running the ollowing command (assuming we are in the ``marius`` root directory):

.. code-block:: bash

   $ python datasets/custom_nc_example/custom_nc_graphsage.py 
    Downloading cora.tgz to cora/cora.tgz
    Reading edges
    Remapping Edges
    Node mapping written to: cora/nodes/node_mapping.txt
    Dataset statistics written to: cora/dataset.yaml

In this example, assume we have not created the ``datasets/custom_nc_example/cora/`` repository, ``custom_nc_graphsage.py`` will create it for us. 

For detailed usages of Marius python API, please refer to (link).

Let's check what is inside the created directory:

.. code-block:: bash

   $ ls -1 datasets/custom_nc_example/cora/
   dataset.yaml                       # input dataset statistics                                
   nodes/  
     node_mapping.txt                 # mapping of raw node ids to integer uuids
     features.bin                     # preprocessed features list
     labels.bin                       # preprocessed labels list
     test_nodes.bin                   # preprocessed testing nodes list
     train_nodes.bin                  # preprocessed training nodes list
     validation_nodes.bin             # preprocessed validation nodes list
   edges/   
     train_edges.bin                  # mapping of raw edge(relation) ids to integer uuids
   cora/                              # downloaded source files
     ...
   edge.csv                           # raw edge list
   train.csv                          # raw training edge list                                              
   test.csv                           # raw testing edge list    
   valid.csv                          # raw validation edge list    
   node-feat.csv                      # node features
   node-label.csv                     # node labels
   cora.tgz                           # downloaded Cora dataset


Let's check what is inside the generated ``dataset.yaml`` file:

.. code-block:: bash

   $ cat datasets/ogbn_arxiv_example/dataset.yaml
    dataset_dir: /marius-internal/datasets/custom_nc_example/cora/
    num_edges: 5429
    num_nodes: 2708
    num_relations: 1
    num_train: 2166
    num_valid: 270
    num_test: 272
    node_feature_dim: 1433
    rel_feature_dim: -1
    num_classes: 40
    initialized: false


2. Define Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train a model, we need to define a YAML configuration file based on information created from the preprocessing python script. 

The configuration file contains information including but not limited to the inputs to the model, training procedure, and hyperparameters to optimize. Given a configuration file, marius assembles a model depending on the given parameters. The configuration file is grouped up into four sections:

* Model: Defines the architecture of the model, neighbor sampling configuration, loss, and optimizer(s)
* Storage: Specifies the input dataset and how to store the graph, features, and embeddings.
* Training: Sets options for the training procedure and hyperparameters. E.g. batch size, negative sampling.
* Evaluation: Sets options for the evaluation procedure (if any). The options here are similar to those in the training section.

For the full configuration schema, please refer to ``docs/config_interface``.

An example YAML configuration file for the Cora dataset is given in ``examples/configuration/custom_nc.yaml``. Note that the ``dataset_dir`` is set to the preprocessing output directory, in our example, ``datasets/custom_nc_example/cora/``.

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
                  output_dim: 1433 # set to 1433 (to match "node_feature_dim=1433" in "dataset.yaml") for each layer except for the last
                  bias: true
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 1433 # set to 1433 (to match "node_feature_dim=1433" in "dataset.yaml") for each layer except for the last
                  output_dim: 1433
                  bias: true
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 1433
                  output_dim: 1433
                  bias: true
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 1433
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
      
#. | Next, we set the **storage** and **dataset**. We begin by setting all required parameters. This includes ``dataset``. Here, the ``dataset_dir`` is set to ``datasets/custom_nc_example/cora/``, which is the preprocessing output directory.

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          device_type: cuda
          dataset:
            dataset_dir: datasets/custom_nc_example/cora/
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

   $ marius_train datasets/custom_nc_example/cora/custom_nc.yaml
    [2022-04-05 18:41:44.987] [info] [marius.cpp:45] Start initialization
    [04/05/22 18:41:49.122] Initialization Complete: 4.134s
    [04/05/22 18:41:49.135] ################ Starting training epoch 1 ################
    [04/05/22 18:41:49.161] Nodes processed: [1000/2166], 46.17%
    [04/05/22 18:41:49.180] Nodes processed: [2000/2166], 92.34%
    [04/05/22 18:41:49.199] Nodes processed: [2166/2166], 100.00%
    [04/05/22 18:41:49.199] ################ Finished training epoch 1 ################
    [04/05/22 18:41:49.199] Epoch Runtime: 63ms
    [04/05/22 18:41:49.199] Nodes per Second: 34380.953
    [04/05/22 18:41:49.199] Evaluating validation set
    [04/05/22 18:41:49.213]
    =================================
    Node Classification: 270 nodes evaluated
    Accuracy: 12.962963%
    =================================
    [04/05/22 18:41:49.213] Evaluating test set
    [04/05/22 18:41:49.221]
    =================================
    Node Classification: 272 nodes evaluated
    Accuracy: 16.176471%
    =================================

After running this configuration for 10 epochs, we should see a result similar to below with arruracy roughly equal to 86%:

.. code-block:: bash

    =================================
    [04/05/22 18:41:49.820] ################ Starting training epoch 10 ################
    [04/05/22 18:41:49.833] Nodes processed: [1000/2166], 46.17%
    [04/05/22 18:41:49.854] Nodes processed: [2000/2166], 92.34%
    [04/05/22 18:41:49.872] Nodes processed: [2166/2166], 100.00%
    [04/05/22 18:41:49.872] ################ Finished training epoch 10 ################
    [04/05/22 18:41:49.872] Epoch Runtime: 51ms
    [04/05/22 18:41:49.872] Nodes per Second: 42470.59
    [04/05/22 18:41:49.872] Evaluating validation set
    [04/05/22 18:41:49.883]
    =================================
    Node Classification: 270 nodes evaluated
    Accuracy: 84.814815%
    =================================
    [04/05/22 18:41:49.883] Evaluating test set
    [04/05/22 18:41:49.891]
    =================================
    Node Classification: 272 nodes evaluated
    Accuracy: 88.970588%
    =================================

Let's check again what was added in the ``datasets/custom_nc_example/cora/`` directory. For clarity, we only list the files that were created in training. Notice that several files have been created, including the trained model, the embedding table, a full configuration file, and output logs:

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
