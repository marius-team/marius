.. _quickstart

Getting Started
=========================

Build and Install
##############################

Requirements
****************************
* CUDA >= 10.1
* CuDNN >= 7
* pytorch >= 1.8
* python >= 3.7
* GCC >= 7 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8

Pip installation
****************************

First check that the required software is installed (see above).

    .. code-block:: bash

        git clone https://github.com/marius-team/marius.git
        pip3 install .

The Python API can be accessed with ``import marius``.

The following commands will be installed:
- marius_train: Train models using configuration files and the command line
- marius_eval: Command line model evaluation
- marius_preprocess: Built-in dataset downloading and preprocessing
- marius_predict: Batch inference tool for link prediction or node classification


CMake build (No Python API)
****************************

This does not build the Python API, but only the C++ sources and marius_train executable.

    .. code-block:: bash

        git clone https://github.com/marius-team/marius.git

        # installs only marius.tools (required)
        MARIUS_NO_BINDINGS=1 pip3 install .

        mkdir build
        cd build
        cmake ../ -DUSE_CUDA=1
        make marius_train -j
        cd ..

        # run with build/marius_train config.yaml


Configuration Interface
##############################

.. _config_examples_link: http://marius-project.org/marius/examples/config/index.html
.. _schema_link: http://marius-project.org/marius/config_interface/full_schema.html

See configuration `examples <config_examples_link_>`_ for detailed examples and the `configuration schema <schema_link_>`_ for all options.

Preprocess & Configuration
****************************


Preprocess dataset: this downloads and preprocesses the dataset into the arxiv_example/ directory


    .. code-block:: bash

        marius_preprocess --dataset ogbn_arxiv --output_dir arxiv_example/




Define configuration file: 1-layer GraphSage GNN

    .. code-block:: yaml

        model:
          learning_task: NODE_CLASSIFICATION
          encoder:
            train_neighbor_sampling:
              - type: ALL
            layers:
              - - type: FEATURE
                  output_dim: 128
              - - type: GNN
                  options:
                    type: GRAPH_SAGE
                    aggregator: MEAN
                  input_dim: 128
                  output_dim: 40
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
          device_type: cuda
          dataset:
            dataset_dir: arxiv_example/
            num_edges: 1166243
            num_train: 90941
            num_nodes: 169343
            num_relations: 1
            num_valid: 29799
            num_test: 48603
            node_feature_dim: 128
            num_classes: 40
          edges:
            type: DEVICE_MEMORY
            options:
              dtype: int
          features:
            type: DEVICE_MEMORY
            options:
              dtype: float
        training:
          batch_size: 1000
          num_epochs: 10
          pipeline:
            sync: true
        evaluation:
          batch_size: 1000
          pipeline:
            sync: true


Training
****************************


    Train the model described in the configuration file for 10 epochs.

    .. code-block:: bash

        marius_train arxiv_config.yaml

    The output will look similar to:

    .. code-block:: text

        [04/08/22 01:12:10.693] ################ Starting training epoch 1 ################
        [04/08/22 01:12:10.721] Nodes processed: [10000/90941], 11.00%
        [04/08/22 01:12:10.741] Nodes processed: [20000/90941], 21.99%
        [04/08/22 01:12:10.762] Nodes processed: [30000/90941], 32.99%
        [04/08/22 01:12:10.800] Nodes processed: [40000/90941], 43.98%
        [04/08/22 01:12:10.820] Nodes processed: [50000/90941], 54.98%
        [04/08/22 01:12:10.840] Nodes processed: [60000/90941], 65.98%
        [04/08/22 01:12:10.863] Nodes processed: [70000/90941], 76.97%
        [04/08/22 01:12:10.883] Nodes processed: [80000/90941], 87.97%
        [04/08/22 01:12:10.916] Nodes processed: [90000/90941], 98.97%
        [04/08/22 01:12:10.918] Nodes processed: [90941/90941], 100.00%
        [04/08/22 01:12:10.918] ################ Finished training epoch 1 ################
        [04/08/22 01:12:10.918] Epoch Runtime: 224ms
        [04/08/22 01:12:10.918] Nodes per Second: 405986.6
        [04/08/22 01:12:10.918] Evaluating validation set
        [04/08/22 01:12:11.005]
        =================================
        Node Classification: 29799 nodes evaluated
        Accuracy: 58.669754%
        =================================
        [04/08/22 01:12:11.005] Evaluating test set
        [04/08/22 01:12:11.133]
        =================================
        Node Classification: 48603 nodes evaluated
        Accuracy: 57.936753%
        =================================
        ...



Inference
****************************

    Evaluate the test set for the dataset after 10 epochs have completed.

    .. code-block:: bash

        marius_eval arxiv_config.yaml


    Output:

    .. code-block:: text

        [04/08/22 02:06:25.330] Evaluating test set
        [04/08/22 02:06:25.585]
        =================================
        Node Classification: 48603 nodes evaluated
        Accuracy: 64.963068%
        =================================


Python API
##############################


.. _python_examples_link: http://marius-project.org/marius/examples/python/index.html

.. _python_api_link: http://marius-project.org/marius/python_api/index.html

See the `Python examples <python_examples_link_>`_ and `API docs <python_api_link_>`_ (under construction) for more details.

Preprocess Dataset and load graph data
**************************************

Import marius and preprocess ogbn_arxiv for node classifcation.

    .. code-block:: python

        import marius as m
        import torch
        from marius.tools.preprocess.datasets.ogbn_arxiv import OGBNArxiv

        # initialize and preprocess dataset
        dataset = OGBNArxiv("arvix_example/")
        dataset.download()
        dataset_stats = dataset.preprocess()

Load dataset tensors into GPU memory

    .. code-block:: python

        device = torch.device("cuda")

        edges = m.storage.tensor_from_file(filename=dataset.edge_list_file,
                                           shape=[dataset_stats.num_edges, -1],
                                           dtype=torch.int32,
                                           device=device)
        train_nodes = m.storage.tensor_from_file(filename=dataset.train_nodes_file,
                                                 shape=[dataset_stats.num_train],
                                                 dtype=torch.int32,
                                                 device=device)
        test_nodes = m.storage.tensor_from_file(filename=dataset.test_nodes_file,
                                                shape=[dataset_stats.num_test],
                                                dtype=torch.int32,
                                                device=device)
        features = m.storage.tensor_from_file(filename=dataset.node_features_file,
                                              shape=[dataset_stats.num_nodes, -1],
                                              dtype=torch.float32,
                                              device=device)
        labels = m.storage.tensor_from_file(filename=dataset.node_labels_file,
                                            shape=[dataset_stats.num_nodes],
                                            dtype=torch.int32,
                                            device=device)

Define Model
****************************

Define single layer graph sage model

    .. code-block:: python

        feature_dim = dataset_stats.node_feature_dim
        num_classes = dataset_stats.num_classes

        feature_layer = m.nn.layers.FeatureLayer(dimension=feature_dim,
                                                 device=device)

        graph_sage_layer = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                      output_dim=num_classes,
                                                      device=device)

        encoder = m.encoders.GeneralEncoder(layers=[[feature_layer],
                                                    [graph_sage_layer]])

        decoder = m.nn.decoders.node.NoOpNodeDecoder()
        loss = m.nn.CrossEntropyLoss(reduction="sum")

        reporter = m.report.NodeClassificationReporter()
        reporter.add_metric(m.report.CategoricalAccuracy())

        model = m.nn.Model(encoder, decoder, loss, reporter)
        model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(), lr=.01)]

        nbr_sampler = m.data.samplers.LayeredNeighborSampler(num_neighbors=[-1])

Training and Evaluation
****************************

Setup training and evaluation dataloaders

    .. code-block:: python

        train_loader = m.data.DataLoader(edges=edges,
                                         batch_size=1000
                                         nodes=train_nodes,
                                         nbr_sampler=nbr_sampler,
                                         learning_task="nc")

        eval_loader = m.data.DataLoader(edges=edges,
                                        batch_size=1000
                                        nodes=test_nodes,
                                        nbr_sampler=nbr_sampler,
                                        learning_task="nc)


Train 10 epochs

    .. code-block:: python

        num_epochs = 10
        for i in range(num_epochs)

            train_loader.initializeBatches()
            while train_loader.hasNextBatch():
                batch = train_loader.getBatch()
                model.train_batch(batch)

Evaluate Test Set

    .. code-block:: python

        eval_loader.initializeBatches()
        while eval_loader.hasNextBatch():
            batch = eval_loader.getBatch()
            model.evaluate_batch(batch)

        model.reporter.report()

