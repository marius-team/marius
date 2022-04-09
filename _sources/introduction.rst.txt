.. _introduction

Introduction
=========================

Marius is a system for scaling graph learning on a single machine. Marius supports training and evaluation of GNNs and graph embedding models for link prediction or node classification. See our papers Marius and Marius++ for technical details.

Feature Overview
##############################


* **Billion scale** link prediction and node classification training and evaluation
* **High performance** configuration-file based execution
* **PyTorch compatible** Python API for custom training and evaluation routines


.. container:: twocol

    .. container:: leftside

        Define 3-layer GraphSage model in Python

        ::

            nbr_sampler = m.nn.LayeredNeighborSampler([-1, -1, -1])

            feat_dim = 128
            num_classes = 40

            device = torch.device("cuda")

            feat_layer = m.nn.layers.FeatureLayer(dimension=feature_dim,
                                                  device=device)

            gs_layer1 = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                   output_dim=feature_dim,
                                                   device=device)

            gs_layer2 = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                   output_dim=feature_dim,
                                                   device=device)

            gs_layer3 = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                   output_dim=num_classes,
                                                   device=device)

            encoder = m.encoders.GeneralEncoder(layers=[[feature_layer],
                                                        [graph_sage_layer1],
                                                        [graph_sage_layer2],
                                                        [graph_sage_layer3]])

            decoder = m.nn.decoders.node.NoOpNodeDecoder()
            loss = m.nn.CrossEntropyLoss(reduction="sum")

            model = m.nn.Model(encoder, decoder, loss)
            model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(),
                                                   lr=.01)]

    .. container:: rightside

        or with YAML configuration
        ::

            model:
              learning_task: node_classification
              encoder:
                train_neighbor_sampling:
                  - type: all
                  - type: all
                  - type: all
                layers:
                  - - type: feature
                      output_dim: 128
                  - - type: gnn
                      options:
                        type: graph_sage
                      input_dim: 128
                      output_dim: 128
                  - - type: GNN
                      options:
                        type: graph_sage
                      input_dim: 128
                      output_dim: 128
                  - - type: gnn
                      options:
                        type: graph_sage
                      input_dim: 128
                      output_dim: 40
              decoder:
                type: node
              loss:
                type: cross_entropy
                options:
                  reduction: sum
              dense_optimizer:
                type: adam
                options:
                  learning_rate: 0.01


Preprocessing
""""""""""""""""""""

* Performant dataset preprocessing of raw datasets in CSV format
* 13 built-in datasets for link prediction or node classification
* Custom dataset support

Training & Evaluation
"""""""""""""""""""""""

* CPU-GPU pipeline to mitigate data movement overheads
* Optimized neighborhood sampling and datastructures for GNN aggregation
* Scale beyond CPU memory with a partition buffer

Supported Input Graphs
"""""""""""""""""""""""

* Formats: CSV/TSVs, PyTorch tensors, Numpy arrays
* Graphs with or without edge-types or node features
* Scales to graphs with billions of edges and 100s of millions of nodes

Supported Models
"""""""""""""""""""""""

* Tasks: Link prediction, node classification
* GNN layers: GraphSage, GCN, RGCN, GAT
* Link prediction decoders: ComplEx, DistMult, TransE

Upcoming Features
##############################

* Configuration file optimizer and generator (in testing)
* SQL database to graph conversion tool (in testing)
* Multi-GPU training (in progress)
* Model checkpointing (in progress)
* KNN inference module
* marius_preprocess parquet file support
* Remote storage for graph data and embeddings
* Additional encoder layers and decoder layers