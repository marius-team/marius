.. _marius_predict:

Batch Inference (marius_predict)
==================================================

This document contains an overview of the inference module for link prediction and node classification models trained using the configuration API. The module supports both in memory and disk-based inference.

The input test set can be be preprocessed, or can be in the raw input format and then preprocessed (partitioned, remapped and converted to binary format) before input to evaluation.

Link Prediction
##############################

Input
**********

- A configuration file for a previously trained link prediction model

- A set of test edges (preprocessed or unpreprocessed)

- A list of metrics to compute (optional)

- Negative sampling configuration (optional)

Output
****************************

Text file containing a summary of metrics for the evaluation set: ``<output_dir>/metrics.txt`` (optional)

CSV file where each row denotes an edge, and it’s corresponding score and link prediction rank ``<output_dir>/scores.csv`` (optional)

Example Usage
****************************

    .. code-block:: bash

        marius_predict --config configs/fb15k237.yaml --metrics mrr mr hits3 hits5 hits10 hits50 hits100 hits2129 --save_ranks --save_scores --output_dir results/`

This command takes in a trained configuration file, ``configs/fb15k237.yaml``, which defines a model that has been previous trained.

The list of metrics over the training set will be computed and output to results/metrics.txt. The ranks and scores for each edge are output to ``results/scores.csv``.

Contents of ``configs/fb15k237.yaml``. The test set here has been created during preprocessing and is stored in ``<storage.dataset.dataset_dir>/edges/test_edges.bin``

    .. code-block:: yaml

        model:
          learning_task: LINK_PREDICTION
          encoder:
            layers:

              - - type: EMBEDDING
                  output_dim: 10
                  bias: true
                  init:
                    type: GLOROT_NORMAL

          decoder:
            type: DISTMULT
          loss:
            type: SOFTMAX_CE
            options:
              reduction: SUM
          dense_optimizer:
            type: ADAM
            options:
              learning_rate: 0.01
          sparse_optimizer:
            type: ADAGRAD
            options:
              learning_rate: 0.1

        storage:
          device_type: cpu
          dataset:
            dataset_dir: ./fb15k_237_example/
          edges:
            type: HOST_MEMORY
            options:
              dtype: int
          embeddings:
            type: HOST_MEMORY
            options:
              dtype: float
        training:
          batch_size: 1000
          negative_sampling:
            num_chunks: 10
            negatives_per_positive: 10
            degree_fraction: 0
            filtered: false
          num_epochs: 10
          pipeline:
            sync: true
        evaluation:
          batch_size: 1000
          negative_sampling:
            filtered: true
          pipeline:
            sync: true

Since ``storage.model_dir`` is not specified in the above configuration, ``marius_predict`` will use the latest trained model present in ``storage.dataset.dataset_dir``.
When ``storage.model_dir`` is not specified, ``marius_train`` stores the model parameters in `model_x` directory within the `storage.dataset.dataset_dir`, where x changes 
incrementally from 0 - 10. A maximum of 11 models are stored when `model_dir` is not specified, post which the contents in `model_10/` directory are overwritten with the 
latest parameters. ``marius_predict`` will use the latest model for inference and save the files to that directory. If ``storage.model_dir`` is specified, the model 
parameters will be loaded from the given directory and the generated files will be saved to the same. 

Example output
****************************
Two files are output by the above command:


metrics.txt
    .. code-block:: text

        Link Prediction: 40932 edges evaluated
        MRR: 0.125147
        Mean Rank: 426.079766
        Hits@3: 0.156259
        Hits@5: 0.207148
        Hits@10: 0.285229
        Hits@50: 0.510383
        Hits@100: 0.598725
        Hits@2129: 0.947987


scores.csv
    .. code-block:: text

        src,rel,dst,rank,score
        14469,149,11486,26,32.206722
        8558,74,7904,2789,5.628761
        3160,73,8048,282,7.548909
        7240,168,4510,149,1.634745
        2393,211,10586,2,96.834641
        12773,198,5262,3136,9.098152
        11469,88,8946,18,15.922592
        2045,166,3344,289,0.407495


Input a new test set
****************************************

If the dataset does not have a predefined test set. (e.g. ``storage.dataset.num_test == 0``). Then users can specify a separate test set with the ``--input_file <path_to_test_set>``. This test set can either be preprocessed and in binary format, or unpreprocessed.

Preprocessed input_test set usage:

    .. code-block:: bash

        marius_predict --config configs/fb15k237.yaml --input_file test_edges.bin --metrics mrr --save_ranks --save_scores --output_dir results/

Unpreprocessed input_test set usage:

If the input test set is unpreprocessed and in some raw input format. Then the ``--preprocess_input`` flag can be given. Users will need to specify the format of their input with ``--input_format <format>``. Currently delimited formats are only supported.

    .. code-block:: bash

        marius_predict --config configs/fb15k237.yaml --input_file test_edges.csv --preprocess_input --input_format CSV --metrics mrr --save_ranks --save_scores --output_dir results/


Node Classification
##############################

Input
**********

A configuration file for a previously trained node classification model

A set of test nodes (preprocessed or unpreprocessed)

A list of metrics to compute (optional)

Output
**********

Text file containing a summary of metrics for the evaluation set: ``<output_dir>/metrics.txt`` (optional)

CSV file where each row denotes an node, and it’s corresponding node classification label ``<output_dir>/labels.csv`` (optional)

Example Usage
********************


    .. code-block:: bash

        marius_predict --config configs/arxiv.yaml --metrics accuracy --save_labels --output_dir results/

This command takes in a trained configuration file, ``configs/arxiv.yaml``, which defines the previously trained model.

The list of metrics over the training set will be computed and output to ``results/metrics.txt``. The ranks and scores for each node are output to ``results/labels.csv``.


Command line arguments
##############################

Below is the help message for the tool, containing an overview of the tools arguments and usage.


    .. code-block:: text

        $ marius_predict --help
        usage: predict [-h] --config config [--output_dir output_dir] [--metrics [metrics ...]] [--save_labels] [--save_scores] [--save_ranks] [--batch_size batch_size] [--num_nbrs num_nbrs]
                       [--num_negs num_negs] [--num_chunks num_chunks] [--deg_frac deg_frac] [--filtered filtered] [--input_file input_file] [--input_format input_format] [--preprocess_input preprocess_input]
                       [--columns columns] [--header_length header_length] [--delim delim] [--dtype dtype]

        Tool for performing link prediction or node classification inference with trained models.

        Link prediction example usage:
        marius_predict <trained_config> --output_dir results/ --metrics mrr mean_rank hits1 hits10 hits50 --save_scores --save_ranks
        Assuming <trained_config> contains a link prediction model, this command will perform link prediction evaluation over the test set of edges provided in the config file. Metrics are saved to results/metrics.csv and scores and ranks for each test edge are saved to results/scores.csv

        Node classification example usage:
        marius_predict <trained_config> --output_dir results/ --metrics accuracy --save_labels
        This command will perform node classification evaluation over the test set of nodes provided in the config file. Metrics are saved to results/metrics.csv and labels for each test node are saved to results/labels.csv

        Custom inputs:
        The test set can be directly specified setting --input_file <test_set_file>. If the test set has not been preprocessed, then --preprocess_input should be enabled. The default format is a binary file, but additional formats can be specified with --input_format.

        optional arguments:
          -h, --help            show this help message and exit
          --config config       Configuration file for trained model
          --output_dir output_dir
                                Path to output directory
          --metrics [metrics ...]
                                List of metrics to report.
          --save_labels         (Node Classification) If true, the node classification labels of each test node will be saved to <output_dir>/labels.csv
          --save_scores         (Link Prediction) If true, the link prediction scores of each test edge will be saved to <output_dir>/scores.csv
          --save_ranks          (Link Prediction) If true, the link prediction ranks of each test edge will be saved to <output_dir>/scores.csv
          --batch_size batch_size
                                Number of examples to evaluate at a time.
          --num_nbrs num_nbrs   Number of neighbors to sample for each GNN layer. If not provided, then the module will check if the output of the encoder has been saved after training (see
                                storage.export_encoded_nodes). If the encoder outputs exist, the the module will skip the encode step (incl. neighbor sampling) and only perform the decode over the saved
                                inputs. If encoder outputs are not saved, model.encoder.eval_neighbor_sampling will be used for the neighbor sampling configuration. If model.encoder.eval_neighbor_sampling does
                                not exist, then model.encoder.train_neighbor_sampling will be used.If none of the above are given, then the model is assumed to not require neighbor sampling.
          --num_negs num_negs   (Link Prediction) Number of negatives to compare per positive edge for link prediction. If -1, then all nodes are used as negatives. Otherwise, num_neg*num_chunks nodes will be
                                sampled and used as negatives. If not provided, the evaluation.negative_sampling configuration will be used.if evaluation.negative_sampling is not provided, then negative
                                sampling will not occur and only the scores for the input edges will be computed, this means that any ranking metrics cannot be calculated.
          --num_chunks num_chunks
                                (Link Prediction) Specifies the amount of reuse of negative samples. A given set of num_neg sampled nodes will be reused to corrupt (batch_size // num_chunks) edges.
          --deg_frac deg_frac   (Link Prediction) Specifies the fraction of the num_neg nodes sampled as negatives that should be sampled according to their degree. This sampling procedure approximates degree
                                based sampling by sampling nodes that appear in the current batch of edges.
          --filtered filtered   (Link Prediction) If true, then false negative samples will be filtered out. This is only supported when evaluating with all nodes.
          --input_file input_file
                                Path to input file containing the test set, if not provided then the test set described in the configuration file will be used.
          --input_format input_format
                                Format of the input file to test. Options are [BINARY, CSV, TSV, DELIMITED] files. If DELIMITED, then --delim must be specified.
          --preprocess_input preprocess_input
                                If true, the input file (if provided) will be preprocessed before evaluation.
          --columns columns     List of column ids of input delimited file which denote the src node, edge-type, and dst node of edges.E.g. columns=[0, 2, 1] means that the source nodes are found in the first
                                column of the file, the edge-types are found in the third column, and the destination nodes are found in the second column.For graphs without edge types, only the location node
                                columns need to be provided. E.g. [0, 1]If the input file contains node ids rather than edges, then only a single id is needed. E.g. [2]
          --header_length header_length
                                Length of the header for input delimited file
          --delim delim         Delimiter for input file
          --dtype dtype         Datatype of input file elements. Defaults to the dataset specified in the configuration file.