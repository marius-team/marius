.. _evaluation:

*************
Evaluation
*************

Evaluation During Training
--------------------------

If a validation and/or test is defined in the configuration file passed to ``marius_train``, then ranking metrics such as MRR and hits@k will be computed after each epoch on the validation set, and then after all epochs are complete evaluation will be performed on the test set.

For example, here is the output after training a single epoch wth ``marius_train``. First a single epoch is trained, then the embeddings are evaluated on the validation set according the evaluation configuration parameters.

::

    [info] [03/18/21 01:33:18.778] Metadata initialized
    [info] [03/18/21 01:33:18.778] Training set initialized
    [info] [03/18/21 01:33:18.779] Evaluation set initialized
    [info] [03/18/21 01:33:18.779] Preprocessing Complete: 2.605s
    [info] [03/18/21 01:33:18.791] ################ Starting training epoch 1 ################
    [info] [03/18/21 01:33:18.836] Total Edges Processed: 40000, Percent Complete: 0.082
    [info] [03/18/21 01:33:18.862] Total Edges Processed: 80000, Percent Complete: 0.163
    [info] [03/18/21 01:33:18.892] Total Edges Processed: 120000, Percent Complete: 0.245
    [info] [03/18/21 01:33:18.918] Total Edges Processed: 160000, Percent Complete: 0.327
    [info] [03/18/21 01:33:18.944] Total Edges Processed: 200000, Percent Complete: 0.408
    [info] [03/18/21 01:33:18.970] Total Edges Processed: 240000, Percent Complete: 0.490
    [info] [03/18/21 01:33:18.996] Total Edges Processed: 280000, Percent Complete: 0.571
    [info] [03/18/21 01:33:19.021] Total Edges Processed: 320000, Percent Complete: 0.653
    [info] [03/18/21 01:33:19.046] Total Edges Processed: 360000, Percent Complete: 0.735
    [info] [03/18/21 01:33:19.071] Total Edges Processed: 400000, Percent Complete: 0.816
    [info] [03/18/21 01:33:19.096] Total Edges Processed: 440000, Percent Complete: 0.898
    [info] [03/18/21 01:33:19.122] Total Edges Processed: 480000, Percent Complete: 0.980
    [info] [03/18/21 01:33:19.130] ################ Finished training epoch 1 ################
    [info] [03/18/21 01:33:19.130] Epoch Runtime (Before shuffle/sync): 339ms
    [info] [03/18/21 01:33:19.130] Edges per Second (Before shuffle/sync): 1425197.8
    [info] [03/18/21 01:33:19.130] Edges Shuffled
    [info] [03/18/21 01:33:19.130] Epoch Runtime (Including shuffle/sync): 339ms
    [info] [03/18/21 01:33:19.130] Edges per Second (Including shuffle/sync): 1425197.8
    [info] [03/18/21 01:33:19.148] Starting evaluating
    [info] [03/18/21 01:33:19.254] Pipeline flush complete
    [info] [03/18/21 01:33:19.271] Num Eval Edges: 50000
    [info] [03/18/21 01:33:19.271] Num Eval Batches: 50
    [info] [03/18/21 01:33:19.271] Auc: 0.973, Avg Ranks: 24.477, MRR: 0.491, Hits@1: 0.357, Hits@5: 0.651, Hits@10: 0.733, Hits@20: 0.806, Hits@50: 0.895, Hits@100: 0.943

Evaluation After Training
--------------------------

To evaluate previously trained embeddings on a test set ``marius_eval`` can be used. See :ref:`user_guide_marius_eval`.

Configuring for Evaluation
--------------------------

The following parameters are the main ones of interest for evaluation.  See :ref:`configuration` for a full list of parameters and details for each parameter.

::

    [evaluation]
    negatives=1000
    evaluation_method=LinkPrediction
    filtered_evaluation=false


This set of options states that the score of each edge in the test set will be ranked against the scores of 1000 negative samples for the link prediction tasks.

However, these negative samples may be false negatives (i.e true edges in the graph) and thus may score highly. To filter out false negatives filtered_evaluation can be set to True. However this does not scale well to large datasets, so it is recommended to keep this disabled.