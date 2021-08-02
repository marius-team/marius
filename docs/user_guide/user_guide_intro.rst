.. _User_guide_intro:

****************
User Guide Intro
****************

User Guide
==========

This is the Marius user guide. It helps users figure out how to deploy Marius,
what configuration parameters are available for graph embedding model architectures.
Below are the links to each section of the guide.

Basic workflow
=====================

1. Preprocess input dataset using ``marius_preprocess`` See :ref:`preprocessing`.
2. Create a configuration file which denotes the dataset, model, trainingm and evaluation process. See. :ref:`configuration`.
3. Train and evaluate with ``marius_train`` and ``marius_eval``. See :ref:`training` :ref:`evaluation`.
4. Convert trained embeddings to output format with ``marius_postprocess`` See :ref:`postprocessing`.
5. Perform downstream task with embeddings. E.g edge/link prediction with ``marius_predict``. See :ref:`prediction`

Dataset Preprocessing
=====================

Marius requires converting the input dataset into an internal binary representation. Marius supports converting datasets that are an edge list format stored as a CSV, TSV or other type of delimited file. Other formats will need to be converted with a custom converter.
Details on the preprocessing and how to write a customer converter can be found at :ref:`preprocessing`.

Configuration
=============

Training and evaluating a dataset requires defining a configuration file which denotes the model, hyperparameters, training, evaluation, and system parameters. Information on configuration options is found here :ref:`configuration`.

Training
========

Marius supports efficient training of datasets and configurations without being limited to GPU and CPU memory sizes. For more information on training small, medium and large scale graphs, see :ref:`training`.

Evaluation
==========

Marius supports evaluation during training and after training with ``marius_eval``. See :ref:`evaluation`.

Postprocessing
==============

After training, embeddings are stored in a file as a dump of tensor memory, which can be loaded using ``np.from_file`` or ``torch.load``. With ``marius_postprocess`` we provide conversion from this format to other formats such as TSV or CSV. See :ref:`postprocessing`.

Downstream Tasks
================

The trained embeddings can be used for a variety of downstream tasks such as link prediction or nearest neighbor search. For these tasks we provide the ``marius_predict`` tool. See :ref:`prediction`.