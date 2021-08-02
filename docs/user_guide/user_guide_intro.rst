.. _User_guide_intro:

****************
User Guide Intro
****************

User Guide
==========

This guide helps users figure out how to train models with Marius.

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