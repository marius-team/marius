.. _getting_started:

***************
Getting Started
***************

.. _getting_started_introduction:

Introduction
============
Marius is a system that allows users to train graph embedding models 
over large-scale data on a single machine without the need to write code. 

To train a graph embedding model, a configuration file and the dataset are needed. Simple command line tools can be used to preprocess the dataset,
train embeddings/models, postprocess the trained models and embeddings, and perform link prediction.

Marius also has an experimental Python API, allowing for definition of custom models and training procedures.

Marius offers a set of model architectures that users can combine to choose 
a suitable model that fits their needs. Different model architectures
can be chosen via the Marius configuration file.

Basic workflow
=====================

The basic workflow of training and using embeddings with Marius can be divided into the below steps, each with an associated command line tool.

1. Preprocess input dataset using ``marius_preprocess`` See :ref:`preprocessing`.
2. Create a configuration file which denotes the dataset, model, training and evaluation process. See. :ref:`configuration`.
3. Train and evaluate with ``marius_train`` and ``marius_eval``. See :ref:`training` :ref:`evaluation`.
4. Convert trained embeddings to output format with ``marius_postprocess`` See :ref:`postprocessing`.
5. Perform downstream task with embeddings. E.g edge/link prediction with ``marius_predict``. See :ref:`prediction`

Training a graph
^^^^^^^^^^^^^^^^

For example, given a graph dataset like the following:

==================  ==================  =======================
Source              Relation            Destination
------------------  ------------------  -----------------------
__wisconsin_NN_2    _instance_hypernym  __madison_NN_2
__scandinavia_NN_2  _member_meronym     __sweden_NN_1
__kobenhavn_NN_1    _instance_hypernym  __national_capital_NN_1
...                 ...                 ...
==================  ==================  =======================

Training embeddings on such a graph requires three steps:

#. Preprocess the dataset ``marius_preprocess output_dir/ --files custom_dataset.csv``

    ``output_dir`` defines the directory to save all the preprocessed data. 
    The option ``--files`` can be used to pass the files containing the custom dataset.

#. Define a configuration file ``config.ini``.

    ::

        [general]
        device=GPU
        num_train=141442
        num_nodes=40943
        num_relations=18
        num_valid=5000
        num_test=5000

        [path]
        base_directory=data/
        train_edges=./output_dir/train_edges.pt
        validation_edges=./output_dir/valid_edges.pt
        test_edges=./output_dir/test_edges.pt
        node_ids=./output_dir/node_mapping.bin
        relations_ids=./output_dir/rel_mapping.bin

    In this case, we choose to use GPU as the training device.
    The ``[path]`` section contains all the locations of preprocessed data.
    If you want to specify the model for training to be ``TransE`` and increase the 
    training epochs to a certain number, you can simply add the following lines to the
    configuration file:

    ::

        [model]
        decoder = TransE

        [training]
        num_epochs = 100

    See :ref:`User Guide<User_Guide>` for full details on the configuration options.

    Marius also offers ``marius_config_generator`` to generate a configuration file
    for the users given the basic information of dataset statistics and where to store
    the created configuration file. ``marius_config_generator`` can be used to generate
    configuration files for both custom and supported datasets by passing different
    options.
    All other configuration parameters will be set to the default value.
    Users are given the options to specify the values of certain parameters.
    The following command shows how to use ``marius_config_generator`` to generate 
    a Marius configuration file for the same dataset mention above.
    The generated config file is saved to the same directory for storing data.
    The value of ``embedding_size`` is changed to 512.

    ::

        marius_config_generator ./output_dir -s 40943 18 141442 5000 5000 --model.embedding_size=512

    See ::ref:`User Guide<User_Guide>` for full details on usage of ``marius_config_generator``.

#. Run the training executable with the Marius configuration file. 

    ::

        marius_train config.ini

Evaluation
^^^^^^^^^^

Marius prints out training progress and evaluation information to the terminal during the training.
After training, Marius also creates a ``data/`` directory to store all the trained embeddings,
models, and evaluation statistics. ``marius_eval`` can be used for evaluation the trained embeddings
and models.

Run the following command to perform evaluation. The ``config.ini`` is the same Marius configuration
file used for ``marius_train``.

::

    marius_eval config.ini


Prediction
^^^^^^^^^^

After the training task is completed, ``marius_postprocess`` and ``marius_predict``
can help retrieve the trained embeddings and perform link prediction tasks.

Marius provides ``marius_postprocess`` for users to retrieve the trained embeddings in the 
required format.
The following command retrieves the trained embeddings stored in ``./data/``
and store them in CSV format in
the directory ``embeddings/``. The directory ``./training_data/`` is the directory
containing the preprocessed data used for training.
Other data formats, such as TSV, PyTorch tensor 
are also supported by ``marius_postprocess``.
Users just need to replace ``CSV`` with name of the format they want in the following command.

::

    marius_postprocess ./data/ ./training_data/ --output_directory ./embeddings/ --format CSV

Link prediction on trained embeddings is supported by ``marius_predict``. 
Given a source node and type of relation, the 
following command returns the top-ten destinations nodes. 
Number of predicted destinations can be controlled by changing the number ``10`` in 
the command.

::
    
    marius_predict ./data/ ./training_data/ 10 --src __saxony_NN_1  --rel _member_meronym

The left-hand-side relation type is used since the link prediction performed in 
this case starts from source node to destination node.
User can also do batch inference by using a file as inference input.
Checkout the :ref:`User Guide<User_Guide>` for more detailed usage of ``marius_postprocess`` and ``marius_predict``.



Programmatic API
================

Marius also provides a programmatic API that could allow users to deploy training pipeline.

::

    import marius as m
    from marius.tools import preprocess

    def fb15k_example():

        preprocess.fb15k(output_dir="output_dir/")
        
        config_path = "examples/training/configs/fb15k_cpu.ini"
        config = m.parseConfig(config_path)

        train_set, eval_set = m.initializeDatasets(config)

        model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)

        trainer = m.SynchronousTrainer(train_set, model)
        evaluator = m.SynchronousEvaluator(eval_set, model)

        trainer.train(1)
        evaluator.evaluate(True)


    if __name__ == "__main__":
        fb15k_example()

``fb15k`` is the dataset would be trained over in this example.


Extensibility
=============

While Marius already comes equipped 
with a number of commonly used models and functions, advanced users can implement 
their own custom models in Python and use them for the training process. These
models can then be used in the training process by setting the associated model decoder
parameter.
Refer to the :ref:`Developer Guide<Developer_Guide>` for full details about extending
Marius to custom models.