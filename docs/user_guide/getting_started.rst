.. _getting_started:

***************
Getting Started
***************

.. _getting_started_introduction:

Introduction
============
Marius is a system that allows users to train graph embedding models 
over large-scale data on a single machine without the need to write code. 
Marius uses PyTorch as the underlying tensor engine.

To train a graph embedding model, a Marius Configuration file and the data are the only 
things you need to provide. Simple commands can be used to preprocess the dataset,
deploy Marius, postprocess the trained models and embeddings, and perform link prediction.

Users can also use Marius via a Python API. It can work seamlessly with existing popular
machine learning libraries such as PyTorch.

Marius offers a set of model architectures that users can combine to choose 
a suitable model that fits their needs. Different model architectures
can be chosen via the Marius configuration file.


.. _getting_started_installation
Installation
============

Requirements
^^^^^^^^^^^^

(Other versions may work, but are untested)

* Ubuntu 18.04 or macOS 10.15
* CUDA 10.1 ro 10.2 (If using GPU training)
* CuDNN 7 (If using GPU training)
* 1.7 >= pytorch
* python >=3.6
* pip >= 21
* GCC >= 9 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8

Installation from source with Pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Install latest version of PyTorch for your CUDA version:

    Linux:

    * CUDA 10.1: ``python3 -m pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html``
    * CUDA 10.2: ``python3 -m pip install torch==1.7.1``
    * CPU Only: ``python3 -m pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html``

    MacOS:

    * CPU Only: ``python3 -m pip install torch==1.7.1``

#. Clone the repository ``git clone https://github.com/marius-team/marius.git``
#. Build and install Marius ``cd marius; python3 -m pip install .``

Full script (without torch install)
"""""""""""""""""""""""""""""""""""

::

    git clone https://github.com/marius-team/marius.git
    cd marius
    python3 -m pip install .


Installation from source with CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Clone the repository ``git clone https://github.com/marius-team/marius.git``
#. Install dependencies ``cd marius; python3 -m pip install -r requirements.txt``
#. Create build directory ``mkdir build; cd build``
#. Run cmake in the build directory ``cmake ../`` (CPU-only build) or ``cmake ../ -DUSE_CUDA=1`` (GPU build)
#. Make the Marius executable ``make marius_train -j``

Full script (without torch install)
"""""""""""""""""""""""""""""""""""

::

    git clone https://github.com/marius-team/marius.git
    cd marius
    python3 -m pip install -r requirements.txt
    mkdir build
    cd build
    cmake ../ -DUSE_CUDA=1
    make -j

Main Functionality
==================
Marius offers four functionalities:

* train graph embedding models
* evaluate the trained embeddings/models
* perform link prediction based on trained embeddings/models

For out-of-the-box deployment, Marius adopts a configuration-based programming paradigm. 
Users are only required to provide a configuration file and dataset as input.
There are currently over ninety configurable parameters divided into nine main sections,
including model, storage, training, pipelining and evaluation, that can define the 
training pipeline and model architectures. 

For basic functionality, all you need to do is specifying the statistics of the dataset,
such as number of edges, number of nodes, number of edge types, 
location of the data and the device for training (CPU/GPU). 
Marius can train a graph embedding model for you. 
For users who want more advanced functionality of the system, there are additional
configuration parameters available for them to tune based on their needs.


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
    the created configuration file.
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

    Refer to :ref:`Examples<Examples>` to find out how to perform link prediction task.


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