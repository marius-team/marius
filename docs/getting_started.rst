.. _getting_started:

***************
Getting Started
***************

.. _getting_started_introduction:
Introduction
============
Marius is a toolbox that allows users to train and evaluate graph embedding models 
over large-scale data on a single machine without the need to write code. 
It is build on top of PyTorch.

To train a graph embedding model, a Marius Configuration file and the data are the only 
things you need to provide. Simple commands can be used to preprocess the dataset,
deploy Marius, postprocess the trained models and embeddings, and make link prediction.

Users can also use Marius as a Python API. It can work seamlessly with existing popular
machine learning libraries such as PyTorch.

Practitioners can use Marius to conveniently train and evaluate graph embedding models, 
while researchers can use Marius for experiment settings that is compatible with
data in various formats and comparable in terms of same data processing and evaluation process.

Marius offers a set of model architectures options that users can combine to form 
the suitable model based on their needs. All these choices of model architectures
can be performed by turning the Marius configuration file.

The core design principles for Marius are:

* **No code paradigm for graph embedding pipelines**
  
    The user of configuration based programming allows users to run Marius without 
    ever having to write a line of code. Only commands in terminal is required to deploy
    a graph embedding pipeline including the dataset for training, the data storage,
    the execution hardware, model, hyperparameters etc.

* **Seamless integration with popular machine learning libraries**
  
    Marius' API can be used seamlessly with other popular machine learning libraries,
    such as PyTorch, to define the training pipeline of a graph embedding model.



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
================
Marius offers the users three main functionalities:

* preprocess datasets
* train graph embedding models
* evaluate the trained embeddings/models
* perform link prediction based on trained embeddings/models

For out-of-the-box deployment Marius adopts a configuration-based programming paradigm. 
Users are only required to provide a configuration file and dataset as input.
There are currently over 90 configurable parameters divided into 9 main sections,
including model, storage, training, pipelining and evaluation, that can define the 
training pipeline and model architectures. 

For basic functionality, all you need to do is to specify the statistics of the dataset,
such as number of edges, number of nodes, number of edge types, location of the data and the device for training (CPU/GPU). 
Marius can put together and train a graph embedding model for you. 
For users who want more advanced functionality of the system, there are additional
configuration parameters available for them to tune based on their needs.

Currently, the available graph formats are:   (**is this section necessary?**)

* graphs with multiple edge types
* graphs with only one edge type


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

#. Define a configuration file ``config_gpu.ini``.

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
    If you want to specify the model for train to be ``TransE`` and increase the 
    train epochs to a certain number, you can simply add the following lines to the
    configuration file:

    ::

        [model]
        decoder = TransE

        [training]
        num_epochs = 100

    See :ref:`User Guide<User_Guide>` for full details on the configuration options.

#. Proprocess the dataset ``marius_preprocess <dataset> output_dir/``

    The first argument of ``marius_preprocess`` defines the dataset we want to preprocess.
    The second argument tells ``marius_preprocess`` where to put the preprocessed dataset.

#. Run the training executable with the Marius configuration file. 

    ::

        marius_train config_gpu.ini

    Refer to :ref:`Examples<Examples>` to find out how to perform link prediction task on
    the 2 supported formats of graphs.


Evaluation
^^^^^^^^^^

Marius prints out training progress and evaluation information to the terminal during the training.
After training, Marius also creates a ``data/`` directory to store all the trained embeddings,
models, and evaluation statistics. You can utilize these statistics for evaluation.


Prediction
^^^^^^^^^^

Marius provides ``marius_postprocess`` for user to perform link prediction task
on trained embeddings and models. Given the source and type of relation, the 
following command would give the most matched few destinations:

::
    
    marius_postprocess __saxony_NN_1 _member_meronym 

Checkout the :ref:`User Guide<User_Guide>` for more detailed usage of ``marius_postprocess``.



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

Marius can be imported as a Python library. While Marius already comes equipped 
with a number of commonly used models and functions, advanced users can implement 
their own custom models in Python and use them for the training process. These
models can then be used in the training process by setting the associated model decoder
parameter.
Refer to the :ref:`Developer Guide<Developer_Guide>` for full details about extending
Marius to custom models.