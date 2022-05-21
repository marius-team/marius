Resume Training (FB15K-237)
---------------------------------------------

In this tutorial, we use the **FB15K_237 knowledge graph** as an example to demonstrate the resume training functionality available in Marius using the :doc:`fb15k_237 <../examples/config/lp_fb15k237>` example. 

Using ``marius_preprocess``, we pre-process the data to make it available under path ``datasets/fb15k_237_rt``

.. code-block:: bash

   $ marius_preprocess --dataset fb15k_237 --output_directory datasets/fb15k_237_rt/
   Downloading FB15K-237.2.zip to datasets/fb15k_237_rt/FB15K-237.2.zip
   Reading edges
   Remapping Edges
   Node mapping written to: datasets/fb15k_237_rt/nodes/node_mapping.txt
   Relation mapping written to: datasets/fb15k_237_rt/edges/relation_mapping.txt
   Dataset statistics written to: datasets/fb15k_237_rt/dataset.yaml

Train the model at least once before trying to resume training.

.. code-block:: bash

   $ marius_train fb15k_237_config.yaml
   [05/06/22 18:08:21.037] ################ Finished training epoch 10 ################
   ...
   $ ls datasets/fb15k_237_rt/
   README.txt
   dataset.yaml
   edges
   model_0
   nodes

The current model parameters are present in ``datasets/fb15k_237_rt/model_0``


1. Resume training and overwrite existing model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Assuming the model is trained at least once, ``training.resume_training`` can be set to ``true`` to train the previously saved model further for n epochs (default 10). 

.. code-block:: yaml

   training:
     batch_size: 1000
     num_epochs: 10
     resume_training: true

Running ``marius_train`` with the updated config will now overwrite the model parameters in ``datasets/fb15k_237_rt/model_0/``

.. code-block:: bash

   $ marius_train fb15k_237_config.yaml
   [05/06/22 18:13:41.662] ################ Starting training epoch 11 ################
   ...
   [05/06/22 18:13:59.233] ################ Finished training epoch 20 ################
   ...


2. Resume training from given checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``training.resume_from_checkpoint`` can be set to preserve the existing checkpointed model and write the new model to a different directory. 
If ``storage.model_dir`` is set, the new model will be written to the given directory, else a new directory of the pattern ``datasets/fb15k_237_rt/model_x``
will be created where `x` changes incrementally from 0-10 and will take the least non-existent value. 

.. code-block:: bash

   $ ls datasets/fb15k_237_rt/
   README.txt
   dataset.yaml
   edges
   model_0
   nodes

Resuming training from the above config with ``training.resume_from_checkpoint`` set will write the model to ``datasets/fb15k_237_rt/model_1`` if 
``storage.model_dir`` is not set. Since ``datasets/fb15k_237_rt/model_0`` now has a model trained for 20 epochs, the new model will further be 
trained 10 epochs from there.

.. code-block:: yaml

   training:
     batch_size: 1000
     num_epochs: 10
     resume_training: true
     resume_from_checkpoint: datasets/fb15k_237_rt/model_0/

Running ``marius_train`` with the updated config will save the new model parameters to ``datasets/fb15k_237_rt/model_1/``

.. code-block:: bash

   $ marius_train fb15k_237_config.yaml
   [05/06/22 18:13:41.662] ################ Starting training epoch 21 ################
   ...
   [05/06/22 18:13:59.233] ################ Finished training epoch 30 ################
   ...
   $ ls datasets/fb15k_237_rt/
   README.txt
   dataset.yaml
   edges
   model_0
   nodes
