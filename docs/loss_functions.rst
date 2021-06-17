.. _loss_functions:

Loss Functions
********************

Multiple loss functions are supported by the system and can be set in the :ref:`[loss]<loss_option>` section of the configuration files.

==================  =============
   Name             Description
------------------  -------------
SoftMaxLoss         A loss function which gives the `cross entropy`_ loss after softmax output
RankingLoss         A loss function which gives the `margin ranking loss`_
BCEAfterSigmoid     A loss function which applies a sigmoid layer before `binary cross entropy`_ loss. This is more numerically unstable than BCEWithLogitsLoss.
BCEWithLogitsLoss   A loss function which combines a `sigmoid layer and the binary cross entropy`_ loss
MSELoss             A loss function which gives the  `mean square error loss`_
SoftPlusLoss        A loss function which gives the `softplus`_ loss
==================  =============

.. _cross entropy: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1functional_1a29daa086ce1ac3cd9f80676f81701944.html
.. _margin ranking loss: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1functional_1a14df0f947a85c31fbc40820f04013255.html?highlight=functional%20margin_ranking_loss
.. _binary cross entropy: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1functional_1a36c9877fdc4730b6adfb32c34e708e9c.html?highlight=binary_cross_entropy
.. _sigmoid layer and the binary cross entropy: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1functional_1a0e88f9c4d5549d413457a2ab47663b70.html?highlight=binary_cross_entropy_with_logits
.. _mean square error loss: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1functional_1aba6a5341298f43163c046b3dd50dc6ce.html?highlight=functional%20mse_loss
.. _softplus: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1functional_1a2ed3b13bed41c0ea148ce50a4678f667.html?highlight=functional%20softplus